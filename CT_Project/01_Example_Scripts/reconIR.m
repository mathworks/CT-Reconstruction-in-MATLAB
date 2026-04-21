function hybridIR_vol = reconIR(inputMat, fdkMat, varargin)
% RUN_IR  Hybrid IR refinement starting from FDK (GPU if available, else CPU).
%
% Usage:
%   vol = run_IR('CT_high.mat','FDK_high.mat');                     % auto GPU
%   vol = run_IR('CT_high.mat','FDK_high.mat','UseGPU',false);      % force CPU
%   vol = run_IR('CT_high.mat','FDK_high.mat','NumIter',20, ...
%                 'Alpha',0.03,'Lambda',0.01,'NumSamples',64);
%
% Name-Value (all optional):
%   'UseGPU'      : true | false | 'auto' (default 'auto')
%   'NumIter'     : default 15
%   'Alpha'       : step size (default 0.03)
%   'Lambda'      : TV-like Gaussian regularization weight (default 0.01)
%   'GaussSigma'  : sigma for imgaussfilt3 (default 1)
%   'PadFactor'   : passed to simpleFDK_CBCT_vox (default 2)
%   'BatchZ'      : passed to simpleFDK_CBCT_vox (default 16)
%   'NumSamples'  : forward projector samples per ray (default 64)
%   'BatchPixels' : forward projector pixels per batch (default 4096)
%   'Verbose'     : print progress (default true)
%
% Requirements:
%   - simpleFDK_CBCT_vox.m (unified auto-GPU from previous step)
%   - forwardProject_CBCT.m (this file)

    % -------- Parse options --------
    p = inputParser;
    addParameter(p, 'UseGPU', 'auto', @(x)islogical(x) || (ischar(x) && strcmpi(x,'auto')));
    addParameter(p, 'NumIter', 1, @(x)isscalar(x) && x>=1);
    addParameter(p, 'Alpha', 0.03, @(x)isscalar(x) && x>0);
    addParameter(p, 'Lambda', 0.01, @(x)isscalar(x) && x>=0);
    addParameter(p, 'GaussSigma', 1, @(x)isscalar(x) && x>=0);
    addParameter(p, 'PadFactor', 2, @(x)isscalar(x) && x>=1);
    addParameter(p, 'BatchZ', 16, @(x)isscalar(x) && x>=1);
    addParameter(p, 'NumSamples', 64, @(x)isscalar(x) && x>=2);
    addParameter(p, 'BatchPixels', 4096, @(x)isscalar(x) && x>=128);
    addParameter(p, 'Verbose', true, @(x)islogical(x) || isnumeric(x));
    parse(p, varargin{:});

    UseGPU      = p.Results.UseGPU;
    numIter     = p.Results.NumIter;
    alpha       = p.Results.Alpha;
    lambda      = p.Results.Lambda;
    sigG        = p.Results.GaussSigma;
    padFactor   = p.Results.PadFactor;
    batchZ      = p.Results.BatchZ;
    numSamples  = p.Results.NumSamples;
    batchPixels = p.Results.BatchPixels;
    verbose     = logical(p.Results.Verbose);

    % -------- Load data --------
    if nargin < 2 || isempty(fdkMat)
        error('Please provide both inputMat (CT_*.mat) and fdkMat (FDK_*.mat)');
    end
    S = load(inputMat);


    % -------- Resolve fdkMat: file OR object --------
    if ischar(fdkMat) || isstring(fdkMat)
        % Old behavior: fdkMat is a .mat file path
        FDKdat = load(string(fdkMat));
    else
        % New behavior: fdkMat is already the FDK result (volume/object)
        FDKdat = fdkMat;
    end



    % Projections & angles
    P          = S.P;                           % could be [views nv nu] or [nu nv views]
    angles_rad = S.angles_rad(:)';              % [1 x nViews]
    DSO        = S.DSO; DSD = S.DSD;

    % Detector geometry
    if isscalar(S.du), du = S.du; dv = S.du; else, du = S.du; dv = S.dv; end
    u0 = S.u0_pixels;  v0 = S.v0_pixels;

    % Make P as [nu nv nViews]
    if ndims(P)==3
        szP = size(P);
        % Heuristic: if first dim equals nViews, assume [views nv nu]
        if szP(1) == numel(angles_rad)
            P = permute(P, [3 2 1]);  % [nu nv views]
        elseif szP(3) == numel(angles_rad)
            % already [nu nv views]
        else
            error('Unexpected projection array shape. Got %s, nViews=%d', mat2str(szP), numel(angles_rad));
        end
    else
        error('P must be 3D.');
    end
    [nu, nv, nViews] = size(P); %#ok<ASGLU>

    % -------- Initial estimate from FDK --------
  % -------- Initial estimate from FDK --------
  if isstruct(FDKdat)
      if isfield(FDKdat,'FDKvol')
          x = FDKdat.FDKvol;
      elseif isfield(FDKdat,'recon_volume')
          x = FDKdat.recon_volume;
      else
          error('FDK MAT file must contain FDKvol or recon_volume.');
      end
  else
      % FDKdat is already a volume/object from reconFDK
      x = FDKdat;
  end
  [nx, ny, nz] = size(x);

    % -------- Voxel size (try to be consistent) --------
    % Prefer metadata if present; else infer FOV in x/y and slice thickness
    if isfield(S,'extra') && isfield(S.extra,'meta')
        if isfield(S.extra.meta,'attr_fov_mm')
            fov_xy_mm = double(S.extra.meta.attr_fov_mm);
            dx = fov_xy_mm / nx; dy = fov_xy_mm / ny;
        else
            dx = du; dy = dv;  % fallback: match detector pixel size
        end
        if isfield(S.extra.meta,'attr_slice_thickness_mm')
            dz = double(S.extra.meta.attr_slice_thickness_mm);
        else
            dz = dx; % fallback
        end
    else
        % legacy fallback
        fov_xy_mm = 300;
        dx = fov_xy_mm / nx; dy = fov_xy_mm / ny;
        dz = dx;
    end
    voxel_size = [dx dy dz];

    if verbose
        fprintf('[IR] Start | UseGPU=%s | nx=%d ny=%d nz=%d | nu=%d nv=%d | du=%.3f dv=%.3f | dx=%.3f dy=%.3f dz=%.3f\n', ...
            ternary(decideGPU(UseGPU),'true','false'), nx, ny, nz, nu, nv, du, dv, dx, dy, dz);
    end

    % -------- Iterative refinement --------
    for k = 1:numIter
        % Forward projection A*x
        Ax = forwardProject_CBCT(x, angles_rad, DSD, DSO, [du dv], voxel_size, nu, nv, u0, v0, ...
                                 'UseGPU', UseGPU, 'NumSamples', numSamples, 'BatchPixels', batchPixels);

        % Residual
        res = single(P) - single(Ax);

        % Backproject gradient ~ A' * res  (uses unified FDK backprojector math)
        grad = simpleFDK_CBCT_vox(res, angles_rad, DSD, DSO, [du dv], voxel_size, ...
                                  nx, ny, nz, 'ramp', 'u0', u0, 'v0', v0, ...
                                  'UseGPU', UseGPU, 'PadFactor', padFactor, 'BatchZ', batchZ);

        % Simple Gaussian-denoised prior (Tikhonov-like)
        reg  = lambda * (x - imgaussfilt3(x, sigG, 'FilterSize', 3*ceil(2*sigG)+1));

        % Update
        x = x + alpha * (grad - reg);

        if verbose
            fprintf('Iter %d/%d | ||res||_2 = %.3e\n', k, numIter, norm(res(:)));
        end
    end

    hybridIR_vol = x;

    % Optional save (commented — up to you)
    % save('Hybrid_IR_high.mat','hybridIR_vol');
end

% --- shared helpers (same as in other file; kept local for portability) ---
function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
function tf = decideGPU(useGPUOpt)
    if ischar(useGPUOpt) || isstring(useGPUOpt)
        tf = gpuAvailable();
    else
        tf = logical(useGPUOpt) && gpuAvailable();
    end
end
function tf = gpuAvailable()
    tf = false;
    try
        if exist('gpuDeviceCount','file')
            tf = gpuDeviceCount > 0;
            if tf, gpuDevice; end
        end
    catch
        tf = false;
    end
end