function recon_volume = simpleFDK_CBCT_vox(projections, angles, dsd, dso, det_pixel_size, voxel_size, nx, ny, nz, filter_type, varargin)
% SIMPLEFDK_CBCT_VOX  3D FDK reconstruction with explicit voxel spacing [dx dy dz]
% Auto-selects GPU (for FFT filtering) if available; otherwise uses CPU.
%
% Inputs:
%   projections     : [nu, nv, nViews]   (line integrals)
%   angles          : [1 x nViews] (radians)
%   dsd, dso        : source-to-detector, source-to-isocenter (mm)
%   det_pixel_size  : scalar (du=dv) or [du dv] (mm)
%   voxel_size      : scalar (dx=dy=dz) or [dx dy dz] (mm)
%   nx, ny, nz      : reconstruction grid size
%   filter_type     : 'ramp' | 'shepp-logan' | 'hamming'
%
% Name-Value optional parameters:
%   'u0'            : detector principal point in u (pixels). Default: (nu+1)/2
%   'v0'            : detector principal point in v (pixels). Default: (nv+1)/2
%   'Verbose'       : logical, print diagnostics (default: false)
%   'PadFactor'     : zero-padding factor along u for filtering (default: 2)
%   'BatchZ'        : z-slices per batch (default: 16)
%   'UseGPU'        : true | false | 'auto' (default: 'auto')  % GPU for FFT stage
%
% Notes:
% - 2D cosine weighting (u & v) mitigates cone-angle shading.
% - 1D filtering along u with zero-padding (GPU-accelerated if available).
% - Voxel-driven backprojection with z-batching (CPU; identical math).
% - (u0,v0) are in *pixels*; du,dv are in mm; coordinates combine both.
%
% Output:
%   recon_volume    : [nx, ny, nz] (single)

    % ------------ Defaults & NV parsing ------------
    p = inputParser;
    p.FunctionName = 'simpleFDK_CBCT_vox';
    addParameter(p, 'u0', [], @(x) isempty(x) || (isscalar(x) && isnumeric(x)));
    addParameter(p, 'v0', [], @(x) isempty(x) || (isscalar(x) && isnumeric(x)));
    addParameter(p, 'Verbose', false, @(x)islogical(x) || isnumeric(x));
    addParameter(p, 'PadFactor', 2, @(x)isscalar(x) && x>=1);
    addParameter(p, 'BatchZ', 16, @(x)isscalar(x) && x>=1);
    addParameter(p, 'UseGPU', 'auto', @(x) (ischar(x) && any(strcmpi(x,{'auto'}))) || islogical(x));
    parse(p, varargin{:});

    verbose   = logical(p.Results.Verbose);
    padFactor = p.Results.PadFactor;
    batchZ    = p.Results.BatchZ;
    useGPU    = decideGPU(p.Results.UseGPU);

    % ---- Sizes and geometry ----
    [nu, nv, nViews] = size(projections);
    angles = angles(:)';                           % row vector

    if isscalar(det_pixel_size)
        du = det_pixel_size; dv = det_pixel_size;
    else
        du = det_pixel_size(1);
        dv = det_pixel_size(2);
    end

    if isscalar(voxel_size)
        dx = voxel_size; dy = voxel_size; dz = voxel_size;
    else
        dx = voxel_size(1); dy = voxel_size(2); dz = voxel_size(3);
    end

    % ---- Principal point (pixels) ----
    if isempty(p.Results.u0), u0 = (nu + 1)/2; else, u0 = p.Results.u0; end
    if isempty(p.Results.v0), v0 = (nv + 1)/2; else, v0 = p.Results.v0; end

    % ---- Image grids (iso-centered) ----
    x = ((0:nx-1) - (nx-1)/2) * dx;   % mm
    y = ((0:ny-1) - (ny-1)/2) * dy;   % mm
    z = ((0:nz-1) - (nz-1)/2) * dz;   % mm
    [X2D, Y2D] = meshgrid(x, y);      % [ny, nx]

    % ============================================================
    % 2D Cosine weighting (fan + cone) using (u0,v0)
    % ============================================================
    u = ((1:nu) - u0) * du;           % mm
    v = ((1:nv) - v0) * dv;           % mm
    [UU, VV] = ndgrid(u, v);          % [nu, nv]
    cosw_2d = dsd ./ sqrt(dsd^2 + UU.^2 + VV.^2);   % [nu, nv]
    projections = projections .* reshape(cosw_2d, [nu nv 1]);

    % ---- 1D ramp-domain filtering along u (zero-padded) ----
    Nf = 2^nextpow2(padFactor * nu);

    % Frequency axis in cycles/mm (centered)
    k  = (0:Nf-1).';
    df = 1 / (Nf * du);
    f  = zeros(Nf,1);
    half = floor(Nf/2);
    f(1:half+1)      = (0:half).' * df;           % [0 .. +Nyquist]
    f(half+2:end)    = (-(half-1):-1).' * df;     % negative freqs
    ramp = abs(f);                                 % |f|

    switch lower(filter_type)
        case 'ramp'
            H = ramp;
        case 'shepp-logan'
            H = ramp .* sinc(f * du);             % MATLAB's sinc uses pi
        case 'hamming'
            w = 0.54 + 0.46 * cos(2*pi*((k/(Nf-1)) - 0.5));
            H = ramp .* w;
        otherwise
            error('Unknown filter type: %s', filter_type);
    end
    H = reshape(H, [Nf 1]);    % broadcast across u

    % Zero-padding indices
    pre  = floor((Nf - nu)/2);
    post = ceil((Nf - nu)/2);

    % ---- Per-view filtering (GPU if available) ----
    projections_filt = zeros(nu, nv, nViews, 'single');
    if verbose
        fprintf('[FDK] Filtering %d views with %s FFT ...\n', nViews, ternary(useGPU,'GPU','CPU'));
    end
    for ia = 1:nViews
        % Pad single view
        Pp = padarray(projections(:,:,ia), [pre 0], 0, 'pre');
        Pp = padarray(Pp,                   [post 0], 0, 'post');

        if useGPU
            Pp_gpu = gpuArray(single(Pp));
            Pf_gpu = fft(Pp_gpu, [], 1);
            Pf_gpu = Pf_gpu .* gpuArray(H);
            Pp_gpu = real(ifft(Pf_gpu, [], 1));
            Pp     = gather(Pp_gpu);
        else
            Pf = fft(Pp, [], 1);
            Pf = Pf .* H;
            Pp = real(ifft(Pf, [], 1));
        end

        % Crop back
        projections_filt(:,:,ia) = single(Pp(pre+1 : pre+nu, :));

    end
    sliceViewer(projections_filt);

    % ---- Angle weights for integration ----
    if nViews > 1
        d = diff(angles);
        if max(abs(d - mean(d))) < 1e-6
            dth = repmat(mean(d), 1, nViews);
        else
            dth = zeros(1, nViews);
            dth(1) = d(1);
            dth(2:end-1) = (d(1:end-1) + d(2:end))/2;
            dth(end) = d(end);
        end
    else
        dth = 2*pi;
    end

    denom_eps = max(1e-6, 1e-6 * dso);
    uGrid = 1:nu; vGrid = 1:nv;

    if verbose
        fprintf('[FDK] nu=%d nv=%d V=%d | nx=%d ny=%d nz=%d | du=%.3f dv=%.3f | dx=%.3f dy=%.3f dz=%.3f | u0=%.3f v0=%.3f | GPU=%d\n', ...
            nu, nv, nViews, nx, ny, nz, du, dv, dx, dy, dz, u0, v0, useGPU);
    end

    % ---- Backprojection with z-batching (CPU) ----
    recon_yx_z = zeros(ny, nx, nz, 'single');
    kStarts = 1:batchZ:nz;

    for ia = 1:nViews
        ca = cos(angles(ia)); sa = sin(angles(ia));
        % rotate (x,y) into view coordinates
        Xr =  ca * X2D - sa * Y2D;             % [ny, nx]
        Yr =  sa * X2D + ca * Y2D;             % [ny, nx]

        % Interpolant for current view (u along rows, v along cols)
        F = griddedInterpolant({uGrid, vGrid}, projections_filt(:,:,ia), 'linear', 'none');

        % geometry terms reused within z-batches
        denom = dso - Xr;                       % [ny, nx]
        valid = denom > denom_eps;
        denom = max(denom, denom_eps);
        t = dsd ./ denom;                       % magnification [ny, nx]

        % Precompute u_idx base (independent of z)
        u_idx_base = u0 + (t .* (Yr / du));     % [ny, nx]

        for k0 = kStarts
            k1 = min(nz, k0 + batchZ - 1);
            nb = k1 - k0 + 1;

            Zk = reshape(z(k0:k1), 1, 1, nb);   % [1,1,nb]

            % Project voxel centers -> detector indices
            u_idx = repmat(u_idx_base, 1, 1, nb);               % [ny, nx, nb]
            v_idx = v0 + (t .* (Zk / dv));                      % [ny, nx, nb]

            % Sample filtered projections
            samp = F(u_idx, v_idx);                             % [ny, nx, nb]
            samp(~isfinite(samp)) = 0;

            % FDK backprojection weighting (circular trajectory)
            wbp = (dso ./ denom).^2;                            % [ny, nx]
            contrib = single(samp .* repmat(wbp, 1, 1, nb) * dth(ia));

            if any(~valid(:))
                mask3 = repmat(~valid, 1, 1, nb);
                contrib(mask3) = 0;
            end

            recon_yx_z(:,:,k0:k1) = recon_yx_z(:,:,k0:k1) + contrib;
        end
    end

    % normalize by 2*pi (continuous-angle equivalent)
    recon_yx_z = recon_yx_z / (2*pi);
    recon_yx_z(~isfinite(recon_yx_z)) = 0;

    % return as [nx ny nz]
    recon_volume = permute(recon_yx_z, [2 1 3]);
end

% ---------------------- helpers ----------------------
function tf = decideGPU(useGPUOpt)
    if ischar(useGPUOpt) || isstring(useGPUOpt)
        % 'auto'
        tf = gpuAvailable();
    else
        % true/false, but ensure device exists if true
        tf = logical(useGPUOpt) && gpuAvailable();
    end
end

function tf = gpuAvailable()
    tf = false;
    try
        % Fast check; available in recent MATLAB
        if exist('gpuDeviceCount','file')
            tf = gpuDeviceCount > 0;
            if tf
                % Ping device to ensure it initializes
                gpuDevice;
            end
        end
    catch
        tf = false;
    end
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end