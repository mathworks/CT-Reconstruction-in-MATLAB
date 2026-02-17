function Ax = forwardProject_CBCT(volume, angles_rad, DSD, DSO, det_pixel_size, voxel_size, nu, nv, u0, v0, varargin)
% forwardProject_CBCT: Cone-beam forward projection (GPU if available, else CPU)
%
% Inputs:
%   volume         - 3D array [nx x ny x nz] (iso-centered, spacing voxel_size)
%   angles_rad     - [1 x nViews] (radians)
%   DSD, DSO       - source-to-detector, source-to-isocenter (mm)
%   det_pixel_size - scalar or [du dv] (mm)
%   voxel_size     - scalar or [dx dy dz] (mm)  (used for volume grid spacing)
%   nu, nv         - detector size (pixels)
%   u0, v0         - detector principal point (pixels)
%
% Name-Value:
%   'UseGPU'       - true | false | 'auto' (default: 'auto')
%   'NumSamples'   - # of samples along ray (default: 64)
%   'BatchPixels'  - per-chunk ray count (default: 4096)
%   'Verbose'      - print info (default: false)
%
% Output:
%   Ax             - [nu x nv x nViews] (single)

    % ---------- Parse options ----------
    p = inputParser;
    addParameter(p, 'UseGPU', 'auto', @(x)islogical(x) || (ischar(x)&&strcmpi(x,'auto')));
    addParameter(p, 'NumSamples', 64, @(x)isscalar(x) && x>=2);
    addParameter(p, 'BatchPixels', 4096, @(x)isscalar(x) && x>=128);
    addParameter(p, 'Verbose', false, @(x)islogical(x) || isnumeric(x));
    parse(p, varargin{:});

    useGPU      = decideGPU(p.Results.UseGPU);
    numSamples  = p.Results.NumSamples;
    batchSize   = p.Results.BatchPixels;
    verbose     = logical(p.Results.Verbose);

    % ---------- Geometry / sizes ----------
    [nx, ny, nz] = size(volume);
    if isscalar(det_pixel_size), du = det_pixel_size; dv = det_pixel_size;
    else, du = det_pixel_size(1); dv = det_pixel_size(2); end

    if isscalar(voxel_size), dx = voxel_size; dy = voxel_size; dz = voxel_size;
    else, dx = voxel_size(1); dy = voxel_size(2); dz = voxel_size(3); end

    angles_rad = angles_rad(:)';
    nViews     = numel(angles_rad);

    % ---------- Volume grid in mm (iso-centered) ----------
    xGrid = single(((0:nx-1) - (nx-1)/2) * dx);
    yGrid = single(((0:ny-1) - (ny-1)/2) * dy);
    zGrid = single(((0:nz-1) - (nz-1)/2) * dz);

    % Detector pixel coords (mm) centered at u0,v0
    u_vec = single(((1:nu) - u0) * du);       % [1 x nu] in mm
    v_vec = single(((1:nv) - v0) * dv);       % [1 x nv] in mm
    [U2, V2] = ndgrid(u_vec, v_vec);          % [nu x nv]

    % Flatten rays into 1D list
    U_list = reshape(U2, [], 1);  % [numPixels x 1]
    V_list = reshape(V2, [], 1);
    numPixels = numel(U_list);

    % Pre-allocate output
    if useGPU
        AxGPU = gpuArray.zeros(nu, nv, nViews, 'single');
        Vvol  = gpuArray(single(volume));
    else
        Ax    = zeros(nu, nv, nViews, 'single');
        Vvol  = single(volume);
    end

    if verbose
        fprintf('[FWD] %s | nx=%d ny=%d nz=%d | nu=%d nv=%d | du=%.3f dv=%.3f | dx=%.3f dy=%.3f dz=%.3f | samples=%d\n', ...
            ternary(useGPU,'GPU','CPU'), nx, ny, nz, nu, nv, du, dv, dx, dy, dz, numSamples);
    end

    % ---------- For each view ----------
    % Note: keep CPU work on CPU, GPU on GPU; we only move per-view result if needed.
    f = single(linspace(0, 1, numSamples));  % param [0..1] along ray
    for ia = 1:nViews
        theta = angles_rad(ia);

        % Basis vectors / positions (in mm)
        rhat = [cos(theta),  sin(theta), 0];
        uhat = [-sin(theta), cos(theta), 0];
        vhat = [0, 0, 1];

        src  = [-DSO * cos(theta), -DSO * sin(theta), 0];   % source @ distance DSO from iso
        D0   = src + rhat * DSD;                            % detector center

        % Expand per-batch to control memory
        if useGPU
            projAll = gpuArray.zeros(numPixels, 1, 'single');
        else
            projAll = zeros(numPixels, 1, 'single');
        end

        % Process rays in chunks
        for s = 1:batchSize:numPixels
            e  = min(s + batchSize - 1, numPixels);
            ub = U_list(s:e);
            vb = V_list(s:e);

            % Detector points in 3D
            detX = D0(1) + ub * uhat(1);
            detY = D0(2) + ub * uhat(2);
            detZ = D0(3) + vb * vhat(3);

            % Ray directions (normalize)
            dirX = detX - src(1);
            dirY = detY - src(2);
            dirZ = detZ - src(3);
            normDir = sqrt(dirX.^2 + dirY.^2 + dirZ.^2) + eps('single');
            dirX = dirX ./ normDir;
            dirY = dirY ./ normDir;
            dirZ = dirZ ./ normDir;

            % Sample points along ray: t in [0 .. DSD] mm (approx path length)
            % Xq: [nrays x nsamples]
            if useGPU
                % Ensure arrays are on GPU
                srcx = gpuArray(single(src(1))); srcy = gpuArray(single(src(2))); srcz = gpuArray(single(src(3)));
                dirX = gpuArray(single(dirX));   dirY = gpuArray(single(dirY));   dirZ = gpuArray(single(dirZ));
                T    = gpuArray(single(f) * DSD);
                Xq   = srcx + dirX .* T;  % implicit expansion
                Yq   = srcy + dirY .* T;
                Zq   = srcz + dirZ .* T;

                % Interpolate on the physical grid
                vals = interp3(gpuArray(xGrid), gpuArray(yGrid), gpuArray(zGrid), Vvol, Xq, Yq, Zq, 'linear', 0);

                % Approximate line integral
                projAll(s:e) = sum(vals, 2) .* (DSD / numSamples);
            else
                srcx = single(src(1)); srcy = single(src(2)); srcz = single(src(3));
                dirX = single(dirX);   dirY = single(dirY);   dirZ = single(dirZ);
                T    = single(f) * DSD;
                Xq   = srcx + dirX .* T;
                Yq   = srcy + dirY .* T;
                Zq   = srcz + dirZ .* T;

                vals = interp3(xGrid, yGrid, zGrid, Vvol, Xq, Yq, Zq, 'linear', 0);
                projAll(s:e) = sum(vals, 2) .* (DSD / numSamples);
            end
        end

        % Reshape back to [nu x nv]
        if useGPU
            AxGPU(:,:,ia) = reshape(projAll, nu, nv);
        else
            Ax(:,:,ia) = reshape(projAll, nu, nv);
        end
    end

    if useGPU
        Ax = gather(AxGPU);
    end
end

% -------- helpers --------
function tf = decideGPU(useGPUOpt)
    if ischar(useGPUOpt) || isstring(useGPUOpt)   % 'auto'
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

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end