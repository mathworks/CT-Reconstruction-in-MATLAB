function hybridIR_vol = reconIR_codegen( ...
    P, angles_rad, DSD, DSO, du, dv, u0_pixels, v0_pixels, ...
    dx, dy, dz, nx, ny, nz, x0, ...
    numIter, alpha, lambda, gaussSigma, ...
    padFactor, batchZ, numSamples, batchPixels, useGPU, verbose)
%#codegen
% reconIR_codegen
% Hybrid iterative refinement starting from an initial FDK volume (x0).
% CUDA-MEX entrypoint (no file I/O). FFTs in backprojector map to cuFFT.
%
% Inputs:
%   P           : [nu,nv,nViews] single
%   angles_rad  : [1,nViews] single (radians)
%   DSD, DSO    : single
%   du, dv      : detector pixel size in mm (single)
%   u0_pixels, v0_pixels : detector principal point (pixels, single)
%   dx,dy,dz    : voxel size in mm (single)
%   nx,ny,nz    : int32
%   x0          : initial volume [nx,ny,nz] single (e.g., FDK)
%   numIter     : int32
%   alpha       : single (step size)
%   lambda      : single (reg weight)
%   gaussSigma  : single (sigma for Gaussian prior)
%   padFactor   : single (>=1) for backproject filt padding
%   batchZ      : int32 (z batch for backproj)
%   numSamples  : int32 (kept for signature; forward projector ignores it)
%   batchPixels : int32 (rays per batch for forward projection)
%   useGPU      : logical (enables GPU for forward+backprojection)
%   verbose     : logical
%
% Output:
%   hybridIR_vol: [nx,ny,nz] single
%
% Notes:
%   - Forward projector uses ray-box intersection + voxel step (fast).
%   - Backprojector uses cuFFT when useGPU=true and running in MATLAB.
%   - Progress bars only print in MATLAB (not in generated CUDA).

    % -------- Types & shapes --------
    P           = single(P);
    angles_rad  = single(angles_rad(:).');     % [1,nViews]
    DSD         = single(DSD);
    DSO         = single(DSO);
    du          = single(du);    dv = single(dv);
    u0_pixels   = single(u0_pixels);
    v0_pixels   = single(v0_pixels);
    dx          = single(dx);    dy = single(dy);    dz = single(dz);
    nx          = int32(nx);     ny = int32(ny);     nz = int32(nz);
    x           = single(x0);                      % current estimate
    numIter     = int32(numIter);
    alpha       = single(alpha);
    lambda      = single(lambda);
    gaussSigma  = single(gaussSigma);
    padFactor   = single(max(1, padFactor));
    batchZ      = int32(max(1, batchZ));
    numSamples  = int32(max(0, numSamples));       % kept for signature
    batchPixels = int32(max(128, batchPixels));
    useGPU      = logical(useGPU);
    verbose     = logical(verbose);

    [nu, nv, nViews] = size(P); %#ok<ASGLU>

    if verbose && coder.target('MATLAB')
        fprintf('[IR] start | iter=%d | alpha=%.3g | lambda=%.3g | sigma=%.3g | useGPU=%d\n', ...
            double(numIter), double(alpha), double(lambda), double(gaussSigma), useGPU);
    end

    % Geometry bundles
    det_pix   = single([du dv]);
    vox_size  = single([dx dy dz]);

    % -------- Iterative refinement with progress --------
    if verbose && (coder.target('MATLAB'))
        t_iter = progress_tic();
        progress_update('[IR]', 0, numIter, t_iter);
    end

    for it = int32(1) : numIter
        % Forward projection A*x  (GPU/CPU depending on useGPU)
        % NOTE: numSamples is ignored inside forward projector (ray-box + voxel step).
        Ax = forwardProject_CBCT_cg( ...
            x, angles_rad, DSD, DSO, det_pix, vox_size, ...
            int32(nu), int32(nv), u0_pixels, v0_pixels, ...
            numSamples, batchPixels, useGPU, verbose && it==1);

        % Residual
        res = P - Ax;   % single

        % Backproject gradient ~ A' * res  (cuFFT for filtering if useGPU=true)
        grad = simpleFDK_CBCT_vox_cg( ...
            res, angles_rad, DSD, DSO, det_pix, vox_size, ...
            nx, ny, nz, coder.const('ramp'), int32(1024), ...
            u0_pixels, v0_pixels, verbose && (it==1), padFactor, batchZ, useGPU);

        % Gaussian prior (separable, codegen-safe)
        if lambda > 0 && gaussSigma > 0
            smooth = gaussian_denoise3d_cg(x, gaussSigma);
            reg    = lambda * (x - smooth);
        else
            reg = zeros(size(x), 'like', x);
        end

        % Update
        x = x + alpha * (grad - reg);

        if verbose && coder.target('MATLAB')
            % Lightweight metric
            r2 = sqrt(mean(res(:).^2));
            fprintf('  iter %d/%d | RMS(res)=%.3e\n', double(it), double(numIter), double(r2));
            progress_update('[IR]', it, numIter, t_iter);
        end
    end

    hybridIR_vol = x;
end

% ======================================================================
% Cone-beam forward projector (GPU/CPU; trilinear; RAY-BOX + VOXEL STEP)
% ======================================================================
function Ax = forwardProject_CBCT_cg(volume, angles_rad, DSD, DSO, det_pixel_size, voxel_size, ...
                                     nu, nv, u0, v0, numSamples, batchPixels, useGPU, verbose)
%#codegen
% Output Ax: [nu,nv,nViews] single
%
% Speedups:
%   - Ray/box intersection (slab) => sample only inside volume.
%   - Step size ~ min(dx,dy,dz) (voxel-scale) => fewer samples per ray.
%   - MATLAB GPU path keeps volume on device; CUDA MEX uses device kernel.
%   - CPU fallback also uses ray-box + voxel step.
%
% NOTE: 'numSamples' is kept for signature compatibility but NOT USED here.

    volume = single(volume);
    [nx, ny, nz] = size(volume);

    % Detector sizes
    if isscalar(det_pixel_size)
        du = det_pixel_size; dv = det_pixel_size;
    else
        du = det_pixel_size(1); dv = det_pixel_size(2);
    end
    % Voxel sizes
    if isscalar(voxel_size)
        dx = voxel_size; dy = voxel_size; dz = voxel_size;
    else
        dx = voxel_size(1); dy = voxel_size(2); dz = voxel_size(3);
    end

    % Index-space centers for (1-based) mapping:
    % idx = x/dx + (nx-1)/2 + 1, etc.
    cx = (double(nx)-1)/2 + 1;
    cy = (double(ny)-1)/2 + 1;
    cz = (double(nz)-1)/2 + 1;

    % Volume AABB in physical coords (mm) around isocenter
    boxMin = [ (1 - cx)*double(dx), (1 - cy)*double(dy), (1 - cz)*double(dz) ];
    boxMax = [ (double(nx)-cx)*double(dx), (double(ny)-cy)*double(dy), (double(nz)-cz)*double(dz) ];

    % Detector pixel coordinates (mm) centered at u0,v0
    u_vec = single(((1:double(nu)) - u0) * du);
    v_vec = single(((1:double(nv)) - v0) * dv);
    [U2, V2] = ndgrid(u_vec, v_vec);     % [nu x nv]

    % Flatten rays
    U_list = reshape(U2, [], 1);         % [numPixels x 1]
    V_list = reshape(V2, [], 1);
    numPixelsTot = int32(numel(U_list));

    nViews = int32(numel(angles_rad));
    Ax = zeros(nu, nv, nViews, 'single');

    % Step size ~ voxel size in mm
    step = single(max( min([dx,dy,dz]), eps('single')));

    % MATLAB runtime GPU: keep volume on device persistently
    persistent volG gx gy gz nxG nyG nzG
    if coder.target('MATLAB') && useGPU
        nxG = size(volume,1); nyG = size(volume,2); nzG = size(volume,3); %#ok<NASGU>
        volG = gpuArray(volume);
        gx = gpuArray.colon(1, nxG);
        gy = gpuArray.colon(1, nyG);
        gz = gpuArray.colon(1, nzG);
    end

    if verbose && coder.target('MATLAB')
        fprintf('[FWD*] nx=%d ny=%d nz=%d | nu=%d nv=%d | step=%.3g mm | batch=%d | GPU=%d (ray-box)\n', ...
            nx, ny, nz, nu, nv, double(step), batchPixels, useGPU);
        t_view = progress_tic();
        progress_update('[FWD* views]', 0, nViews, t_view);
    end

    for ia = int32(1) : nViews
        theta = angles_rad(double(ia));

        if useGPU
            if coder.target('MATLAB')
                % MATLAB runtime GPU path
                viewProj = fwdproj_view_GPU_runtime_raybox( ...
                    volG, gx, gy, gz, U_list, V_list, theta, ...
                    DSD, DSO, du, dv, dx, dy, dz, ...
                    cx, cy, cz, boxMin, boxMax, step, numPixelsTot);
            else
                % CUDA MEX (GPU Coder) path
                viewProj = fwdproj_view_GPU_codegen_raybox( ...
                    volume, U_list, V_list, theta, ...
                    DSD, DSO, du, dv, dx, dy, dz, ...
                    cx, cy, cz, boxMin, boxMax, step, numPixelsTot, batchPixels);
            end
        else
            % CPU fallback with ray-box + voxel step
            viewProj = fwdproj_view_CPU_raybox( ...
                volume, U_list, V_list, theta, ...
                DSD, DSO, du, dv, dx, dy, dz, ...
                cx, cy, cz, boxMin, boxMax, step, numPixelsTot, batchPixels);
        end

        Ax(:,:,double(ia)) = reshape(viewProj, double(nu), double(nv));

        if verbose && coder.target('MATLAB')
            progress_update('[FWD* views]', ia, nViews, t_view);
        end
    end
end

% ---------- Ray-box intersection (slab method), returns t0<=t1 ----------
function [t0, t1, hit] = rayBoxIntersect(src, dir, boxMin, boxMax)
%#codegen
% src, dir: 1x3 doubles (world coords); boxMin/Max: 1x3 doubles
    t0 = -realmax('double');
    t1 =  realmax('double');
    for a = 1:3
        if abs(dir(a)) < 1e-20
            if src(a) < boxMin(a) || src(a) > boxMax(a)
                hit = false; t0 = 0; t1 = -1; return;
            end
        else
            invD = 1.0 / dir(a);
            tNear = (boxMin(a) - src(a)) * invD;
            tFar  = (boxMax(a) - src(a)) * invD;
            if tNear > tFar
                tmp = tNear; tNear = tFar; tFar = tmp;
            end
            if tNear > t0, t0 = tNear; end
            if tFar  < t1, t1 = tFar;  end
            if t0 > t1
                hit = false; t0 = 0; t1 = -1; return;
            end
        end
    end
    hit = (t1 > 0);
    if t0 < 0, t0 = 0; end
end

% ---------------- CPU path (ray-box + voxel step) ----------------
function viewProj = fwdproj_view_CPU_raybox( ...
    volume, U_list, V_list, theta, ...
    DSD, DSO, du, dv, dx, dy, dz, ...
    cx, cy, cz, boxMin, boxMax, step, numPixelsTot, batchPixels)
%#codegen
    viewProj = zeros(double(numPixelsTot), 1, 'single');

    ca = cos(theta); sa = sin(theta);
    rhat = [ca,  sa, 0];
    uhat = [-sa, ca, 0];
    vhat = [0,   0,  1];

    src  = [-DSO * ca, -DSO * sa, 0];
    D0   = [src(1) + rhat(1)*DSD, src(2) + rhat(2)*DSD, src(3) + rhat(3)*DSD];

    stepBatch = max(batchPixels,int32(256));
    for s = int32(1) : stepBatch : numPixelsTot
        e  = min(numPixelsTot, s + stepBatch - int32(1));
        ub = U_list(double(s):double(e));
        vb = V_list(double(s):double(e));
        nb = int32(numel(ub));

        detX = D0(1) + ub * uhat(1);
        detY = D0(2) + ub * uhat(2);
        detZ = D0(3) + vb * vhat(3);

        dirX = detX - src(1); dirY = detY - src(2); dirZ = detZ - src(3);
        normDir = sqrt(dirX.^2 + dirY.^2 + dirZ.^2) + eps('single');
        dirX = dirX ./ normDir; dirY = dirY ./ normDir; dirZ = dirZ ./ normDir;

        for i = 1:double(nb)
            dir = [double(dirX(i)) double(dirY(i)) double(dirZ(i))];
            [t0, t1, hit] = rayBoxIntersect(double(src), dir, boxMin, boxMax);
            acc = single(0);
            if hit && (t1 > t0)
                ns = max(1, floor( (t1 - t0) / double(step) ));
                dt = single((t1 - t0) / double(ns));
                % march along [t0,t1]
                t = single(t0) + dt * single(0:ns-1);
                X = single(src(1)) + single(dir(1)) * t;
                Y = single(src(2)) + single(dir(2)) * t;
                Z = single(src(3)) + single(dir(3)) * t;
                IX = X ./ dx + cx;  IY = Y ./ dy + cy;  IZ = Z ./ dz + cz;
                vals = trilinearSample3D(volume, IX, IY, IZ);
                acc  = sum(vals,'native') * dt;
            end
            viewProj(double(s)+i-1) = acc;
        end
    end
end

% -------------- MATLAB runtime GPU path (gpuArray + interpn) --------------
function viewProj = fwdproj_view_GPU_runtime_raybox( ...
    volG, gx, gy, gz, U_list, V_list, theta, ...
    DSD, DSO, du, dv, dx, dy, dz, ...
    cx, cy, cz, boxMin, boxMax, step, numPixelsTot)
%#codegen
    % Larger batches reduce overhead
    gpuBatch = int32(256*1024);
    viewProj = zeros(double(numPixelsTot), 1, 'single');

    ca = cos(theta); sa = sin(theta);
    rhat = [ca,  sa, 0];
    uhat = [-sa, ca, 0];
    vhat = [0,   0,  1];

    src  = [-DSO * ca, -DSO * sa, 0];
    D0   = [src(1) + rhat(1)*DSD, src(2) + rhat(2)*DSD, src(3) + rhat(3)*DSD];

    nx = size(volG,1); ny = size(volG,2); nz = size(volG,3);

    for s = int32(1) : gpuBatch : numPixelsTot
        e  = min(numPixelsTot, s + gpuBatch - int32(1));
        ub = U_list(double(s):double(e));      % CPU
        vb = V_list(double(s):double(e));      % CPU
        nb = int32(numel(ub));

        % Detector points (CPU)
        detXc = D0(1) + ub * uhat(1);
        detYc = D0(2) + ub * uhat(2);
        detZc = D0(3) + vb * vhat(3);

        % Ray directions (CPU)
        dirXc = detXc - src(1); dirYc = detYc - src(2); dirZc = detZc - src(3);
        normDir = sqrt(dirXc.^2 + dirYc.^2 + dirZc.^2) + eps('single');
        dirXc = dirXc ./ normDir; dirYc = dirYc ./ normDir; dirZc = dirZc ./ normDir;

        accG = zeros(double(nb),1,'single','gpuArray');

        % For each ray: compute t0,t1 on CPU; integrate on GPU
        for i = 1:double(nb)
            dir = [double(dirXc(i)) double(dirYc(i)) double(dirZc(i))];
            [t0, t1, hit] = rayBoxIntersect(double(src), dir, boxMin, boxMax);
            if hit && (t1 > t0)
                ns = max(1, floor( (t1 - t0) / double(step) ));
                dt = single((t1 - t0) / double(ns));
                t  = gpuArray(single(t0) + dt * single(0:ns-1));
                X  = single(src(1)) + single(dir(1)) * t;
                Y  = single(src(2)) + single(dir(2)) * t;
                Z  = single(src(3)) + single(dir(3)) * t;

                IX = X ./ dx + cx; IY = Y ./ dy + cy; IZ = Z ./ dz + cz;
                IX = max(1, min(nx, IX)); IY = max(1, min(ny, IY)); IZ = max(1, min(nz, IZ));

                vals = interpn(gx, gy, gz, volG, IX, IY, IZ, 'linear', 0);
                accG(i) = sum(vals,'native') * dt;
            end
        end

        viewProj(double(s):double(e)) = gather(accG);
    end
end

% ---------------- CUDA MEX (GPU Coder) path with ray-box ----------------
function viewProj = fwdproj_view_GPU_codegen_raybox( ...
    volume, U_list, V_list, theta, ...
    DSD, DSO, du, dv, dx, dy, dz, ...
    cx, cy, cz, boxMin, boxMax, step, numPixelsTot, batchPixels)
%#codegen
    viewProj = zeros(double(numPixelsTot), 1, 'single');
    nx = int32(size(volume,1)); ny = int32(size(volume,2)); nz = int32(size(volume,3));
    if batchPixels <= 0, batchPixels = min(int32(256*1024), numPixelsTot); end

    ca = cos(theta); sa = sin(theta);
    rhat = [ca,  sa, 0];
    uhat = [-sa, ca, 0];
    vhat = [0,   0,  1];

    src  = [-DSO * ca, -DSO * sa, 0];
    D0x  = src(1) + rhat(1)*DSD;
    D0y  = src(2) + rhat(2)*DSD;
    D0z  = src(3) + rhat(3)*DSD;

    for s = int32(1) : batchPixels : numPixelsTot
        e  = min(numPixelsTot, s + batchPixels - int32(1));
        nb = int32(e - s + int32(1));

        coder.gpu.kernelfun();
        for p = int32(0) : nb - int32(1)
            idx = s + p;
            ub  = U_list(double(idx));
            vb  = V_list(double(idx));

            detX = D0x + ub * uhat(1);
            detY = D0y + ub * uhat(2);
            detZ = D0z + vb * vhat(3);

            dirX = detX - src(1); dirY = detY - src(2); dirZ = detZ - src(3);
            normDir = sqrt(dirX*dirX + dirY*dirY + dirZ*dirZ) + eps('single');
            dirX = dirX / normDir; dirY = dirY / normDir; dirZ = dirZ / normDir;

            % ray-box intersection (double math for robustness)
            srcd = [double(src(1)) double(src(2)) double(src(3))];
            dird = [double(dirX)   double(dirY)   double(dirZ)];
            [t0, t1, hit] = rayBoxIntersect(srcd, dird, boxMin, boxMax);

            acc = single(0);
            if hit && (t1 > t0)
                ns = max(int32(1), int32( floor( (t1 - t0) / double(step) ) ));
                dt = single( (t1 - t0) / double(ns) );

                for q = int32(0):ns-1
                    tq = single(t0) + dt * single(q);
                    X = single(src(1)) + single(dirX) * tq;
                    Y = single(src(2)) + single(dirY) * tq;
                    Z = single(src(3)) + single(dirZ) * tq;

                    IX = X / dx + cx;  IY = Y / dy + cy;  IZ = Z / dz + cz;
                    val = triSample3D_linear(volume, IX, IY, IZ, nx, ny, nz);
                    acc = acc + val;
                end
                acc = acc * dt;
            end

            viewProj(double(idx)) = acc;
        end
    end
end

% ======================================================================
% Bilinear backprojector (filters via cuFFT when gpuConfig MEX or MATLAB GPU)
% ======================================================================
function recon_volume = simpleFDK_CBCT_vox_cg( ...
    projections, angles, dsd, dso, det_pixel_size, voxel_size, ...
    nx, ny, nz, filter_type, Nf_fft, ...
    u0, v0, verbose, padFactor, batchZ, useGPU)
%#codegen
    angles = angles(:)';                          % [1,nViews]
    [nu, nv, nViews] = size(projections);

    if isscalar(det_pixel_size), du = det_pixel_size; dv = det_pixel_size;
    else, du = det_pixel_size(1); dv = det_pixel_size(2); end
    if isscalar(voxel_size), dx = voxel_size; dy = voxel_size; dz = voxel_size;
    else, dx = voxel_size(1); dy = voxel_size(2); dz = voxel_size(3); end

    if isnan(u0), u0 = cast((double(nu)+1)/2, 'like', u0); end
    if isnan(v0), v0 = cast((double(nv)+1)/2, 'like', v0); end

    x = ((0:double(nx)-1) - (double(nx)-1)/2) * double(dx);
    y = ((0:double(ny)-1) - (double(ny)-1)/2) * double(dy);
    z = ((0:double(nz)-1) - (double(nz)-1)/2) * double(dz);
    x = cast(x,'like',dx); y = cast(y,'like',dy); z = cast(z,'like',dz);
    [X2D, Y2D] = meshgrid(x, y);

    % Cosine weighting
    u_mm = (single(1:double(nu)) - u0) * du;
    v_mm = (single(1:double(nv)) - v0) * dv;
    [UU, VV] = ndgrid(u_mm, v_mm);
    cosw_2d = dsd ./ sqrt(dsd^2 + UU.^2 + VV.^2);
    projections = single(projections) .* reshape(cosw_2d, [double(nu) double(nv) 1]);

    % Angle weights
    if nViews > 1
        d = diff(angles);
        if max(abs(d - mean(d))) < single(1e-6)
            dth = repmat(mean(d), 1, nViews);
        else
            dth = zeros(1, nViews, 'like', angles);
            dth(1) = d(1);
            if nViews > 2, dth(2:end-1) = (d(1:end-1)+d(2:end))/2; end
            dth(end) = d(end);
        end
    else
        dth = cast(2*pi,'like',angles);
    end

    denom_eps = max(single(1e-6), single(1e-6) * dso);
    if verbose && coder.target('MATLAB')
        fprintf('[FDK] nu=%d nv=%d V=%d | nx=%d ny=%d nz=%d | du=%.3f dv=%.3f | dx=%.3f dy=%.3f dz=%.3f | GPU=%d | pad=%.2f | Nf=%d\n', ...
            nu, nv, nViews, nx, ny, nz, du, dv, dx, dy, dz, useGPU, padFactor, Nf_fft);
        t_view = progress_tic();
        progress_update('[FDK views]', 0, int32(nViews), t_view);
    end

    % Effective FFT length
    coder.internal.errorIf(Nf_fft < int32(nu), 'Nf_fft must be >= nu');
    Nf_target = pow2ceil_from_scalar(double(padFactor) * double(nu));
    Nf_eff_d  = min(double(Nf_fft), max(double(nu), Nf_target));
    Nf_eff    = int32(Nf_eff_d);
    pre       = floor((double(Nf_eff) - double(nu)) / 2);

    % Build filter H
    k  = (0:double(Nf_eff)-1).';
    df = 1 / (double(Nf_eff) * double(du));
    f  = zeros(double(Nf_eff),1,'like',single(0));
    half = floor(double(Nf_eff)/2);
    f(1:half+1) = single((0:half).' * df);
    if Nf_eff > 2, f(half+2:end) = single((-(half-1):-1).' * df); end
    ramp = abs(f);
    switch lower(filter_type)
        case 'ramp',        H = ramp;
        case 'shepp-logan', H = ramp .* sinc(f * du);
        case 'hamming'
            phi = single((k/(double(Nf_eff)-1)) - 0.5);
            H   = ramp .* (single(0.54) + single(0.46) * cos(single(2*pi)*phi));
        otherwise, coder.internal.errorIf(true,'Unknown filter');
    end
    H = reshape(H, [double(Nf_eff) 1 1]);

    recon_yx_z = zeros(double(ny), double(nx), double(nz), 'single');
    maxChunkViews = int32(8);
    nViews_i32 = int32(nViews);

    for ia0 = int32(1) : maxChunkViews : nViews_i32
        ia1    = min(nViews_i32, ia0 + maxChunkViews - int32(1));
        nChunk = ia1 - ia0 + int32(1);

        Pp_chunk = zeros(Nf_fft, double(nv), double(nChunk), 'single');
        Pp_chunk(pre+1:pre+double(nu), :, :) = projections(:,:,double(ia0):double(ia1));

        if coder.target('MATLAB')
            if useGPU
                Pseg_gpu = gpuArray(Pp_chunk(1:double(Nf_eff), :, :));
                H_gpu    = gpuArray(H);
                Pf_gpu   = fft(Pseg_gpu, [], 1);
                Pf_gpu   = Pf_gpu .* H_gpu;
                Pseg_gpu = real(ifft(Pf_gpu, [], 1));
                Pp_chunk(1:double(Nf_eff), :, :) = gather(Pseg_gpu);
            else
                Pseg = Pp_chunk(1:double(Nf_eff), :, :);
                Pf   = fft(Pseg, [], 1);
                Pf   = Pf .* H;
                Pseg = real(ifft(Pf, [], 1));
                Pp_chunk(1:double(Nf_eff), :, :) = Pseg;
            end
        else
            Pseg = Pp_chunk(1:double(Nf_eff), :, :);
            Pf   = fft(Pseg, [], 1);
            Pf   = Pf .* H;
            Pseg = real(ifft(Pf, [], 1));
            Pp_chunk(1:double(Nf_eff), :, :) = Pseg;
        end

        Filt_chunk = Pp_chunk(pre+1:pre+double(nu), :, :);

        for j = int32(0) : nChunk - int32(1)
            ia = ia0 + j;
            ca = cos(angles(double(ia))); sa = sin(angles(double(ia)));
            Xr =  ca * X2D - sa * Y2D;
            Yr =  sa * X2D + ca * Y2D;

            denom = dso - Xr;
            valid = denom > denom_eps;
            denom = max(denom, denom_eps);
            t = dsd ./ denom;

            u_idx_base = u0 + (t .* (Yr / du));

            for k0 = int32(1) : batchZ : nz
                k1 = min(nz, k0 + batchZ - int32(1));
                nb = k1 - k0 + int32(1);

                Zk = reshape(z(double(k0):double(k1)), 1, 1, double(nb));

                u_idx = repmat(u_idx_base, 1, 1, double(nb));
                v_idx = v0 + (t .* (Zk / dv));

                curView = Filt_chunk(:,:,double(j)+1);
                samp = bilinearSample2D(curView, u_idx, v_idx);
                samp(~isfinite(samp)) = cast(0,'like',samp);

                wbp = (dso ./ denom).^2;
                contrib = samp .* repmat(wbp, 1, 1, double(nb)) * dth(double(ia));

                if any(~valid(:))
                    mask3 = repmat(~valid, 1, 1, double(nb));
                    contrib(mask3) = cast(0,'like',contrib);
                end

                recon_yx_z(:,:,double(k0):double(k1)) = recon_yx_z(:,:,double(k0):double(k1)) + contrib;
            end

            if verbose && coder.target('MATLAB')
                progress_update('[FDK views]', double(ia), nViews, t_view);
            end
        end
    end

    recon_yx_z = recon_yx_z / cast(2*pi, 'like', recon_yx_z);
    recon_yx_z(~isfinite(recon_yx_z)) = cast(0,'like',recon_yx_z);
    recon_volume = permute(recon_yx_z, [2 1 3]); % [nx,ny,nz]
end

% ======================================================================
% Separable 3D Gaussian (codegen-safe CPU path)
% ======================================================================
function Y = gaussian_denoise3d_cg(X, sigma)
%#codegen
    if sigma <= 0
        Y = X; return;
    end
    % Build 1D kernel (truncate at 3*sigma)
    r = int32(ceil(3 * double(sigma)));
    rad = double(r);
    t = (-rad:rad);
    g = exp(-0.5 * (t / double(sigma)).^2);
    g = single(g / sum(g));

    % Separable conv using convn
    kx = reshape(g, [], 1, 1);
    ky = reshape(g, 1, [], 1);
    kz = reshape(g, 1, 1, []);

    Y = convn(single(X), kx, 'same');
    Y = convn(Y, ky, 'same');
    Y = convn(Y, kz, 'same');
end

% ======================================================================
% Utilities
% ======================================================================
function p2 = pow2ceil_from_scalar(x)
%#codegen
    target = max(1.0, x);
    p = 1.0;
    while p < target
        p = p * 2.0;
    end
    p2 = p;
end

function s = linspace_single(a, b, n)
%#codegen
    if n <= 1
        s = single(a);
    else
        s = single(a) + single(0:single(n-1)) .* single((b - a) / single(n-1));
    end
end

function val = bilinearSample2D(img, u, v)
%#codegen
    nu = size(img,1); nv = size(img,2);
    u = max(single(1), min(single(nu - 1), u));
    v = max(single(1), min(single(nv - 1), v));
    u0 = floor(u); v0 = floor(v);
    du = u - u0;   dv = v - v0;
    u0i = int32(u0); v0i = int32(v0);
    u1i = u0i + 1;  v1i = v0i + 1;
    sz = [nu nv];
    f00 = img(sub2ind(sz, u0i, v0i));
    f10 = img(sub2ind(sz, u1i, v0i));
    f01 = img(sub2ind(sz, u0i, v1i));
    f11 = img(sub2ind(sz, u1i, v1i));
    val = (1-du).*(1-dv).*f00 + du.*(1-dv).*f10 + (1-du).*dv.*f01 + du.*dv.*f11;
end

function vals = trilinearSample3D(vol, ix, iy, iz)
%#codegen
% vol: [nx,ny,nz] single
% ix,iy,iz: index-space coords (1-based), same size (e.g., [1 x ns] or [nb x ns])
    nx = size(vol,1); ny = size(vol,2); nz = size(vol,3);

    % Clamp to [1..N-1] so +1 neighbor exists
    ix = max(single(1), min(single(nx - 1), ix));
    iy = max(single(1), min(single(ny - 1), iy));
    iz = max(single(1), min(single(nz - 1), iz));

    i0 = floor(ix); j0 = floor(iy); k0 = floor(iz);
    dx = ix - i0;   dy = iy - j0;   dz = iz - k0;

    i0i = int32(i0); j0i = int32(j0); k0i = int32(k0);
    i1i = i0i + 1;   j1i = j0i + 1;   k1i = k0i + 1;

    % Gather 8 neighbors using sub2ind
    sz = [nx ny nz];
    c000 = vol(sub2ind(sz, i0i, j0i, k0i));
    c100 = vol(sub2ind(sz, i1i, j0i, k0i));
    c010 = vol(sub2ind(sz, i0i, j1i, k0i));
    c110 = vol(sub2ind(sz, i1i, j1i, k0i));
    c001 = vol(sub2ind(sz, i0i, j0i, k1i));
    c101 = vol(sub2ind(sz, i1i, j0i, k1i));
    c011 = vol(sub2ind(sz, i0i, j1i, k1i));
    c111 = vol(sub2ind(sz, i1i, j1i, k1i));

    % Trilinear blend
    c00 = (1-dx).*c000 + dx.*c100;
    c01 = (1-dx).*c001 + dx.*c101;
    c10 = (1-dx).*c010 + dx.*c110;
    c11 = (1-dx).*c011 + dx.*c111;

    c0 = (1-dy).*c00 + dy.*c10;
    c1 = (1-dy).*c01 + dy.*c11;

    vals = (1-dz).*c0 + dz.*c1;
end

% Device-friendly scalar version used in CUDA MEX path
function val = triSample3D_linear(vol, ix, iy, iz, nx, ny, nz)
%#codegen
    % Clamp to [1..N-1] so +1 neighbor exists
    ix = max(single(1), min(single(nx - 1), ix));
    iy = max(single(1), min(single(ny - 1), iy));
    iz = max(single(1), min(single(nz - 1), iz));

    i0 = floor(ix); j0 = floor(iy); k0 = floor(iz);
    dx = ix - i0;   dy = iy - j0;   dz = iz - k0;

    i0i = int32(i0); j0i = int32(j0); k0i = int32(k0);
    i1i = i0i + 1;   j1i = j0i + 1;   k1i = k0i + 1;

    % linear index helper: idx = i + (j-1)*nx + (k-1)*nx*ny (1-based)
    nxny = nx * ny;
    idx000 = i0i + (j0i-1)*nx + (k0i-1)*nxny;
    idx100 = i1i + (j0i-1)*nx + (k0i-1)*nxny;
    idx010 = i0i + (j1i-1)*nx + (k0i-1)*nxny;
    idx110 = i1i + (j1i-1)*nx + (k0i-1)*nxny;
    idx001 = i0i + (j0i-1)*nx + (k1i-1)*nxny;
    idx101 = i1i + (j0i-1)*nx + (k1i-1)*nxny;
    idx011 = i0i + (j1i-1)*nx + (k1i-1)*nxny;
    idx111 = i1i + (j1i-1)*nx + (k1i-1)*nxny;

    c000 = vol(idx000); c100 = vol(idx100);
    c010 = vol(idx010); c110 = vol(idx110);
    c001 = vol(idx001); c101 = vol(idx101);
    c011 = vol(idx011); c111 = vol(idx111);

    c00 = (1-dx).*c000 + dx.*c100;
    c10 = (1-dx).*c010 + dx.*c110;
    c01 = (1-dx).*c001 + dx.*c101;
    c11 = (1-dx).*c011 + dx.*c111;

    c0 = (1-dy).*c00 + dy.*c10;
    c1 = (1-dy).*c01 + dy.*c11;

    val = (1-dz).*c0 + dz.*c1;
end

% ----------------------------------------------------------------------
% Progress utilities (MATLAB-only output; codegen-safe stubs)
% ----------------------------------------------------------------------
function t0 = progress_tic()
%#codegen
    if coder.target('MATLAB')
        t0 = tic;
    else
        t0 = 0; % dummy for codegen
    end
end

function progress_update(tag, k, K, t0)
%#codegen
% Print a single-line textual progress bar with ETA (MATLAB only).
    if coder.target('MATLAB')
        if k <= 0
            fprintf('%s: initializing...\n', tag);
        end
        pct = max(0, min(1, double(k)/max(1,double(K))));
        nbar = 40; nfill = floor(pct * nbar);
        bar = [repmat('#',1,nfill) repmat('.',1,nbar-nfill)];
        dt = toc(t0);
        if pct > 0
            eta = dt*(1-pct)/pct;
        else
            eta = NaN;
        end
        fprintf('\r%s [%s] %3.0f%%  | elapsed %.1fs | ETA %.1fs', ...
            tag, bar, pct*100, dt, eta);
        if k == K
            fprintf('\n');
        end
    end
end