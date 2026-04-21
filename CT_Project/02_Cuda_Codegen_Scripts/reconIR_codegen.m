function hybridIR_vol = reconIR_codegen( ...
        P, angles_rad, DSD, DSO, du, dv, u0_pixels, v0_pixels, ...
        dx, dy, dz, x0, ...
        numIter, alpha, lambda, gaussSigma, ...
        padFactor)
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
    [nx, ny, nz] = size(x0);
    x           = single(x0);                      % current estimate
    numIter     = int32(numIter);
    alpha       = single(alpha);
    lambda      = single(lambda);
    gaussSigma  = single(gaussSigma);
    padFactor   = single(max(1, padFactor));

    [nu, nv, nViews] = size(P); %#ok<ASGLU>


    % Geometry bundles
    det_pix   = single([du dv]);
    vox_size  = single([dx dy dz]);

    % -------- Iterative refinement with progress --------

    for it = int32(1) : numIter
        % Forward projection A*x  (GPU/CPU depending on useGPU)
        % NOTE: numSamples is ignored inside forward projector (ray-box + voxel step).
        Ax = forwardProject_CBCT_cg( ...
            x, angles_rad, DSD, DSO, det_pix, vox_size, ...
            int32(nu), int32(nv), u0_pixels, v0_pixels ...
            );

        % Residual
        res = P - Ax;   % single

        % Backproject gradient ~ A' * res  (cuFFT for filtering if useGPU=true)
        grad = simpleFDK_CBCT_vox_cg( ...
            res, angles_rad, DSD, DSO, det_pix, vox_size, ...
            nx, ny, nz, int32(1024), ...
            u0_pixels, v0_pixels, padFactor);

        % Gaussian prior (separable, codegen-safe)
        if lambda > 0 && gaussSigma > 0
            smooth = gaussian_denoise3d_cg(x, gaussSigma);
            reg    = lambda * (x - smooth);
        else
            reg = zeros(size(x), 'like', x);
        end

        % Update
        x = x + alpha * (grad - reg);
    end

    hybridIR_vol = x;
end

% ======================================================================
% Cone-beam forward projector (GPU/CPU; trilinear; RAY-BOX + VOXEL STEP)
% ======================================================================
function Ax = forwardProject_CBCT_cg(volume, angles_rad, DSD, DSO, det_pixel_size, voxel_size, ...
        nu, nv, u0, v0)
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

    nViews = int32(numel(angles_rad));
    Ax = zeros(nu, nv, nViews, 'single');

    % Step size ~ voxel size in mm
    step = single(max( min([dx,dy,dz]), eps('single')));

    % MATLAB runtime GPU: keep volume on device persistently

    % Inline fwdproj_view_GPU_codegen_raybox
    for ia = int32(1) : nViews
        for sy = 1:nv
            for sx = 1:nu
                theta = angles_rad(double(ia));
                ca = cos(theta); sa = sin(theta);
                rhat = [ca,  sa, 0];
                uhat = [-sa, ca, 0];
                vhat = [0,   0,  1];

                src  = [-DSO * ca, -DSO * sa, 0];
                D0x  = src(1) + rhat(1)*DSD;
                D0y  = src(2) + rhat(2)*DSD;
                D0z  = src(3) + rhat(3)*DSD;

                ub = U2(sx, sy);
                vb = V2(sx, sy);

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
                Ax(sx,sy,ia) = acc;
            end
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

% ======================================================================
% Bilinear backprojector (filters via cuFFT when gpuConfig MEX or MATLAB GPU)
% ======================================================================
function recon_volume = simpleFDK_CBCT_vox_cg( ...
        projections, angles, dsd, dso, det_pixel_size, voxel_size, ...
        nx, ny, nz, Nf_fft, ...
        u0, v0, padFactor)
    %#codegen
    % SIMPLEFDK_CBCT_VOX_CG
    % Streamed, chunked FFT across views with padFactor-controlled effective FFT size.

    angles = angles(:)';                          % 1x360
    [nu, nv, nViews] = size(projections); % 600x600x360

    % Detector & voxel sizes (single)
    if isscalar(det_pixel_size)
        du = det_pixel_size; dv = det_pixel_size;
    else
        du = det_pixel_size(1); dv = det_pixel_size(2);
    end
    if isscalar(voxel_size)
        dx = voxel_size; dy = voxel_size; dz = voxel_size;
    else
        dx = voxel_size(1); dy = voxel_size(2); dz = voxel_size(3);
    end

    % Principal point defaults (type-safe)
    if isnan(u0)
        u0 = cast((double(nu) + 1.0) / 2.0, 'like', u0);
    end
    if isnan(v0)
        v0 = cast((double(nv) + 1.0) / 2.0, 'like', v0);
    end

    % Image grids (iso-centered) -> single
    x = ((0:double(nx)-1) - (double(nx)-1)/2) * double(dx);
    y = ((0:double(ny)-1) - (double(ny)-1)/2) * double(dy);
    z = ((0:double(nz)-1) - (double(nz)-1)/2) * double(dz);
    x = cast(x, 'like', dx);
    y = cast(y, 'like', dy);
    z = cast(z, 'like', dz);
    [X2D, Y2D] = meshgrid(x, y);                 % 600x600

    % ------------------------------------------------------------
    % 2D cosine weighting (fan + cone)
    % ------------------------------------------------------------
    u_mm = (single(1:nu) - u0) * du;             % single
    v_mm = (single(1:nv) - v0) * dv;             % single
    [UU, VV] = ndgrid(u_mm, v_mm);               % 600x600
    cosw_2d  = dsd ./ sqrt(dsd^2 + UU.^2 + VV.^2); % 600x600
    projections = single(projections) .* reshape(cosw_2d, [nu nv 1]);  % 600x600x360

    % ------------------------------------------------------------
    % Angle weights (single)
    % ------------------------------------------------------------
    if nViews > 1
        d = diff(angles);
        if max(abs(d - mean(d))) < single(1e-6)
            dth = repmat(mean(d), 1, nViews);    % single
        else
            dth = zeros(1, nViews, 'like', angles);
            dth(1) = d(1);
            if nViews > 2
                dth(2:end-1) = (d(1:end-1) + d(2:end))/2;
            end
            dth(end) = d(end);
        end
    else
        dth = cast(2*pi, 'like', angles);        % single(2*pi)
    end

    denom_eps = max(single(1e-6), single(1e-6) * dso);

    % ------------------------------------------------------------
    % STREAMED FILTER + IMMEDIATE BACKPROJECTION (chunked across views)
    % ------------------------------------------------------------

    % 1) Clamp padFactor to >=1 and compute effective FFT length
    padFactor = max(single(1), padFactor);


    % Target FFT length from padFactor*nu, rounded up to next power-of-two
    Nf_target = pow2ceil_from_scalar(double(padFactor) * double(nu));  % double
    % Effective length cannot exceed Nf_fft and cannot be < nu
    Nf_eff_d  = min(double(Nf_fft), max(double(nu), Nf_target));       % double
    Nf_eff    = int32(Nf_eff_d);                                       % int32
    pre       = floor((double(Nf_eff) - double(nu)) / 2);              % padding

    % 2) Build frequency response H for Nf_eff (single)
    k  = (0:double(Nf_eff)-1).';                 % double
    df = 1 / (double(Nf_eff) * double(du));      % double
    f  = zeros(double(Nf_eff),1,'like',single(0));
    half = floor(double(Nf_eff)/2);
    f(1:half+1) = single((0:half).' * df);
    if Nf_eff > 2
        f(half+2:end) = single((-(half-1):-1).' * df);
    end
    ramp = abs(f);                                % 1024x1

    H = ramp;
    H = reshape(H, [Nf_eff 1 1]);                % 1024x1

    % 3) Recon accumulator
    recon_yx_z = zeros(ny, nx, nz, 'like', projections);  % 600x600x360

    % 4) Chunk across views to bound memory
    nViews_i32    = int32(nViews);

    % Generate Filt in one shot
    Pp_chunk = zeros(Nf_fft, nv, nViews, 'like', projections);
    % Write for-loops to parallelize the copy
    coder.gpu.kernelfun;
    coder.gpu.kernel;
    for k = 1:nViews
        coder.gpu.kernel;
        for j = 1:nv
            coder.gpu.kernel;
            for i = 1:nu
                Pp_chunk(pre+i, j, k) = projections(i,j,k);
            end
        end
    end
    %Pp_chunk(pre+1:pre+nu, :, :) = projections;
    Pseg = Pp_chunk(1:double(Nf_eff), :, :);
    Pf = fft(Pseg,[],1);
    Pf = Pf.*H;
    Pseg = real(ifft(Pf, [], 1));
    Pp_chunk(1:double(Nf_eff), :, :) = Pseg;
    Filt_chunk = Pp_chunk(pre+1:pre+nu, :, :); %600x600x8

    % Loop over view chunks
    for ia0 = 1 : nViews_i32
        ia = ia0;                         % int32 view index
        % Geometry rotation for this view
        ca = cos(angles(double(ia))); sa = sin(angles(double(ia)));
        Xr =  ca * X2D - sa * Y2D;            % [ny,nx] single
        Yr =  sa * X2D + ca * Y2D;            % [ny,nx] single

        denom = dso - Xr;
        valid = denom > denom_eps;
        denom = max(denom, denom_eps);
        t = dsd ./ denom;                      % magnification

        u_idx_base = u0 + (t .* (Yr / du));    % [ny,nx] single

        % z-batches to bound temporaries
        % TODO: move the recon_yx_z update out the loop.
        % produce contrib of 600x600*360
        contrib = coder.nullcopy(zeros(ny, nx, nz, 'like', projections));
        curView = Filt_chunk(:,:,ia);
        wbp = (dso ./ denom).^2;
        for k_ = 1:nz
            for j_ = 1:nx
                for i_ = 1:ny
                    u_idx = u_idx_base(i_,j_);
                    v_idx = v0+ t(i_,j_) * z(k_) / dv;
                    u_idx = max(single(1), min(single(ny-1), u_idx));
                    v_idx = max(single(1), min(single(nx-1), v_idx));
                    u0_ = floor(u_idx);
                    v0_ = floor(v_idx);
                    du_ = u_idx - u0_;
                    dv_ = v_idx - v0_;
                    u1_ = u0_+1;
                    v1_ = v0_+1;
                    f00 = curView(u0_, v0_);
                    f10 = curView(u1_, v0_);
                    f01 = curView(u0_, v1_);
                    f11 = curView(u1_, v1_);
                    val = (1-du_).*(1-dv_).*f00 + du_.*(1-dv_).*f10 + (1-du_).*dv_.*f01 + du_.*dv_.*f11;
                    if ~isfinite(val)
                        val = zeros('like', projections);
                    end
                    val = val * wbp(i_,j_) * dth(ia);
                    if ~valid(i_,j_)
                        val = zeros('like', projections);
                    end
                    contrib(i_,j_,k_)  = val;
                end
            end
        end
        % accumulate
        recon_yx_z = recon_yx_z + contrib;
    end

    % Finalize
    recon_yx_z = recon_yx_z / cast(2*pi, 'like', recon_yx_z);
    recon_yx_z(~isfinite(recon_yx_z)) = cast(0, 'like', recon_yx_z);

    recon_volume = permute(recon_yx_z, [2 1 3]); % [nx,ny,nz] single
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