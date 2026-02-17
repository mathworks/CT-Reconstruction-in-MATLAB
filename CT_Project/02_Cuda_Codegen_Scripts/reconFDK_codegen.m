function FDKvol = reconFDK_codegen( ...
    P, angles_rad, DSD, DSO, du, dv, u0_pixels, v0_pixels, ...
    dx, dy, dz, nx, ny, nz, filter_type, ...
    Nf_fft, padFactor, batchZ, useGPU, verbose)
%#codegen
% Entry point for GPU Coder. No file I/O here.
% P: [nu,nv,nViews] single
% angles_rad: [1,nViews] single (radians)
% filter_type: 'ramp'|'hamming'|'shepp-logan' (coder.Constant recommended)
% Nf_fft: int32 (>= nu). Use power-of-two for stability (e.g., 1024)

    % Types & shapes
    P          = single(P);                       % [nu,nv,nViews]
    angles_rad = single(angles_rad(:).');         % [1,nViews]
    DSD        = single(DSD);
    DSO        = single(DSO);
    du         = single(du);  dv = single(dv);
    u0_pixels  = single(u0_pixels);
    v0_pixels  = single(v0_pixels);
    dx         = single(dx);  dy = single(dy);  dz = single(dz);
    padFactor  = single(padFactor);
    batchZ     = int32(batchZ);

    % Pass-through (NaN allowed): defaults handled in the core
    u0_use = u0_pixels;
    v0_use = v0_pixels;

    det_pix  = single([du dv]);
    vox_size = single([dx dy dz]);

    FDKvol = simpleFDK_CBCT_vox_cg( ...
        P, angles_rad, DSD, DSO, det_pix, vox_size, ...
        int32(nx), int32(ny), int32(nz), filter_type, Nf_fft, ...
        u0_use, v0_use, logical(verbose), padFactor, batchZ, logical(useGPU));
end


% ======================================================================
%                       GPU-Coder friendly core
% ======================================================================
function recon_volume = simpleFDK_CBCT_vox_cg( ...
    projections, angles, dsd, dso, det_pixel_size, voxel_size, ...
    nx, ny, nz, filter_type, Nf_fft, ...
    u0, v0, verbose, padFactor, batchZ, useGPU)
%#codegen
% SIMPLEFDK_CBCT_VOX_CG
% Streamed, chunked FFT across views with padFactor-controlled effective FFT size.

    angles = angles(:)';                          % [1,nViews] row
    [nu, nv, nViews] = size(projections);

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
    [X2D, Y2D] = meshgrid(x, y);                 % [ny,nx] single

    % ------------------------------------------------------------
    % 2D cosine weighting (fan + cone)
    % ------------------------------------------------------------
    u_mm = (single(1:nu) - u0) * du;             % single
    v_mm = (single(1:nv) - v0) * dv;             % single
    [UU, VV] = ndgrid(u_mm, v_mm);               % [nu,nv] single
    cosw_2d  = dsd ./ sqrt(dsd^2 + UU.^2 + VV.^2);
    projections = single(projections) .* reshape(cosw_2d, [nu nv 1]);  % single

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
    if verbose && coder.target('MATLAB')
        fprintf('[FDK] nu=%d nv=%d V=%d | nx=%d ny=%d nz=%d | du=%.3f dv=%.3f | dx=%.3f dy=%.3f dz=%.3f | GPU=%d | padFactor=%.2f | Nf_fft=%d\n', ...
            nu, nv, nViews, nx, ny, nz, du, dv, dx, dy, dz, useGPU, padFactor, Nf_fft);
    end

    % ------------------------------------------------------------
    % STREAMED FILTER + IMMEDIATE BACKPROJECTION (chunked across views)
    % ------------------------------------------------------------

    % 1) Clamp padFactor to >=1 and compute effective FFT length
    padFactor = max(single(1), padFactor);
    coder.internal.errorIf(Nf_fft < int32(nu), ...
        'Nf_fft (%d) must be >= nu (%d).', Nf_fft, int32(nu));

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
    ramp = abs(f);                                % single

    switch lower(filter_type)
        case 'ramp'
            H = ramp;
        case 'shepp-logan'
            H = ramp .* sinc(f * du);            % single supported
        case 'hamming'
            phi = single((k/(double(Nf_eff)-1)) - 0.5);
            H   = ramp .* (single(0.54) + single(0.46) * cos(single(2*pi) * phi));
        otherwise
            coder.internal.errorIf(true, 'simpleFDK:UnknownFilter', 'Unknown filter type');
    end
    H = reshape(H, [Nf_eff 1 1]);                % [Nf_eff,1,1], single

    % 3) Recon accumulator
    recon_yx_z = zeros(ny, nx, nz, 'like', projections);  % single

    % 4) Chunk across views to bound memory
    maxChunkViews = int32(8);                     % tune to 8/16 as needed
    nViews_i32    = int32(nViews);

    % Loop over view chunks
    for ia0 = int32(1) : maxChunkViews : nViews_i32
        ia1    = min(nViews_i32, ia0 + maxChunkViews - int32(1));
        nChunk = ia1 - ia0 + int32(1);

        % Allocate with Nf_fft (compile-time shape), operate on 1:Nf_eff rows
        Pp_chunk = zeros(Nf_fft, nv, nChunk, 'single');
        Pp_chunk(pre+1:pre+nu, :, :) = projections(:,:,double(ia0):double(ia1));

        % FFT filter only the used rows (1:Nf_eff)
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
            % In generated MEX with gpuConfig, fft/ifft map to cuFFT
            Pseg = Pp_chunk(1:double(Nf_eff), :, :);
            Pf   = fft(Pseg, [], 1);
            Pf   = Pf .* H;
            Pseg = real(ifft(Pf, [], 1));
            Pp_chunk(1:double(Nf_eff), :, :) = Pseg;
        end

        % Crop back to [nu, nv, nChunk] using Nf_eff padding
        Filt_chunk = Pp_chunk(pre+1:pre+nu, :, :);

        % Backproject each filtered view in the chunk immediately
        for j = int32(0) : nChunk - int32(1)
            ia = ia0 + j;                         % int32 view index

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
            for k0 = int32(1) : batchZ : nz
                k1 = min(nz, k0 + batchZ - int32(1));
                nb = k1 - k0 + int32(1);

                Zk = reshape(z(double(k0):double(k1)), 1, 1, double(nb));  % [1,1,nb] single

                % Expand indices for all Z in batch via repmat (codegen-safe)
                u_idx = repmat(u_idx_base, 1, 1, double(nb));              % [ny,nx,nb]
                v_idx = v0 + (t .* (Zk / dv));                             % [ny,nx,nb]

                % --------- Bilinear interpolation (codegen-safe) ----------
                curView = Filt_chunk(:,:,double(j)+1);                     % [nu,nv] single
                samp = bilinearSample2D(curView, u_idx, v_idx);            % [ny,nx,nb]
                samp(~isfinite(samp)) = cast(0, 'like', samp);
                % ---------------------------------------------------------

                wbp = (dso ./ denom).^2;                                   % [ny,nx]
                contrib = samp .* repmat(wbp, 1, 1, double(nb)) * dth(double(ia));

                if any(~valid(:))
                    mask3 = repmat(~valid, 1, 1, double(nb));
                    contrib(mask3) = cast(0, 'like', contrib);
                end

                recon_yx_z(:,:,k0:k1) = recon_yx_z(:,:,k0:k1) + contrib;
            end
        end
        % (Chunk done)
    end

    % Finalize
    recon_yx_z = recon_yx_z / cast(2*pi, 'like', recon_yx_z);
    recon_yx_z(~isfinite(recon_yx_z)) = cast(0, 'like', recon_yx_z);

    recon_volume = permute(recon_yx_z, [2 1 3]); % [nx,ny,nz] single
end


% ----------------------------------------------------------------------
% Helper: power-of-two ceiling for a positive scalar (codegen-safe)
% ----------------------------------------------------------------------
function p2 = pow2ceil_from_scalar(x)
%#codegen
    % Ensure at least 1
    target = max(1.0, x);
    p = 1.0;
    while p < target
        p = p * 2.0;
    end
    p2 = p;  % double
end


% ----------------------------------------------------------------------
% Helper: Bilinear sampler (host-side, MATLAB Coder safe)
% ----------------------------------------------------------------------
function val = bilinearSample2D(img, u, v)
%#codegen
% img: [nu,nv] single
% u,v: same size, 1-based coordinates in pixel units (single)
% returns val same size as u/v
    nu = size(img,1); nv = size(img,2);

    % Clamp to valid range so neighbors exist (u+1, v+1)
    u = max(single(1), min(single(nu - 1), u));
    v = max(single(1), min(single(nv - 1), v));

    u0 = floor(u); v0 = floor(v);
    du = u - u0;   dv = v - v0;

    u0i = int32(u0); v0i = int32(v0);
    u1i = u0i + 1;  v1i = v0i + 1;

    % Linear indices (column-major)
    sz = [nu nv];
    f00 = img(sub2ind(sz, u0i, v0i));
    f10 = img(sub2ind(sz, u1i, v0i));
    f01 = img(sub2ind(sz, u0i, v1i));
    f11 = img(sub2ind(sz, u1i, v1i));

    val = (1-du).*(1-dv).*f00 + du.*(1-dv).*f10 + (1-du).*dv.*f01 + du.*dv.*f11;
end