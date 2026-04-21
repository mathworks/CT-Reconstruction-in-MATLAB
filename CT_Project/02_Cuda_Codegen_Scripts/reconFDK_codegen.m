function FDKvol = reconFDK_codegen( ...
        P, angles_rad, DSD, DSO, du, dv, u0_pixels, v0_pixels, ...
        dx, dy, dz, ...
        padFactor)
    %#codegen
    % Entry point for GPU Coder. No file I/O here.
    % P: [nu,nv,nViews] single
    % angles_rad: [1,nViews] single (radians)
    % filter_type: 'ramp'|'hamming'|'shepp-logan' (coder.Constant recommended)
    % Nf_fft: int32 (>= nu). Use power-of-two for stability (e.g., 1024)

    % Types & shapes
    P          = single(P);                       % 600x600x360
    % is this true?
    [ny,nx,nz] = size(P);
    Nf_fft = 1024;
    angles_rad = single(angles_rad(:).');         % 1x360
    DSD        = single(DSD);
    DSO        = single(DSO);
    du         = single(du);  dv = single(dv);
    u0_pixels  = single(u0_pixels);
    v0_pixels  = single(v0_pixels);
    dx         = single(dx);  dy = single(dy);  dz = single(dz);
    padFactor  = single(padFactor);

    % Pass-through (NaN allowed): defaults handled in the core
    u0_use = u0_pixels;
    v0_use = v0_pixels;

    det_pix  = single([du dv]);
    vox_size = single([dx dy dz]);

    FDKvol = simpleFDK_CBCT_vox_cg( ...
        P, angles_rad, DSD, DSO, det_pix, vox_size, ...
        int32(nx), int32(ny), int32(nz), Nf_fft, ...
        u0_use, v0_use, padFactor);
end


% ======================================================================
%                       GPU-Coder friendly core
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
    
    for k = 1:nViews
        for j = 1:nv
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
