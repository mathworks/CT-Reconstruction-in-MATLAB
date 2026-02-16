function FDKvol = reconFDK(inputMat, varargin)
% RUN_FDK Load CT_*.mat, run FDK (GPU if available), return volume
% Usage:
%   vol = run_FDK('CT_low.mat');                 % auto GPU
%   vol = run_FDK('CT_low.mat','UseGPU',false);  % force CPU
%   vol = run_FDK('CT_low.mat','UseGPU',true);   % force GPU (if available)
%
% Optional Name-Value pairs are forwarded to simpleFDK_CBCT_vox:
%   'UseGPU','Verbose','PadFactor','BatchZ','u0','v0'

    S = load(inputMat);

    % Unpack (your file uses these names)
    P          = S.P;                 % ensure shape [nu, nv, nViews]
    angles_rad = S.angles_rad(:)';    % [1 x views]
    DSO        = S.DSO; DSD = S.DSD;
    du         = S.du;  dv  = S.dv;
    u0         = S.u0_pixels; v0 = S.v0_pixels;

    % Grid & voxels
    dx = 0.5; dy = 0.5; dz = 0.5;  % mm
    fov_xy_mm = 300;
    nx = round(fov_xy_mm/dx);
    ny = round(fov_xy_mm/dy);
    Lz_mm = 180; nz = max(1, round(Lz_mm / dz));

    % Optional: auto-detect if P is [views, nv, nu] and fix layout
    if size(P,1) == numel(angles_rad) && ndims(P)==3
        % Convert [views, nv, nu] -> [nu, nv, views]
        P = permute(P, [3 2 1]);
    end

    % Run auto-GPU FDK
    FDKvol = simpleFDK_CBCT_vox(P, angles_rad, DSD, DSO, ...
                                [du dv], [dx dy dz], nx, ny, nz, ...
                                'ramp', 'u0', u0, 'v0', v0, varargin{:});
end