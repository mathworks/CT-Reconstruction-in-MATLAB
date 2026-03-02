function Y = reconDL_codegen(X)
%#codegen
% CUDA entry-point that replicates the TIFF preprocessing used in training:
% - Per-slice 1/99 percentile normalization
% - Quantize to uint16, then back to single (im2single)
% - Pad to 608x608x1
% - Predict with persistent net

persistent netInitialized net;

if isempty(netInitialized)
    net = coder.loadDeepLearningNetwork('best_unet2d.mat','net'); % adjust var name if needed
    netInitialized = true;
end

% Ensure single
if ~isa(X,'single'); X = single(X); end

% ---- TIFF-equivalent per-slice preprocessing ----
% 1–2) percentile normalize to [0,1]
p1  = prctile_local(X, single(1));    % codegen-safe percentile (see helper)
p99 = prctile_local(X, single(99));
if p99 > p1
    scaled = (X - p1) ./ (p99 - p1 + eps('single'));
else
    vmin = min(X(:)); vmax = max(X(:));
    if vmax > vmin
        scaled = (X - vmin) ./ (vmax - vmin + eps('single'));
    else
        scaled = zeros(size(X), 'single');
    end
end
scaled = min(max(scaled, 0), 1);

% 3) quantize to uint16 (like im2uint16)
u16 = im2uint16_local(scaled);

% 4) dequantize to single (like im2single on uint16 TIFF)
Xn = single(u16) / 65535;  % exact im2single(u16) behavior

% 5) pad to 608x608x1
Xpad = padTo608_local(Xn);

% 6) predict
Y = predict(net, Xpad);

end

% ----------------- helpers (codegen-friendly) -----------------

function Xout = padTo608_local(X)
% Pad to [608 608 1] with 'replicate' borders
X = single(X);
[h,w,~] = size(X);
if h==608 && w==608
    Xout = reshape(X,[608 608 1]);
    return;
end
hp = 608 - h;  wp = 608 - w;
preH = floor(hp/2); postH = hp - preH;
preW = floor(wp/2); postW = wp - preW;
Xout = padarray(X,[preH preW],'replicate','pre');
Xout = padarray(Xout,[postH postW],'replicate','post');
Xout = reshape(Xout,[608 608 1]);
end

function p = prctile_local(X, pct)
% Simple, codegen-safe percentile approximation via sorting.
% For large arrays this is okay at inference; can be replaced by histogram-based.
x = X(:);
x = sort(x,'ascend');  % Supported by codegen for numeric vectors
N = numel(x);
if N == 0
    p = single(0);
    return;
end
% MATLAB percentile definition ~ linear interpolation between ranks
pos = (double(pct)/100) * (N - 1) + 1;
lo  = floor(pos);
hi  = ceil(pos);
alpha = single(pos - lo);
if lo < 1, lo = 1; end
if hi < 1, hi = 1; end
if lo > N, lo = N; end
if hi > N, hi = N; end
xlo = x(lo); xhi = x(hi);
p = xlo + alpha * (xhi - xlo);
end

function u16 = im2uint16_local(x)
% Equivalent to im2uint16 for x in [0,1].
% Round to nearest integer in [0,65535].
y = round(min(max(x,0),1) * 65535);
u16 = uint16(y);
end