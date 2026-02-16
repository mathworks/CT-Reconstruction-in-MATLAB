function [yPredVol, net, info] = reconDL(fdkIn, tgtIn, varargin)
% reconDL  CBCT 2D U-Net training + inference from external inputs.
%          Accepts volumes or MAT file paths. Returns yPredVol and net.
%
% Signature:
%   [yPredVol, net, info] = reconDL(fdkIn, tgtIn, Name,Value,...)
%
% fdkIn : either a 3D volume [H W Z] or a path to a .mat file
% tgtIn : either a 3D volume [H W Z] or a path to a .mat file
%
% Name-Value options (all optional):
%   'FDKVar'            : variable name inside FDK .mat (default: auto)
%   'TGTVar'            : variable name inside TGT .mat (default: auto)
%   'DoPatchTraining'   : true | false (default false -> full-slice)
%   'PatchSize'         : [h w] (default [256 256])
%   'PatchesPerImage'   : integer (default 8)
%   'ApplyInputArtifacts': true | false (default false)
%   'ValFraction'       : fraction [0,1] (default 0.2)
%   'BaseChannels'      : integer (default 32)
%   'EncoderDepth'      : integer (default 4)
%   'RandomSeed'        : integer (default 42)
%   'OutRoot'           : folder for outputs (default fullfile(pwd,'cbct_run_mid'))
%   'ModelFile'         : filename for saved net (default <OutRoot>/best_unet2d.mat)
%   'ReconFile'         : filename for saved yPredVol (default <OutRoot>/dl_recon_3d.mat)
%   'PrintShapes'       : true | false (default true)
%   'ExecEnv'           : 'auto'|'gpu'|'cpu' (default 'auto')
%
% Output:
%   yPredVol : [H W Z] single — DL reconstruction of the FDK volume
%   net      : trained DAGNetwork
%   info     : struct with useful paths and selections

%% ---------------------------- Parse NV ---------------------------------
p = inputParser;
addParameter(p, 'FDKVar', '', @(s)ischar(s)||isstring(s));
addParameter(p, 'TGTVar', '', @(s)ischar(s)||isstring(s));
addParameter(p, 'DoPatchTraining', false, @(x)islogical(x)||isnumeric(x));
addParameter(p, 'PatchSize', [256 256], @(x)isvector(x)&&numel(x)==2);
addParameter(p, 'PatchesPerImage', 8, @(x)isscalar(x)&&x>=1);
addParameter(p, 'ApplyInputArtifacts', false, @(x)islogical(x)||isnumeric(x));
addParameter(p, 'ValFraction', 0.2, @(x)isscalar(x)&&x>=0&&x<1);
addParameter(p, 'BaseChannels', 32, @(x)isscalar(x)&&x>=8);
addParameter(p, 'EncoderDepth', 4, @(x)isscalar(x)&&x>=2);
addParameter(p, 'RandomSeed', 42, @(x)isscalar(x)&&x>=0);
addParameter(p, 'OutRoot', fullfile(pwd,'cbct_run_mid'), @(s)ischar(s)||isstring(s));
addParameter(p, 'ModelFile', '', @(s)ischar(s)||isstring(s));
addParameter(p, 'ReconFile', '', @(s)ischar(s)||isstring(s));
addParameter(p, 'PrintShapes', true, @(x)islogical(x)||isnumeric(x));
addParameter(p, 'ExecEnv', 'auto', @(s)any(strcmpi(s,{'auto','gpu','cpu'})));
parse(p, varargin{:});
opt = p.Results;

rng(opt.RandomSeed);

%% --------------------- Resolve inputs (volumes/paths) -------------------
xFDK_raw = resolveVolumeInput(fdkIn, opt.FDKVar, 'FDK');
xTGT_raw = resolveVolumeInput(tgtIn, opt.TGTVar, 'TGT');

assert(isequal(size(xFDK_raw), size(xTGT_raw)), ...
    'FDK and target volumes must have the same size.');

xFDK_raw = single(xFDK_raw);
xTGT_raw = single(xTGT_raw);
[H0, W0, Z] = size(xFDK_raw);

%% -------------------------- Output paths --------------------------------
OUT_ROOT   = char(opt.OutRoot);
if isempty(opt.ModelFile), MODEL_FILE = fullfile(OUT_ROOT, 'best_unet2d.mat'); else, MODEL_FILE=char(opt.ModelFile); end
if isempty(opt.ReconFile), RECON_FILE = fullfile(OUT_ROOT, 'dl_recon_3d.mat'); else, RECON_FILE=char(opt.ReconFile); end

if exist(OUT_ROOT,'dir'), rmdir(OUT_ROOT,'s'); end
mkdir(fullfile(OUT_ROOT,'train','input'));
mkdir(fullfile(OUT_ROOT,'train','target'));
mkdir(fullfile(OUT_ROOT,'val','input'));
mkdir(fullfile(OUT_ROOT,'val','target'));
trainIn = fullfile(OUT_ROOT,'train','input');
trainTg = fullfile(OUT_ROOT,'train','target');
valIn   = fullfile(OUT_ROOT,'val','input');
valTg   = fullfile(OUT_ROOT,'val','target');

%% ------------------------ Sizes & stride --------------------------------
NET_STRIDE     = 2^opt.EncoderDepth;
DO_PATCH       = logical(opt.DoPatchTraining);
PATCH_SIZE     = opt.PatchSize;
PATCHES_PER_IM = opt.PatchesPerImage;
PRINT_SHAPES   = logical(opt.PrintShapes);

if DO_PATCH
    assert(all(mod(PATCH_SIZE, NET_STRIDE)==0), ...
        'Patch size must be divisible by network stride (%d).', NET_STRIDE);
    inputSize = [PATCH_SIZE 1];
    TARGET_HW = PATCH_SIZE;
else
    TARGET_HW = ceil([H0 W0] / NET_STRIDE) * NET_STRIDE;
    inputSize = [TARGET_HW 1];
end

if PRINT_SHAPES
    fprintf("\n=== Shape Debug ===\n");
    fprintf("Mode: %s\n", ternary(DO_PATCH,'PATCH','FULL-SLICE'));
    fprintf("Original:  [%d %d]\n", H0, W0);
    fprintf("Padded:    [%d %d]\n", TARGET_HW(1), TARGET_HW(2));
    fprintf("Net input: [%d %d 1]\n", inputSize(1), inputSize(2));
    fprintf("===================\n\n");
end

%% --------- Compute slice stats for robust train/val selection ----------
sliceStd   = zeros(1,Z,'single');
sliceRange = zeros(1,Z,'single');
sliceCov   = zeros(1,Z,'single');

for k = 1:Z
    sl = xFDK_raw(:,:,k);
    p1 = prctile(sl(:),1); p99 = prctile(sl(:),99);
    sliceRange(k) = max(p99 - p1, 0);
    sliceStd(k)   = std(sl(:), 0, 'all');
    try
        level = graythresh(mat2gray(sl));
    catch
        level = 0.1;
    end
    m = mat2gray(sl) > level;
    sliceCov(k) = mean(m(:));
end
normalize01 = @(x) ((x - min(x(:))) ./ max(eps, (max(x(:))-min(x(:)))));

VAL_FRACTION = max(opt.ValFraction, 0.01);
valCount     = max(1, round(VAL_FRACTION * Z));
STD_MIN = 1e-4; RNG_MIN = 5e-4; COV_MIN = 0.02;

valCand = find(sliceStd >= STD_MIN & sliceRange >= RNG_MIN & sliceCov >= COV_MIN);
if numel(valCand) >= valCount
    valCand = valCand(randperm(numel(valCand)));
    valIdx  = valCand(1:valCount);
else
    score = normalize01(sliceStd) + normalize01(sliceRange) + normalize01(sliceCov);
    [~, ord] = sort(score, 'descend');
    valIdx   = ord(1:valCount);
end
isVal = false(1,Z); isVal(valIdx) = true;

fprintf("Selected %d validation slices (requested %d). Median raw-std=%.3g, min raw-std in val=%.3g, median cov=%.3g\n", ...
    nnz(isVal), valCount, median(sliceStd), min(sliceStd(isVal)), median(sliceCov(isVal)));

% Training selection
isTrain   = ~isVal;
trainGood = (sliceStd >= STD_MIN) & (sliceRange >= RNG_MIN) & (sliceCov >= COV_MIN) & ~isVal;
if nnz(trainGood) >= round(0.5 * nnz(~isVal))
    isTrain = trainGood;
else
    notVal = find(~isVal);
    score  = normalize01(sliceStd) + normalize01(sliceRange) + normalize01(sliceCov);
    [~, ordNV] = sort(score(notVal), 'descend');
    keepN = max(1, round(0.5 * numel(notVal)));
    pick  = notVal(ordNV(1:keepN));
    isTrain = false(1,Z); isTrain(pick) = true;
end
fprintf("Train selection: kept %d / %d non-val slices (%.1f%%)\n", ...
    nnz(isTrain), nnz(~isVal), 100*nnz(isTrain)/max(1,nnz(~isVal)));

%% --------------------- Export TIFFs for training -----------------------
for k = 1:Z
    if ~(isVal(k) || isTrain(k)), continue; end

    Xi = xFDK_raw(:,:,k);
    Yi = xTGT_raw(:,:,k);

    if ~DO_PATCH
        [Xi, ~] = padToTarget(Xi, TARGET_HW);
        [Yi, ~] = padToTarget(Yi, TARGET_HW);
    end

    Xi_u16 = toUint16PerSliceSafe(Xi, 1, 99);
    Yi_u16 = toUint16PerSliceSafe(Yi, 1, 99);

    if isVal(k)
        imwrite(Xi_u16, fullfile(valIn,   sprintf('in_%04d.tif',k)));
        imwrite(Yi_u16, fullfile(valTg,   sprintf('tg_%04d.tif',k)));
    else
        imwrite(Xi_u16, fullfile(trainIn, sprintf('in_%04d.tif',k)));
        imwrite(Yi_u16, fullfile(trainTg, sprintf('tg_%04d.tif',k)));
    end
end

% Quick export checks
imdsTrainX_dbg = imageDatastore(trainIn);
imdsValX_dbg   = imageDatastore(valIn);
if isempty(imdsTrainX_dbg.Files), error('Train folder empty after export.'); end
if isempty(imdsValX_dbg.Files),   error('Val folder empty after export.');   end

%% ------------------------- Datastores ----------------------------------
imdsTrainX = imageDatastore(trainIn);
imdsTrainY = imageDatastore(trainTg);
imdsValX   = imageDatastore(valIn);
imdsValY   = imageDatastore(valTg);

dsTrain = combine(imdsTrainX, imdsTrainY);
dsVal   = combine(imdsValX, imdsValY);

APPLY_INPUT_ARTIFACTS = logical(opt.ApplyInputArtifacts);
dsTrainPrep = transform(dsTrain, @(b) preprocessBatch_train(b, APPLY_INPUT_ARTIFACTS));
dsValPrep   = transform(dsVal,   @preprocessBatch_val);

if DO_PATCH
    augmenter = imageDataAugmenter('RandRotation',[-10 10], ...
                                   'RandXReflection', true, ...
                                   'RandYReflection', true);
    pt = randomPatchExtractionDatastore(imdsTrainX, imdsTrainY, PATCH_SIZE, ...
         'PatchesPerImage', PATCHES_PER_IM, 'DataAugmentation', augmenter);

    pv = randomPatchExtractionDatastore(imdsValX, imdsValY, PATCH_SIZE, ...
         'PatchesPerImage', PATCHES_PER_IM);

    dsTrainPrep = transform(pt, @(t) augmentInputOnlyPatchTable(t, APPLY_INPUT_ARTIFACTS));
    dsValPrep   = transform(pv, @identityPatchTableNormalized);
end

% Sanity
reset(dsTrainPrep); tr = read(dsTrainPrep);
reset(dsValPrep);   va = read(dsValPrep);
[Xtr,Ytr] = getXY_any(tr, false);
[Xva,Yva] = getXY_any(va, false);
fprintf("Train X:[%.4f %.4f] Y:[%.4f %.4f]\n", min(Xtr(:)),max(Xtr(:)),min(Ytr(:)),max(Ytr(:)));
fprintf("Val   X:[%.4f %.4f] Y:[%.4f %.4f]\n", min(Xva(:)),max(Xva(:)),min(Yva(:)),max(Yva(:)));
fprintf("Raw RMSE: train=%.4f | val=%.4f\n\n", ...
        sqrt(mean((Xtr(:)-Ytr(:)).^2)), sqrt(mean((Xva(:)-Yva(:)).^2)));

%% -------------------------- Build U-Net --------------------------------
inputSize     = [inputSize(1) inputSize(2) 1];
BASE_CHANNELS = opt.BaseChannels;
ENCODER_DEPTH = opt.EncoderDepth;

lgraph = buildUNet(inputSize, BASE_CHANNELS, ENCODER_DEPTH);

%% ------------------------ Training Options -----------------------------
options = trainingOptions('adam', ...
    'InitialLearnRate',           3e-4, ...
    'LearnRateSchedule',          'piecewise', ...
    'LearnRateDropPeriod',        20, ...
    'LearnRateDropFactor',        0.7, ...
    'MiniBatchSize',              8, ...
    'MaxEpochs',                  60, ...
    'Shuffle',                    'every-epoch', ...
    'ValidationData',             dsValPrep, ...
    'ValidationPatience',         8, ...
    'L2Regularization',           1e-5, ...
    'GradientThresholdMethod',    'l2norm', ...
    'GradientThreshold',          5, ...
    'ExecutionEnvironment',       char(opt.ExecEnv), ...
    'Plots',                      'training-progress');

%% ------------------------------ Train ----------------------------------
net = trainNetwork(dsTrainPrep, lgraph, options);
save(MODEL_FILE, "net");

%% --------------------------- Evaluate Val ------------------------------
reset(dsValPrep)
ps = []; ss = []; n = 0;
while hasdata(dsValPrep)
    b = read(dsValPrep);
    [Xin,Yin] = getXY_any(b, false);
    Yhat = predict(net, Xin);
    ps(end+1) = psnr(squeeze(Yhat), squeeze(Yin)); %#ok<AGROW>
    ss(end+1) = ssim(squeeze(Yhat), squeeze(Yin)); %#ok<AGROW>
    n = n + 1;
end
fprintf("Validation: PSNR=%.2f dB | SSIM=%.4f (N=%d)\n\n", mean(ps), mean(ss), n);

%% --------------------------- Full Inference ----------------------------
USE_PER_SLICE_NORMALIZATION = true;
LOW_PERCENTILE  = 1;
HIGH_PERCENTILE = 99;

yPredVol = zeros([H0 W0 Z], 'single');
tile     = inputSize(1:2);
overlap  = round(0.25 * tile);

for k = 1:Z
    slice = xFDK_raw(:,:,k);

    if USE_PER_SLICE_NORMALIZATION
        sliceN = im2single(toUint16PerSliceSafe(slice, LOW_PERCENTILE, HIGH_PERCENTILE));
    else
        v = xFDK_raw(:);
        lo = prctile(v, LOW_PERCENTILE);
        hi = prctile(v, HIGH_PERCENTILE);
        sliceN = (slice - lo) ./ (hi - lo + eps);
        sliceN = min(max(sliceN, 0), 1);
    end

    if DO_PATCH
        yPredVol(:,:,k) = tiledPredict2D(net, sliceN, tile, overlap);
    else
        [Xpad, pSpec] = padToTarget(sliceN, TARGET_HW);
        Ypad = tiledPredict2D(net, Xpad, tile, overlap);
        yPredVol(:,:,k) = unpadFromSpec(Ypad, pSpec, [H0 W0]);
    end
end

save(RECON_FILE,'yPredVol','-v7.3');
fprintf("Prediction complete ✔  Saved to: %s\n", RECON_FILE);

%% ----------------------------- Info out --------------------------------
info = struct();
info.OutRoot     = OUT_ROOT;
info.ModelFile   = MODEL_FILE;
info.ReconFile   = RECON_FILE;
info.TrainSelect = find(isTrain);
info.ValSelect   = find(isVal);

end  % ===== END MAIN FUNCTION =====

% ========================================================================
%                            Helper Functions
% ========================================================================

function V = resolveVolumeInput(in, varName, tag)
    if isnumeric(in)
        V = in;
        assert(ndims(V)==3, '%s volume must be 3D.', tag);
        return;
    end
    assert(ischar(in)||isstring(in), '%s must be a 3D array or a MAT filepath.', tag);
    S = load(char(in));
    if ~isempty(varName)
        assert(isfield(S, char(varName)), 'Variable %s not found in %s', char(varName), char(in));
        V = S.(char(varName));
        % If it's a struct wrapping same-named inner field (e.g., S.FDKvol.FDKvol)
        if isstruct(V) && isfield(V, char(varName))
            V = V.(char(varName));
        end
        assert(ndims(V)==3, 'Selected variable %s must be 3D.', char(varName));
        return;
    end
    % Auto-detect reasonable candidates
    cand = fieldnames(S);
    % Prefer common names
    pref = {'FDKvol','hybridIR_vol','hybridIR','yPredVol','recon_volume'};
    for i=1:numel(pref)
        if isfield(S, pref{i})
            tmp = S.(pref{i});
            if isstruct(tmp) && isfield(tmp, pref{i}), tmp = tmp.(pref{i}); end
            if isnumeric(tmp) && ndims(tmp)==3, V = tmp; return; end
        end
    end
    % Fall back: first 3D numeric
    for i=1:numel(cand)
        tmp = S.(cand{i});
        if isnumeric(tmp) && ndims(tmp)==3, V = tmp; return; end
        if isstruct(tmp)
            fn = fieldnames(tmp);
            for j=1:numel(fn)
                val = tmp.(fn{j});
                if isnumeric(val) && ndims(val)==3, V = val; return; end
            end
        end
    end
    error('Could not auto-detect a 3D volume from %s. Provide FDKVar/TGTVar.', char(in));
end

function out = ternary(cond, a, b), if cond, out=a; else, out=b; end, end

function [Xp,s] = padToTarget(X,hw)
    Ht=hw(1); Wt=hw(2);
    h=size(X,1); w=size(X,2);
    a=floor((Ht-h)/2); b=ceil((Ht-h)/2);
    c=floor((Wt-w)/2); d=ceil((Wt-w)/2);
    Xp=padarray(X,[a c],'replicate','pre');
    Xp=padarray(Xp,[b d],'replicate','post');
    s=struct('top',a,'bottom',b,'left',c,'right',d);
end

function X = unpadFromSpec(Xp,s,sz)
    X = Xp(1+s.top:end-s.bottom, 1+s.left:end-s.right);
    if any(size(X)~=sz), X=imresize(X,sz,'nearest'); end
end

function u16 = toUint16PerSliceSafe(X, pLo, pHi)
    X = single(X);
    p1 = prctile(X(:), pLo);
    p2 = prctile(X(:), pHi);
    if p2 > p1
        scaled = (X - p1) / (p2 - p1);
    else
        vmin = min(X(:)); vmax = max(X(:));
        if vmax > vmin, scaled = (X - vmin) / (vmax - vmin); else, scaled = zeros(size(X),'single'); end
    end
    scaled = min(max(scaled, 0), 1);
    u16 = im2uint16(scaled);
end

function dataOut = preprocessBatch_train(b, applyArtifacts)
    [X, Y] = getXY_cell(b, false);
    if ndims(X)>2, X=X(:,:,1); end
    if ndims(Y)>2, Y=Y(:,:,1); end
    if rand<0.5, X=fliplr(X); Y=fliplr(Y); end
    if rand<0.5, X=flipud(X); Y=flipud(Y); end
    k=randi([0 3]); X=rot90(X,k); Y=rot90(Y,k);
    if applyArtifacts, X=addArtifacts_safe(X); end
    X=min(max(X,0),1); Y=min(max(Y,0),1);
    X=reshape(X,size(X,1),size(X,2),1); Y=reshape(Y,size(Y,1),size(Y,2),1);
    dataOut={X,Y};
end

function dataOut = preprocessBatch_val(b)
    [X, Y] = getXY_cell(b, false);
    if ndims(X)>2, X=X(:,:,1); end
    if ndims(Y)>2, Y=Y(:,:,1); end
    X=reshape(X,size(X,1),size(X,2),1); Y=reshape(Y,size(Y,1),size(Y,2),1);
    dataOut={X,Y};
end

function [X,Y]=getXY_cell(b, doMat2Gray)
    X = im2single(b{1}); Y = im2single(b{2});
    if doMat2Gray, X=mat2gray(X); Y=mat2gray(Y); end
end

function [X,Y] = getXY_table(t, doMat2Gray)
    vn=t.Properties.VariableNames;
    f1=pickName(vn,{'inputImage','InputImage'});
    f2=pickName(vn,{'responseImage','ResponseImage'});
    X=im2single(t.(f1){1}); Y=im2single(t.(f2){1});
    if doMat2Gray, X=mat2gray(X); Y=mat2gray(Y); end
end

function [X,Y] = getXY_any(b, doMat2Gray)
    if nargin<2, doMat2Gray=false; end
    if iscell(b), [X,Y]=getXY_cell(b, doMat2Gray);
    else,         [X,Y]=getXY_table(b, doMat2Gray);
    end
    if ndims(X)>2, X=X(:,:,1); end
    if ndims(Y)>2, Y=Y(:,:,1); end
    X=reshape(X,size(X,1),size(X,2),1); Y=reshape(Y,size(Y,1),size(Y,2),1);
end

function n = pickName(names,cands)
    for i=1:numel(cands), if any(strcmp(names,cands{i})), n=cands{i}; return; end, end
    error("Expected variable not found.");
end

function dataOut = augmentInputOnlyPatchTable(t,apply)
    [X,Y]=getXY_table(t, false);
    if apply, X=addArtifacts_safe(X); end
    dataOut=table({X},{Y},'VariableNames',{'inputImage','responseImage'});
end

function dataOut = identityPatchTableNormalized(t)
    [X,Y]=getXY_table(t, false);
    dataOut=table({X},{Y},'VariableNames',{'inputImage','responseImage'});
end

function X = addArtifacts_safe(X)
    if rand < 0.4, sigma = 0.002 * (0.5 + rand); X = X + sigma*randn(size(X),'like',X); end
    if rand < 0.3, len = randi([3 5]); ang = randi([0 180]); X = imfilter(X, fspecial('motion', len, ang), 'replicate'); end
    if rand < 0.2, [~,w] = size(X); band = 1 + 0.02*((sin(linspace(0, 4*pi, w))+1)/2 - 0.5); X = X .* reshape(band, 1, w); end
    if rand < 0.3, alpha = 0.95 + 0.1*rand; beta  = -0.02 + 0.04*rand; X = alpha*X + beta; end
    X = min(max(X,0),1);
end

function Y = tiledPredict2D(net,X,tile,ov)
    X=im2single(X);
    [H,W]=size(X); th=tile(1); tw=tile(2); oh=ov(1); ow=ov(2);
    sh=max(th-oh,1); sw=max(tw-ow,1);
    if exist('hann','file'), wh=hann(th); ww=hann(tw);
    else, wh=0.5*(1-cos(2*pi*(0:(th-1))'/(th-1))); ww=0.5*(1-cos(2*pi*(0:(tw-1))'/(tw-1)));
    end
    w2=single(wh*ww.');
    Ysum=zeros(H,W,'single'); Wsum=zeros(H,W,'single');
    for top=1:sh:H
        for left=1:sw:W
            bottom=min(top+th-1,H); right=min(left+tw-1,W);
            topA=max(1,bottom-th+1); leftA=max(1,right-tw+1);
            patch=X(topA:topA+th-1, leftA:leftA+tw-1);
            Yp=predict(net,reshape(patch,[th tw 1]));
            Yp=squeeze(Yp);
            Ysum(topA:topA+th-1,leftA:leftA+tw-1)=Ysum(topA:topA+th-1,leftA:leftA+tw-1)+Yp.*w2;
            Wsum(topA:topA+th-1,leftA:leftA+tw-1)=Wsum(topA:topA+th-1,leftA:leftA+tw-1)+w2;
        end
    end
    Y=Ysum./max(Wsum,eps('single'));
end

function showTriplet(X,Yhat,Y,t1,t2,t3)
    Xs=squeeze(X); Ys=squeeze(Yhat); Gs=squeeze(Y);
    ps=psnr(Ys,Gs); ss=ssim(Ys,Gs);
    figure; tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
    nexttile; imshow(Xs,[]); title(t1);
    nexttile; imshow(Ys,[]); title(sprintf("%s (PSNR %.2f, SSIM %.3f)",t2,ps,ss));
    nexttile; imshow(Gs,[]); title(t3);
end

function lgraph = buildUNet(inputSize, baseFilters, depth)
    layers = imageInputLayer(inputSize, 'Normalization','none', 'Name','input');
    encNames = cell(1,depth); numF = baseFilters;
    for d = 1:depth
        blk = [
            convolution2dLayer(3, numF, 'Padding','same', 'Name',sprintf('enc%d_c1',d))
            batchNormalizationLayer('Name',sprintf('enc%d_b1',d))
            reluLayer('Name',sprintf('enc%d_r1',d))
            convolution2dLayer(3, numF, 'Padding','same', 'Name',sprintf('enc%d_c2',d))
            batchNormalizationLayer('Name',sprintf('enc%d_b2',d))
            reluLayer('Name',sprintf('enc%d_r2',d))
        ];
        layers = [layers; blk]; %#ok<AGROW>
        encNames{d} = sprintf('enc%d_r2', d);
        if d < depth, layers = [layers; maxPooling2dLayer(2,'Stride',2,'Name',sprintf('pool%d',d))]; end %#ok<AGROW>
        numF = numF*2;
    end
    bottleneck = [
        convolution2dLayer(3, numF, 'Padding','same', 'Name','b_c1')
        batchNormalizationLayer('Name','b_b1')
        reluLayer('Name','b_r1')
        convolution2dLayer(3, numF, 'Padding','same', 'Name','b_c2')
        batchNormalizationLayer('Name','b_b2')
        reluLayer('Name','b_r2')
    ];
    lg = layerGraph([layers; bottleneck]);
    prev = 'b_r2';
    for d = depth-1:-1:1
        numF = baseFilters * 2^(d-1);
        upName = sprintf('up%d', d); catName = sprintf('cat%d', d);
        lg = addLayers(lg, transposedConv2dLayer(2, numF, 'Stride',2, 'Cropping','same', 'Name',upName));
        lg = addLayers(lg, concatenationLayer(3,2, 'Name',catName));
        decBlk = [
            convolution2dLayer(3, numF, 'Padding','same', 'Name',sprintf('dec%d_c1',d))
            batchNormalizationLayer('Name',sprintf('dec%d_b1',d))
            reluLayer('Name',sprintf('dec%d_r1',d))
            convolution2dLayer(3, numF, 'Padding','same', 'Name',sprintf('dec%d_c2',d))
            batchNormalizationLayer('Name',sprintf('dec%d_b2',d))
            reluLayer('Name',sprintf('dec%d_r2',d))
        ];
        lg = addLayers(lg, decBlk);
        lg = connectLayers(lg, prev, upName);
        lg = connectLayers(lg, upName, [catName '/in1']);
        lg = connectLayers(lg, encNames{d}, [catName '/in2']);
        lg = connectLayers(lg, catName, sprintf('dec%d_c1',d));
        prev = sprintf('dec%d_r2', d);
    end
    outHead = [convolution2dLayer(1, 1, 'Padding','same', 'Name','final_c') regressionLayer('Name','out')];
    lg = addLayers(lg, outHead); lg = connectLayers(lg, prev, 'final_c');
    lgraph = lg;
end