% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------
% Module: Utils for Fast sparse coding algorithms
% Author: Xiao Ling
% Date  : May 8th, 2014
% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------

% Export Module Functions
function T = SparseEncodeUtils()

    % load core alogrithm
	Core          = SparseEncodeCore();

    % re-export config function from core
    T.config      = @(t)       Core.config(t);            

    % export utility functions
    T.toPatch     = @(a,b)     toPatch     (a,b);
    T.train       = @(a,b)     train       (Core,a,b);
    T.test        = @(a,b,c)   test        (Core,a,b,c);
    T.decode      = @(a,b,c)   decode      (Core,a,b,c);
	T.showBasis   = @(a,b)     showBasis   (a,b);
    T.showWeights = @(a,b,c)   showWeights (a,b,c);
    T.byResponse  = @(a,b,c)   byResponse  (Core,a,b,c);

    % temporary export
    T.withWin     = @(a,b,c)   withWin     (a,b,c);

end

% -----------------------------------------------------------------------------------
% Train Algorithm := toPatch >>= sparseEncode
% -----------------------------------------------------------------------------------

% train :: CONFIG -> Mat m n p -> (Mat m k, Mat r s)
function T = train(Core,CONFIG,M)

    T.config      = CONFIG;
    [X,M]         = toPatch(CONFIG,M);
    T.data.vector = X;
    T.data.patch  = M;
    T.train       = Core.learn(CONFIG,X);

end

% -----------------------------------------------------------------------------------
% Test Algorithm
% -----------------------------------------------------------------------------------

% test img bs = toPatch img >=\(a,_) -> toWeights a bs
function T = test(Core,CONFIG,Img,Bs)

	[X,M]    = toPatch(CONFIG,Img);
    tws      = Core.encode(CONFIG,X,Bs);

    T.Patches = M;
    T.Weights = tws.Weights;
    T.Stat    = tws.Stat;

end

% -----------------------------------------------------------------------------------
% Reconstruct image
% -----------------------------------------------------------------------------------

% Reconstruct image `Im1` with basis `Bs` acoording to configuration `CONFIG` used to
% learn basis Bs
function Im2 = decode(Core,CONFIG,Bs,Im1)

    patchSize = CONFIG.Main.patchSize;
    Im2       = withWin(patchSize,Im1,@(v) mk(Core,CONFIG,patchSize,v,Bs));

    function p = mk(Core,CONFIG,patchSize,Xb,Bs)

        ws = Core.encode(CONFIG,Xb,Bs);
        p  = reshape(Bs*ws.Weights,[patchSize,patchSize,1]);

    end

end

% -----------------------------------------------------------------------------------
% Segment image
% -----------------------------------------------------------------------------------

% Segment 'im1' by reconstructiong each patch (patch size defined in `CONFIG`) 
% with basis from set of classes `Cs`, select reconstruction by the max response 
% of all basis
% byResponse :: Core -> Config -> {Basis} -> Image -> Image
function im2 = byResponse(Core,CONFIG,Cs,im1)

    K   = size(Cs);
    sz  = CONFIG.Main.patchSize;
    im2 = withWin(sz,im1,@(v) segment(Core,CONFIG,v,Cs,K));

    
    function Ret = segment(Core,CONFIG,Xb,Cs,K)

        % Store weights
        % Update tuple of (class, value) for max response and minimum residual
        ws      = [];
        maxResp = [0,0];
        minRes  = [0,10^10];

        for k = 1:K

            Bs = Cs{k};
            t  = Core.encode(CONFIG,Xb,Bs);
            mr       = max(abs(t.Weights));

            % update current leader in max response and minimum residual
            if maxResp(2) < mr;
                maxResp = [k,mr];
            end

            if minRes(2) > t.Stat.fresidue_total
                minRes  = [k,t.Stat.fresidue_total];
            end

            % store class weights
            ws(:,k) = t.Weights;

        end

        % decide class based on set criteria below
        % Ret.ws      = ws;
        % Ret.maxResp = maxResp;
        % Ret.minRes  = minRes;
        Ret = minRes(1)*ones(size(Xb));

    end

end


% -----------------------------------------------------------------------------------
% Module Utils
% -----------------------------------------------------------------------------------

% Runs a square sliding window of size `patchSize` over `Img` and apply function `g` 
% onto each patch. 
% Note function g takes in vector where vector = reshape(patch,[patchSize^2,1])
% Note if `Img` size is not evenly divisble by `patchSize`, then the remainder
% pixels will be truncated and a resized `Img2` will be output
function Img2 = withWin(patchSize,Img,g)

    [m,n,~] = size(Img);
    Img2    =  [];
    vecsz   = [patchSize^2,1];

    for r = 1:m/patchSize
        rmin = r*patchSize - (patchSize - 1);
        rmax = r*patchSize;
        for c = 1:n/patchSize
          cmin    = c*patchSize - (patchSize - 1);
          cmax    = c*patchSize;

          % get patch and reshape into size*size vector
          % apply function g onto vector
          patch   = Img(rmin:rmax,cmin:cmax);
          pavec   = g(reshape(patch,vecsz)) ;

          % reconstruct new image
          Img2(rmin:rmax,cmin:cmax) = reshape(pavec,[patchSize,patchSize]);

        end
    end

end


% -----------------------------------------------------------------------------------
% Generate Image Patches for training
% -----------------------------------------------------------------------------------

function [X,Y] = toPatch(CONFIG,Images)

    [m,n,k] = size(Images);
    sz      = CONFIG.Main.patchSize;
    num     = CONFIG.Main.numPatches;
    BUFF    = 4;

    totalsamples = 1;

    % extract subimages at random from this image to make data vector X
    X = zeros(sz^2, num);
    for i=1:k,

        img       = Images(:,:,i);
        % Determine how many patches to take
        getsample = floor(num/k);

        if i == k, getsample = num-totalsamples; end

        % Extract patches at random from this image to make data vector X
        for j=1:getsample
            r                   = BUFF+ceil((m-sz-2*BUFF)*rand);
            c                   = BUFF+ceil((n-sz-2*BUFF)*rand);

            im                  = img(r:r+sz-1,c:c+sz-1);
            temp                = reshape(im,sz^2,1);
            X(:,totalsamples)   = temp - mean(temp);
            Y(:,:,totalsamples) = im;
            
            totalsamples        = totalsamples + 1;

        end
    end  
end

% -----------------------------------------------------------------------------------
% Plot basis and weights
% -----------------------------------------------------------------------------------

% From HongLak Lee
function h = showBasis(A, str, numcols, figstart)

    %  display_network -- displays the state of the network
    %  A = basis function matrix

    figure;

    if exist('figstart', 'var') && ~isempty(figstart), figure(figstart); end

    [L M]=size(A);
    if ~exist('numcols', 'var')
        numcols = ceil(sqrt(L));
        while mod(L, numcols), numcols= numcols+1; end
    end
    ysz = numcols;
    xsz = ceil(L/ysz);

    m=floor(sqrt(M*ysz/xsz));
    n=ceil(M/m);

    colormap(gray)

    buf=1;
    array=-ones(buf+m*(xsz+buf),buf+n*(ysz+buf));

    k=1;
    for i=1:m
        for j=1:n
            if k>M continue; end
            clim=max(abs(A(:,k)));
            array(buf+(i-1)*(xsz+buf)+[1:xsz],buf+(j-1)*(ysz+buf)+[1:ysz])=...
                reshape(A(:,k),xsz,ysz)/clim;
            k=k+1;
        end
    end

    if isreal(array)
        h=imagesc(array,'EraseMode','none',[-1 1]);
    else
        h=imagesc(20*log10(abs(array)),'EraseMode','none',[-1 1]);
    end;
    axis image off

    drawnow

    title(strcat('Basis', '-', str));
end

function [] = showWeights(ws,n,str)

    [o,p] = size(ws);

    if or(o == 1,p == 1)
        figure; plot(ws,'o','color','b'); title(str);
    else
        is = randperm(p);
        m  = min(length(ws),n);
        js = is(1:m);
        figure; plot(ws(:,js),'o','color','b'); title(str);
    end
end
