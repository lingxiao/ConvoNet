% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------
% Module: Utilities for processing surgical images
% Author: Xiao Ling
% Date  : May 9th, 2014
% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------


function T = Utils()

	Pr = Prelude();
	T.loadAll   = @(b)       loadAll(b);
	T.loadBw    = @(b)       loadBw(b);
	T.crop 		= @(m)       crop(m);
	T.toPatch   = @(a,b)     toPatch(Pr,a,b);
	T.cutPatch  = @(a,b,c,d) cutPatch(a,b,c,d);
	T.toMatrix  = @(m) 		 toMatrix(m);

end

% load all images in blackand white
function X = loadBw(impath)

	X  = [];
	ms = loadAll(impath);
	for k=1:length(ms)
		X(:,:,k) = double(rgb2gray(ms{k}));
	end
end

% Load all images, crop and normalize for intensity of light
% loadAll :: String -> (Mat m n 3)
function ms = loadAll(impath)

	dcell  = dir(impath);
	ms     = {};

	for d = 4:length(dcell)
		ms{d-3} = crop(imread([impath dcell(d).name]));
	end

end

% crop the lacroscope circle out of the image
% crop :: Mat m n 3 -> Mat (m' < m) (n' < n) 3
function m2 = crop(m1)

	m2 = m1(100:510,100:670,:);

end

% Extract some bounding box from set of image `ims` given coordinates `cs`
% Where each c <- cs maps to each im <- ims
% Output patches and patches normalized in size
function T = toPatch(Pr,ims,cs)
	
	Large  = {};
	maxrow = 10^10;
	maxcol = 10^10;

	for k = 1:length(cs)

		im   = ims{k};
		c    = cs{k};

		[m,n,~] = size(im);

		if length(c) > 0
			rmin 	   = min(min(c(:,2)),m);
			rmax       = min(max(c(:,2)),m);
			cmin 	   = min(min(c(:,1)),n);
			cmax	   = min(max(c(:,1)),n);
			lp   	   = im(rmin:rmax,cmin:cmax,:);
			Large{end+1}   = lp;

			% update maxrow and max col
			if (rmax - rmin) < maxrow
				maxrow = rmax - rmin;
			end
			if (cmax - cmin) < maxcol
				maxcol = cmax - cmin;
			end

		end

	end 

	Small   = {};
	T.large = Large;
	T.small = Pr.joinCell(Pr.mapCell(@(m) cutPatch(m,{},maxrow,maxcol), Large));

end

% Cut a large patch `im` into smaller patches `ps` of dim `maxrow` x `maxcol`
function T = cutPatch(im,ps,maxrow,maxcol)
	
	[m,n,~] = size(im);
	if or(m < maxrow,n < maxcol)
		T = ps;
	else 
		ims 	  = split(im,maxrow,maxcol);
		ps{end+1} = ims.fst;
		T      	  = cutPatch(ims.snd,ps,maxrow,maxcol);
	end

	function T = split(im1,maxrow,maxcol)

		[m,n,~] = size(im1);

		if m >= n  		% vertical image
			im2 = im1(maxrow:end,:,:);
		else 			% horizontal image
			im2 = im1(:,maxcol:end,:);
		end

		T.fst = im1(1:maxrow,1:maxcol,:);
		T.snd = im2;

	end
		
end

% toMatrix : {m} -> [m]
function M = toMatrix(ms)

	M = [];

	if size(ms{1},3) > 1
		M = [];
	else
		for k = 1:length(ms)
			M(:,:,k) = ms{k};
		end
	end

end














