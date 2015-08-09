
% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------
% Module: Convolutional Restricted Boltzman Machine
%       
%        The detail of the algorithm is described in the following paper:
%        'Convolutional Deep Belief Networks for Scalable Unsupervised Learning of 
%         Hiearchical Representations'
% 
% Author: Xiao Ling 
%  		  * Based on code written by Honglak Lee
% Date  : July 30th, 2014

% Bash   : /Applications/MATLAB_R2014b.app/bin/matlab -nodesktop -nosplash
% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------

% Export Module Functions
function T = ConvBoltzman()

	T.train      = @(a,b)   trainRBM  (a,b);
	T.trainWith  = @(a,b,c) trainRBM2 (a,b,c);
	T.run        = @(a,b)   runRBM    (a,b);
	T.runAll     = @(a,b)   runRBM2   (a,b);

	T.down       = @(m,h)   toVisible(m.Config,m.W,h    );

end

% -----------------------------------------------------------------------------------
% Inference
% -----------------------------------------------------------------------------------

% runRBM : RBM x Image -> Hidden
function H    = runRBM(rbm,V)

	H1        = toHidden (rbm.Config,rbm.W,rbm.B,V);
	[P1,~]    = pool     (rbm.Config,H1 		  );
	H         = P1 								   ;

end

% runRBM2 : RBM x [Image] -> [Hidden]
function Hs   = runRBM2(rbm,Vs)

	Hs    = {};
	for m = 1:length(Vs)
		Hs{end+1} = runRBM(rbm,Vs{m});
		logRun(); 									% : IO ()
	end

end

% -----------------------------------------------------------------------------------
% Learn
% -----------------------------------------------------------------------------------

% Train a random RBM using data `xs` subject to configuration `CON`
% trainRBM : CONFIG x [Image] -> (RBM,Config,[ℝ1],[ℝ1])
function O  = trainRBM(CON,xs)

	% Random initialization of weights and biases
	W    = 0.01*randn(CON.nv1 * CON.nv2, size(xs{1},3), CON.K);
	B    = -0.1*ones (CON.K,1);
	C    = zeros     (CON.K,1);

	O    = trainRBM2(CON,xs,newRBM(W,B,C));

end

% Continue training with existing `rbm0`
% trainRBM2 :: CONFIG x [Image] x RBM -> (RBM,Config,[ℝ1],[ℝ1])
function O = trainRBM2(CON,xs,rbm0)

	logStart();													% : IO ()
	
	% Initialize with initial `rbm0`
	rbm         = newRBM(rbm0.W,rbm0.B,rbm0.C); 				% : State RBM RBM
	ferr        = []; 											% : State [ℝ1] [ℝ1]
	sparsityErr = []; 											% : State [ℝ1] [ℝ1]

	for t = 1:CON.numTrial
		
		% Main training subroutine
		step        = train(CON,getMomemtum(t),toPatch(CON,xs),rbm); 	

		rbm 	       = step.rbm; 								% : State RBM RBM
		ferr(t)        = mean(step.ferr); 						% : State [ℝ1] [ℝ1]
		sparsityErr(t) = mean(step.sparsityErr); 				% : State [ℝ1] [ℝ1]

		% decay the std_gaussian term 
		if (CON.std_gaussian > CON.sigma_stop)
			CON.std_gaussian = CON.std_gaussian * 0.99;
		end

		% Update progress
		logMsg(t) 												% : IO ()
	end

	% Output results
	O.Rbm.W       = rbm.W;
	O.Rbm.B       = rbm.B;
	O.Rbm.C       = rbm.C;
	O.Config      = CON;
	O.Ferr        = ferr;
	O.SparsityErr = sparsityErr;

end

% train : CONFIG x ℝ1 x [Patch] x RBM -> (RBM,[ℝ1],[ℝ1])
function O = train(CON,m,ps,rbm)

	rbm1  		= rbm;					% : State RBM RBM
	ferr        = []; 					% : State [ℝ1] [ℝ1]
	sparsityErr = []; 					% : State [ℝ1] [ℝ1]

	for n = 1:length(ps)

		% contrastive divergence for current patch
		step     = contraDiv(CON,rbm1,ps{n});

		% update boltzman machine parameters
        rbm1.dW  = m*rbm1.dW + CON.epsilon*step.dW;
		rbm1.dB  = m*rbm1.dB + CON.epsilon*step.dB;
		rbm1.dC  = m*rbm1.dC + CON.epsilon*step.dC;

		rbm1.W   = rbm1.W    + rbm1.dW;
		rbm1.B   = rbm1.B    + rbm1.dB;
		rbm1.C   = rbm1.C    + rbm1.dC;

		% update traning error
		ferr(n)        = step.ferr;
		sparsityErr(n) = step.sparsityErr;

	end

	% output results
	O.rbm   	  = rbm1;
	O.ferr        = ferr;
	O.sparsityErr = sparsityErr;

end

% -----------------------------------------------------------------------------------
% Contrastive Divergence 
% -----------------------------------------------------------------------------------

% contraDiv : CONFIG x RBM x Patch -> (dW,dB,dC,[ℝ1],[ℝ1])
function O     = contraDiv(CON,rbm,H0)

	%%%%%%%%%%%%%%%%%%% Gibbs Sampling %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	W01 		 = rbm.W;
	B0  		 = rbm.B;

	% Positive phase
	H1         	 = toHidden  (CON,W01,B0,H0   );
	[P1,H1n]     = pool      (CON,H1 		 );

	% Negative phase
	P1_   		 = P1; 				% State P1 P1

	for c = 1:CON.cdk
		H0_ 	     = toVisible (CON,W01,P1      );
		H1_          = toHidden  (CON,W01,B0,H0_  );
		[P1_,H1n_]   = pool      (CON,H1_         );
	end


	%%%%%%%%%%%%%%%%% Update  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	poshidact    = squeeze(sum(sum(H1n )));
	neghidact    = squeeze(sum(sum(H1n_)));
	hv  	     = toHV   (CON,H0 ,H1n );
	hv_ 	     = toHV   (CON,H0_,H1n_);

	if strcmp(CON.bias_mode, 'none')
	    dhbias = 0;
	elseif strcmp(CON.bias_mode, 'simple')
	    dhbias = squeeze(mean(mean(H1n,1),2)) - CON.pbias;
	end

	numCase      = size(H1n,1)*size(H1n,2);
	dW_total3    = - CON.pbias_lambda*0; 

	% output deltas
	O.dW         = (hv-hv_)/numCase - CON.l2reg*rbm.W + dW_total3;
	O.dB         = (poshidact-neghidact)/numCase - CON.pbias_lambda*dhbias;
	O.dC         = 0; 

	% output error
	O.ferr  	   = mean((H0(:)-H0_(:)).^2);
	O.sparsityErr  = mean(H1n(:));

end

% toHidden : Config x W x B x Image -> Hidden
function H1    = toHidden(CON,W,B,H0)

	Wr 		     = flipW(CON,W);

	% send each visible unit `H0` to set of hidden units `H1`, sum all `H1`
	H1           = convWeight(H0(:,:,1),Wr(:,:,1,:),'valid');

	for c = 2:size(H0,3)
		H1 = H1 + convWeight(H0(:,:,c),Wr(:,:,c,:),'valid');
	end

	% transform by bias and standard gaussian
	H1           = H1 + repmat(reshape(B,[1,1,length(B)]),[size(H1,1),size(H1,2),1]);
	H1           = 1/(CON.std_gaussian^2) .* H1;

	% reshape and flip weights
	function Wr = flipW(CON,W)
		[~,C,K] = size(W);
		Wr      = reshape(W(end:-1:1,:,:), [CON.nv1,CON.nv2,C,K]);
	end

	
	function H  = convWeight(p,W,opt)

		H  = [];
		for k = 1:size(W,4)
			H(:,:,k) = conv2(p,W(:,:,:,k),opt);
		end

	end

end

% toVisible : Config x Weights x Hidden -> Visible
function H0_   = toVisible(CON,W,H1)

    W_      = reshape(W,[CON.nv1,CON.nv2,size(W,2),size(W,3)]);

    for c = 1:size(W,2)
        H0_(:,:,c) = convHW(H1,W_(:,:,c,:));
    end

    function hw = convHW(H1,W)

        for k = 1:size(H1,3)
            hw(:,:,k) = conv2(H1(:,:,k),W(:,:,:,k),'full');
        end
        hw = sum(hw,3);
    end

end

function HV    = toHV(CON,H0,H1)

	% reverse h1
	H1nr     = H1(size(H1,1):-1:1,size(H1,2):-1:1,:   );
	HV 		 = zeros(CON.nv1,CON.nv2,size(H0,3),size(H1,3));

	for c = 1:size(H0,3)
		for k = 1:size(H1nr,3)
			HV(:,:,c,k) = conv2(H0(:,:,c),H1nr(:,:,k),'valid');
		end
	end

	HV       = reshape(HV,[CON.nv1 * CON.nv2,size(H0,3),size(H1,3)]);

end

% pool : Config x Hidden -> (Hidden,Hidden)
function [maxR normR]    = pool(CON,H1)

	H1      = exp(H1);

	[r,c,l] = size(H1);
	dr      = mod(r,CON.spacing);
	dc      = mod(c,CON.spacing);
	H1a     = H1(1:end-dr,1:end-dc,:);

	H1_mult = zeros(CON.spacing^2+1,size(H1a,1)*size(H1a,2)*size(H1a,3)/CON.spacing^2);
	H1_mult(end,:) = 1;

	% extract patches
	for m = 1:CON.spacing
		for n = 1:CON.spacing
			v = H1a(n:CON.spacing:end,m:CON.spacing:end,:);
			H1_mult((m-1)*CON.spacing+n,:) = v(:);
		end
	end

	% normalize patches
	[S P]    = multrand(H1_mult');

	% project pooled values back into orignal patch sizes
	maxR     = zeros(size(H1a));
	normR    = zeros(size(H1a));

	for m=1:CON.spacing
		for n=1:CON.spacing
		    maxR (m:CON.spacing:end, n:CON.spacing:end, :) = reshape(S((n-1)*CON.spacing+m,:), [size(maxR,1)/CON.spacing, size(maxR,2)/CON.spacing, size(maxR,3)]);
	    	normR(m:CON.spacing:end, n:CON.spacing:end, :) = reshape(P((n-1)*CON.spacing+m,:), [size(maxR,1)/CON.spacing, size(maxR,2)/CON.spacing, size(maxR,3)]);
		end
	end

	if dr ~= 0
		H1r     = zeros  (dr,size(maxR,2),l);
		maxR    = vertcat(maxR ,H1r);
		normR   = vertcat(normR,H1r);
	end
		
	if dc ~= 0
		H1c     = zeros  (size(maxR,1),dc,l);
		maxR    = horzcat(maxR ,H1c);
		normR   = horzcat(normR,H1c);
	end

	function [S P] = multrand(P)

		% normalized response
		sumP 	   = sum(P,2);
		P    	   = P./repmat(sumP, [1,size(P,2)]);

		cumP       = cumsum(P,2);
		unifrnd    = rand(size(P,1),1);
		temp       = cumP > repmat(unifrnd,[1,size(P,2)]);
		Sindx      = diff(temp,1,2);
		S 		   = zeros(size(P));
		S(:,1)     = 1-sum(Sindx,2);
		S(:,2:end) = Sindx;
		S  		   = S';
		P  		   = P';
	end

end

% -----------------------------------------------------------------------------------
% Utils
% -----------------------------------------------------------------------------------

% newRBM : W x B x C -> RBM
function rbm = newRBM(W,B,C)

	rbm.W   = W;
	rbm.B   = B;
	rbm.C   = C;
	rbm.dW  = 0;
	rbm.dB  = 0;
	rbm.dC  = 0;

end


% getMomemtum : Int -> ℝ1
function m   = getMomemtum(t)

	if t<5
	    m = 0.5;
	else
	    m = 0.9;
	end
end

% Note if image dimension is close to or less than size of `CON.patchSize`
% then not all p <- ps would be equal to the dimension of CON.patchSize
% toPatch :: CONFIG x [Image] -> [Patch]
function ps = toPatch(CON,xs)

	n    = length(xs);
	bz   = floor(100/CON.batchSize);
	idxs = randsample(n,bz,n<bz   );

	ps   = {};
	for n = 1:length(idxs)
		ps{end+1} = take(CON,xs{idxs(n)});
	end

	% take :: CONFIG x Image -> Patch
	function patch = take(CON,im)

		% Get patch indices
		[m,n,~]   = size(im);
		m_max     = min(max(CON.patchSize+1,randi(m)),m);
		m_min     = max(1,m_max - CON.patchSize);
		n_max     = min(max(CON.patchSize+1,randi(n)),n);
		n_min     = max(1,n_max - CON.patchSize);

		% get patch and normalize
		patch     = im(m_min:m_max,n_min:n_max,:);
		patch     = bsxfun(@minus,patch,mean(mean(patch)));

		% trim image so it matches spacing
		if mod(size(patch,1)-CON.nv1+1, CON.spacing)~=0
			n = mod(size(patch,1)-CON.nv1+1, CON.spacing);
			patch(1:floor(n/2), : ,:)        = [];
			patch(end-ceil(n/2)+1:end, : ,:) = [];
		end

		if mod(size(patch,2)-CON.nv2+1, CON.spacing)~=0
			n = mod(size(patch,2)-CON.nv2+1, CON.spacing);
			patch(:, 1:floor(n/2), :)        = [];
			patch(:, end-ceil(n/2)+1:end, :) = [];
		end

		% hack: flip the image to simulate larger data set
		if rand()>0.5,
			% patch = fliplr(patch);
			% matlab 2013 use this version
			patch = flipdim(patch,2);
		end

	end
end


% -----------------------------------------------------------------------------------
% Logging
% -----------------------------------------------------------------------------------

% logStart : IO ()
function []   = logStart()
	fprintf('--------------------- Start Convolutional Restricted Boltzman Machine Train -------------------------\n')
end

function []   = logRun()
	fprintf('--------------------- Run Convolutional Restricted Boltzman Machine -------------------------\n')
end

% function []    = logMsg(t,errorHs,sparsityHs)
function []    = logMsg(t)

	str1 = strcat('................ completed trial_', num2str(t), '.........................\n');
	% str2 = strcat('error    = ', num2str(errorHs(end))      ,'. \n');
	% str3 = strcat('sparsity = ', num2str(sparsityHs(end))   ,'. \n');
	fprintf(str1)
	% fprintf(str2)
	% fprintf(str3)
	% fprintf('...........................................................\n');
end


% -----------------------------------------------------------------------------------
% Parrallelized Version of Contrastive Divergence Subroutines 
% Caution: Calling these functions will incur overhead 
% -----------------------------------------------------------------------------------

% toHidden2 : Config x W x B x Image -> Hidden
function H1    = toHidden2(CON,W,B,H0)

	Wr 		     = flipW(CON,W);
	convWeight2  = @(p,W,opt) convWeight(p,W,opt);

	% send each visible unit `H0` to set of hidden units `H1`, sum all `H1`
	H1           = convWeight(H0(:,:,1),Wr(:,:,1,:),'valid');
	parfor c = 2:size(H0,3)
		H1(:,:,:,c) = feval(convWeight2,H0(:,:,c),Wr(:,:,c,:),'valid');
	end

	% Sum across all channels of first layer
	H1           = sum(H1,4);

	% transform by bias and standard gaussian
	H1           = H1 + repmat(reshape(B,[1,1,length(B)]),[size(H1,1),size(H1,2),1]);
	H1           = 1/(CON.std_gaussian^2) .* H1;

	% reshape and flip weights
	function Wr = flipW(CON,W)
		[~,C,K] = size(W);
		Wr      = reshape(W(end:-1:1,:,:), [CON.nv1,CON.nv2,C,K]);
	end

	
	function H  = convWeight(p,W,opt)

		H  = [];
		for k = 1:size(W,4)
			H(:,:,k) = conv2(p,W(:,:,:,k),opt);
		end

	end

end

% toVisible2 : Config x Weights x Hidden -> Visible
function H0_   = toVisible2(CON,W,H1)

    W_      = reshape(W,[CON.nv1,CON.nv2,size(W,2),size(W,3)]);
    convHW2 = @(H1,W) convHW(H1,W);

    parfor c = 1:size(W,2)
        H0_(:,:,c) = feval(convHW2,H1,W_(:,:,c,:));
    end

    function hw = convHW(H1,W)

        for k = 1:size(H1,3)
            hw(:,:,k) = conv2(H1(:,:,k),W(:,:,:,k),'full');
        end
        hw = sum(hw,3);
    end

end

function HV    = toHV2(CON,H0,H1)

	H1nr     = H1(size(H1,1):-1:1,size(H1,2):-1:1,:   );
	HV 		 = zeros(CON.nv1,CON.nv2,size(H0,3),size(H1,3));
	go2      = @(h0,H1nr) go(h0,H1nr);

	parfor c = 1:size(H0,3)
		HV(:,:,c,:) = feval(go2,H0(:,:,c),H1nr);
	end

	HV       = reshape(HV,[CON.nv1 * CON.nv2,size(H0,3),size(H1,3)]);

	function hv = go(h0,H1nr)
		for k = 1:size(H1nr,3)
			hv(:,:,1,k) = conv2(h0,H1nr(:,:,k),'valid');
		end
	end

end

