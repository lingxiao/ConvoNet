
% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------
% Module: Convolutional Deep Belief Networks
%       
%        The detail of the algorithm is described in the following paper:
%        'Convolutional Deep Belief Networks for Scalable Unsupervised Learning of 
%         Hiearchical Representations'
% 
% Author: Xiao Ling
% Date  : July 30th, 2014
% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------


function T = ConvDBN()

	RBM      = ConvBoltzman();
	T.train  = @(a,b) trainDBN (RBM,a,b);
	T.run    = @(a,b) runDBN   (RBM,a,b);
	T.runAll = @(a,b) runDBN2  (RBM,a,b);

end

% -----------------------------------------------------------------------------------
% Learn
% -----------------------------------------------------------------------------------

% Greedy layer wise training using contrastive divergence
% trainDBN : RBM_ x CONFIG x [Image] -> DBN
function dbn = trainDBN(RBM,CON,xs)

	dbn = train(RBM,CON,CON.layer,xs,emptyDBN());	

	% train : RBM x CONFIG x Int x [Image] x DBN  -> DBN
	function o = train(RBM,CON,l,xs,dbn)

		if l == 0
			o   = dbn
		else
			T   = RBM.train (pop (CON) , runDBN2(dbn,xs)     );
			rbm = T.RBM;
			o   = train     (peel(CON) , l-1,xs,cons(rbm,dbn));
		end

	end

end

% -----------------------------------------------------------------------------------
% Inference
% -----------------------------------------------------------------------------------

% runDBN :: RBM_ x DBN x Image -> Hidden
function o = runDBN(RBM,dbn,x)

	if length(dbn) == 0
		o = x
	else
		rbm = dbn{1};
		y   = RBM.run(rbm,x);
		o   = runDBN (RBM,{dbn{2:end}},y);
	end

end

% runDBN2 :: RBM_ x DBN x [Image] -> [Hidden]
function o = runDBN2(RBM,dbn,xs)

	if length(dbn) == 0
		o   = xs
	else
		rbm = dbn{1}
		ys  = RBM.runAll(rbm,xs);
		o   = runDBN2   (RBM,{dbn{2:end}},ys);
	end
end


% -----------------------------------------------------------------------------------
% Combinators and Distinguished Element
% -----------------------------------------------------------------------------------

% identity transformation of visible layer
% emptyDBN : DBN
function dbn = emptyDBN()
	dbn = {};
end

% Stack `rbm` onto existing `dbn`
% cons : DBN -> DBN
function dbn1 = cons(rbm,dbn)
	dbn{end+1} = rbm;
	dbn1       = dbn;
end

% -----------------------------------------------------------------------------------
% Configuration
% -----------------------------------------------------------------------------------

% pop : Config -> Config
function con2 = pop(con)

	con2   = con;
	con2.K = con.K(1);

end

% peel :: Config -> Config
function con2 = peel(con)

	con2   = con;
	con2.K = con.K(2:end);

end









