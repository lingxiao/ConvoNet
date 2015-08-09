% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------
% Module: Core Fast sparse coding algorithms - adapted from code by Honglak lee
%       
%           minimize_B,S   0.5*||X - B*S||^2 + beta*sum(abs(S(:)))
%           subject to   ||B(:,j)||_2 <= l2norm, forall j=1...size(S,1)
%        
%        The detail of the algorithm is described in the following paper:
%        'Efficient Sparse Codig Algorithms', Honglak Lee, Alexis Battle, Rajat Raina,
%         Andrew Y. Ng, 
%        Advances in Neural Information Processing Systems (NIPS) 19, 2007
% 
% Author: Xiao Ling
% Date  : May 8th, 2014
% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------

% Export Module Functions
function T = SparseEncodeCore()

    addpath('Mex')
    T.config         = @(a)       config(a);
	T.learn          = @(a,b)     sparseEncode(a,b);
    T.encode         = @(a,b,c)   encode(a,b,c);

end

% -----------------------------------------------------------------------------------
% Algorithm parameters
% -----------------------------------------------------------------------------------

% Custom and default configurations
function CONFIG = config(T)

    % Algorithm main parameters
    Main.patchSize      = T.patchSize;
    Main.numPatches     = T.numPatch;
    Main.numIter        = T.numIter;
    Main.batchSize      = 1000; 

    % update basis using lagrange dual parameter
    Basis.varBasis     = 1;
    Basis.numBasis     = T.numBasis;
    Basis.c            = Basis.varBasis^2;       % constraint on sum of basis weights
    Basis.options      = optimoptions('fmincon','GradObj','on','Hessian','user-supplied','Algorithm','trust-region-reflective');

    % Conjugate gradient function parameters
    Weights.beta       = T.beta;                 % sparsity penalty param
    Weights.sparseFunc = T.sparseFunc;           % 'epsL1' or 'L1'
    Weights.eps        = T.eps;                  % epsilon for epsilon-L1 sparsity - default should be 0.01
    Weights.sigma      = 1;                      % standard deviation of error (?)
    Weights.noise      = 1;                      % what is this
    Weights.lambda     = 1/Weights.noise;        % what is inverse of noise?
    Weights.tol        = 0.005;                  % solution tolerance for conjugate gradient
    Weights.numIter    = 100;                    % gradient function maximum iteration

    % Main configuration
    CONFIG.Main        = Main;
    CONFIG.Basis       = Basis;
    CONFIG.Weights     = Weights;
end

% -----------------------------------------------------------------------------------
% Algorithm Main
% -----------------------------------------------------------------------------------


% sparseEncode :: CONFIG -> Mat m n -> (Mat m k, Mat k p,Statistics)
function T = sparseEncode(CONFIG,X)

	% select basis update function based on penalty function
    updateWs  = selectPenalty(CONFIG); 

    % Compute parameters based on CONFIG
	batchNum  = size(X,2)/CONFIG.Main.batchSize;
   
	% Randomly initalize basis centered at 0, and initialize weights to be all 0
	Bs         = rand(size(X,1),CONFIG.Basis.numBasis) - 0.5;
	Ws         = zeros(size(Bs,2),size(X,2));
    stat       = emptyStat();
	iter       = 1;							


    % Main loop
	while iter <= CONFIG.Main.numIter 

        is   = randperm(size(X,2));   % take random permutation of samples to prevent overfitting of data
        stat = resetStat(stat);       % overwrite current statisics

        % Update weights and basis wrt batches of data
		for b = 1:batchNum

			% pick out active subset of all image batches 
			% Find best set of weights that describe current batch given current basis
		    % Find best set of basis that describe current batch given current weights
			[Xb,js]   = askBatch(CONFIG,X ,b ,is      );
			weights   = updateWs(iter  ,Xb,Bs,Ws(:,js));
		    basis     = updateBs(CONFIG,Xb,weights    );

            % Overwrite existing Bs,Ws
            Ws(:,js)  = weights;
            Bs        = basis;

            % overwrite current statistics
            stat      = updateStat(CONFIG,stat,basis,weights,Xb);
		end

       % Record averge statiscs
        stat  = incrStat(CONFIG,stat,iter);
        iter  = iter + 1;

	end

    % Output basis, weights, and statistics
    T.Bs     = Bs;
    T.Ws     = Ws;
    T.stat   = stat;

end

% Exported version of `updateWs` where the statisics is calculated
function T = encode(CONFIG,Xb,Bs)

    updateWs  = selectPenalty(CONFIG); 
    ws        = updateWs(1,Xb,Bs,[]);
    T.Weights = ws;
    T.Stat    = updateStat(CONFIG,emptyStat(),Bs,ws,Xb);

end

% -----------------------------------------------------------------------------------
% Algorithm Subroutines - General
% -----------------------------------------------------------------------------------

% Get `bth` batch from sample `X`, where each column is an image patch and the
% columns are arranged by indices `is
function [Xb,isb] = askBatch(CONFIG,X,b,is)
	
	a   = (b-1)*CONFIG.Main.batchSize;
	isb = is(max(a,1):(a+CONFIG.Main.batchSize));
	Xb  = X(:,isb);
end

% Update weights using conjugate dual method (Mex file)
function ws       = updateWsEps(CONFIG,Xb,Bs,~,~)

	ws0  	    = Bs'*Xb;
	nrm  	    = sum(Bs.*Bs)';
	ws0  	    = ws0 ./ repmat(nrm,[1,size(ws0,2)]);
	Weights     = CONFIG.Weights;

	[ws,~,~,~]  = cgf_sc2(Bs,Xb,ws0,2,Weights.lambda,Weights.beta,Weights.sigma,Weights.tol,Weights.numIter,0,0,Weights.eps);
end

function f        = selectPenalty(CONFIG)

    if strcmp(CONFIG.Weights.sparseFunc,'L1')
        f = @(iter,Xb,Bs,ws) updateWsL1 (CONFIG,Xb,Bs,ws,iter);
    else
        f = @(iter,Xb,Bs,ws) updateWsEps(CONFIG,Xb,Bs,ws,iter);
    end

end


% Update weights using feature sign 
function Ws       = updateWsL1(CONFIG,Xb,Bs,ws,iter)

    num = CONFIG.Weights.beta/CONFIG.Weights.sigma*CONFIG.Weights.noise;

    if iter == 1
        Ws = l1ls_featuresign(Bs,Xb,num);
    else
        Ws = l1ls_featuresign(Bs,Xb,num,ws);
    end

end


function Bs       = updateBs(CONFIG,Xb,ws)

    % set up the function to be solved using fmincon
	wwt      = ws*ws';
	xwt      = Xb*ws';
	dualLam0 = 10*abs(rand(size(ws,1),1));
	xbnrm    = sum(sum(Xb.^2));
	lb       = zeros(size(dualLam0));

	% find constrained min of f, a nonlinear multi-variate function
	f        = @(x) objBasisDual(x, wwt, xwt, Xb, CONFIG.Basis.c, xbnrm );
	dualLam1 = fmincon(f,dualLam0,[],[],[],[],lb,[],[],CONFIG.Basis.options);

	Bs       = ((wwt+diag(dualLam1)) \ xwt')';
end

% ComputoBasis  the objective function value at x
function [f,g,H]  = objBasisDual(dual_lambda, SSt, XSt, X, c, trXXt)
    
    L       = size(XSt,1);
    M       = length(dual_lambda);
    SSt_inv = inv(SSt + diag(dual_lambda));

    % trXXt = sum(sum(X.^2));
    if L>M
        % (M*M)*((M*L)*(L*M)) => MLM + MMM = O(M^2(M+L))
        f = -trace(SSt_inv*(XSt'*XSt))+trXXt-c*sum(dual_lambda);
        
    else
        % (L*M)*(M*M)*(M*L) => LMM + LML = O(LM(M+L))
        f = -trace(XSt*SSt_inv*XSt')+trXXt-c*sum(dual_lambda);
    end
    f= -f;

    if nargout > 1   % fun called with two output arguments
        % Gradient of the function evaluated at x
        g    = zeros(M,1);
        temp = XSt*SSt_inv;
        g    = sum(temp.^2) - c;
        g    = -g;
        
        
        if nargout > 2
            % Hessian evaluated at x
            % H = -2.*((SSt_inv*XSt'*XSt*SSt_inv).*SSt_inv);
            H = -2.*((temp'*temp).*SSt_inv);
            H = -H;
        end
    end

    return
end

% -----------------------------------------------------------------------------------
% Algorithm Subroutines - Feature Sign
% -----------------------------------------------------------------------------------

function Xout = l1ls_featuresign (A, Y, gamma, Xinit)
    % The feature-sign search algorithm
    % L1-regularized least squares problem solver
    %
    % This code solves the following problem:
    % 
    %    minimize_s 0.5*||y - A*x||^2 + gamma*||x||_1
    % 
    % The detail of the algorithm is described in the following paper:
    % 'Efficient Sparse Coding Algorithms', Honglak Lee, Alexis Battle, Rajat Raina, Andrew Y. Ng, 
    % Advances in Neural Information Processing Systems (NIPS) 19, 2007
    %
    % Written by Honglak Lee <hllee@cs.stanford.edu>
    % Copyright 2007 by Honglak Lee, Alexis Battle, Rajat Raina, and Andrew Y. Ng

    warning('off', 'MATLAB:divideByZero');

    use_Xinit= false;
    if exist('Xinit', 'var')
        use_Xinit= true;
    end

    Xout  = zeros(size(A,2), size(Y,2));
    AtA   = A'*A;
    AtY   = A'*Y;

    % rankA = rank(AtA);
    rankA = min(size(A,1)-10, size(A,2)-10);

    for i=1:size(Y,2)
        if mod(i, 100)==0, fprintf('.'); end %fprintf(1, 'l1ls_featuresign: %d/%d\r', i, size(Y,2)); end
        
        if use_Xinit
            idx1                = find(Xinit(:,i)~=0);
            maxn                = min(length(idx1), rankA);
            xinit               = zeros(size(Xinit(:,i)));
            xinit(idx1(1:maxn)) =  Xinit(idx1(1:maxn), i);
            [Xout(:,i), fobj]   = ls_featuresign_sub (A, Y(:,i), AtA, AtY(:, i), gamma, xinit);
        else
            [Xout(:,i), fobj]   = ls_featuresign_sub (A, Y(:,i), AtA, AtY(:, i), gamma);
        end
    end

    fprintf(1, '\n');
    warning('on', 'MATLAB:divideByZero');

    % added by me on May 7th -> code bit from top routine sparse_coding.m
    Xout(find(isnan(Xout))) = 0;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [x, fobj] = ls_featuresign_sub (A, y, AtA, Aty, gamma, xinit)

    [L,M] = size(A);

    rankA = min(size(A,1)-10, size(A,2)-10);

    % Step 1: Initialize
    usexinit = false;
    if ~exist('xinit', 'var') || isempty(xinit)
        xinit= [];
        x= sparse(zeros(M,1));
        theta= sparse(zeros(M,1));
        act= sparse(zeros(M,1));
        allowZero = false;
    else
        % xinit = [];
        x= sparse(xinit);
        theta= sparse(sign(x));
        act= sparse(abs(theta));
        usexinit = true;
        allowZero = true;
    end

    fname_debug = sprintf('../tmp/fsdebug_%x.mat', datestr(now, 30));

    fobj = 0; %fobj_featuresign(x, A, y, AtA, Aty, gamma);

    ITERMAX=1000;
    optimality1=false;
    for iter=1:ITERMAX
        % check optimality0
        act_indx0 = find(act == 0);
        grad = AtA*sparse(x) - Aty;
        theta = sign(x);

        optimality0= false;
        % Step 2
        [mx,indx] = max (abs(grad(act_indx0)));

        if ~isempty(mx) && (mx >= gamma) && (iter>1 || ~usexinit)
            act(act_indx0(indx)) = 1;
            theta(act_indx0(indx)) = -sign(grad(act_indx0(indx)));
            usexinit= false;
        else
            optimality0= true;
            if optimality1
                break;
            end
        end
        act_indx1 = find(act == 1);

        if length(act_indx1)>rankA
            warning('sparsity penalty is too small: too many coefficients are activated');
            return;
        end

        if isempty(act_indx1) %length(act_indx1)==0
            % if ~assert(max(abs(x))==0), save(fname_debug, 'A', 'y', 'gamma', 'xinit'); error('error'); end
            if allowZero, allowZero= false; continue, end
            return;
        end

        % if ~assert(length(act_indx1) == length(find(act==1))), save(fname_debug, 'A', 'y', 'gamma', 'xinit'); error('error'); end
        k=0;
        while 1
            k=k+1;

            if k>ITERMAX
                warning('Maximum number of iteration reached. The solution may not be optimal');
                % save(fname_debug, 'A', 'y', 'gamma', 'xinit');
                return;
            end

            if isempty(act_indx1) % length(act_indx1)==0
                % if ~assert(max(abs(x))==0), save(fname_debug, 'A', 'y', 'gamma', 'xinit'); error('error'); end
                if allowZero, allowZero= false; break, end
                return;
            end

            % Step 3: feature-sign step
            [x, theta, act, act_indx1, optimality1, lsearch, fobj] = compute_FS_step (x, A, y, AtA, Aty, theta, act, act_indx1, gamma);

            % Step 4: check optimality condition 1
            if optimality1 break; end;
            if lsearch >0 continue; end;

        end
    end

    if iter >= ITERMAX
        warning('Maximum number of iteration reached. The solution may not be optimal');
        % save(fname_debug, 'A', 'y', 'gamma', 'xinit');
    end

    if 0  % check if optimality
        act_indx1 = find(act==1);
        grad = AtA*sparse(x) - Aty;
        norm(grad(act_indx1) + gamma.*sign(x(act_indx1)),'inf')
        find(abs(grad(setdiff(1:M, act_indx1)))>gamma)
    end

    fobj = fobj_featuresign(x, A, y, AtA, Aty, gamma);

    return;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x, theta, act, act_indx1, optimality1, lsearch, fobj] = compute_FS_step (x, A, y, AtA, Aty, theta, act, act_indx1, gamma)

    x2 = x(act_indx1);
    % A2 = A(:, act_indx1);
    AtA2 = AtA(act_indx1, act_indx1);
    theta2 = theta(act_indx1);

    % call matlab optimization solver..
    x_new = AtA2 \ ( Aty(act_indx1) - gamma.*theta2 ); % RR
    % opts.POSDEF=true; opts.SYM=true; % RR
    % x_new = linsolve(AtA2, ( Aty(act_indx1) - gamma.*theta2 ), opts); % RR
    optimality1= false;
    if (sign(x_new) == sign(x2)) 
        optimality1= true;
        x(act_indx1) = x_new;
        fobj = 0; %fobj_featuresign(x, A, y, AtA, Aty, gamma);
        lsearch = 1;
        return; 
    end

    % do line search: x -> x_new
    progress = (0 - x2)./(x_new - x2);
    lsearch=0;
    %a= 0.5*sum((A2*(x_new- x2)).^2);
    a= 0.5*sum((A(:, act_indx1)*(x_new- x2)).^2);
    b= (x2'*AtA2*(x_new- x2) - (x_new- x2)'*Aty(act_indx1));
    fobj_lsearch = gamma*sum(abs(x2));
    [sort_lsearch, ix_lsearch] = sort([progress',1]);
    remove_idx=[];
    for i = 1:length(sort_lsearch)
        t = sort_lsearch(i); if t<=0 | t>1 continue; end
        s_temp= x2+ (x_new- x2).*t;
        fobj_temp = a*t^2 + b*t + gamma*sum(abs(s_temp));
        if fobj_temp < fobj_lsearch
            fobj_lsearch = fobj_temp;
            lsearch = t;
            if t<1  remove_idx = [remove_idx ix_lsearch(i)]; end % remove_idx can be more than two..
        elseif fobj_temp > fobj_lsearch
            break;
        else
            if (sum(x2==0)) == 0
                lsearch = t;
                fobj_lsearch = fobj_temp;
                if t<1  remove_idx = [remove_idx ix_lsearch(i)]; end % remove_idx can be more than two..
            end
        end
    end

    % if ~assert(lsearch >=0 && lsearch <=1), save(fname_debug, 'A', 'y', 'gamma', 'xinit'); error('error'); end

    if lsearch >0
        % update x
        x_new = x2 + (x_new - x2).*lsearch;
        x(act_indx1) = x_new;
        theta(act_indx1) = sign(x_new);  % this is not clear...
    end

    % if x encounters zero along the line search, then remove it from
    % active set
    if lsearch<1 & lsearch>0
        %remove_idx = find(x(act_indx1)==0);
        remove_idx = find(abs(x(act_indx1)) < eps);
        x(act_indx1(remove_idx))=0;

        theta(act_indx1(remove_idx))=0;
        act(act_indx1(remove_idx))=0;
        act_indx1(remove_idx)=[];
    end
    fobj_new = 0; %fobj_featuresign(x, A, y, AtA, Aty, gamma);

    fobj = fobj_new;

    return;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [f, g] = fobj_featuresign(x, A, y, AtA, Aty, gamma)

    f= 0.5*norm(y-A*x)^2;
    f= f+ gamma*norm(x,1);

    if nargout >1
        g= AtA*x - Aty;
        g= g+ gamma*sign(x);
    end

    return;
end

%%%%%%%%%%%%%%%%%%%%%

function retval = assert(expr)
    retval = true;
    if ~expr 
        % error('Assertion failed');
        warning ('Assertion failed');
        retval = false;
    end
    return
end

% -----------------------------------------------------------------------------------
% Algorithm Statistics 
% -----------------------------------------------------------------------------------

function T0 = emptyStat()

    T0 = resetStat({});
end

function T1 = resetStat(T0)

    T0.fobj_total          = 0;
    T0.fresidue_total      = 0;
    T0.fsparsity_total     = 0;
    T0.var_tot             = 0;
    T0.svar_tot            = 0;
    T1                     = T0;
end

% Compute and record statistics
function T1 = updateStat(CONFIG,T0,Bs,ws,Xb)

    W = CONFIG.Weights;

    [fobj, fres, fspars]   = getObjective(Bs,ws,Xb,W.sparseFunc, W.noise, W.beta, W.sigma,W.eps);
    
    T0.fobj_total          = T0.fobj_total       + fobj;
    T0.fresidue_total      = T0.fresidue_total   + fres;
    T0.fsparsity_total     = T0.fsparsity_total  + fspars;
    T0.var_tot             = T0.var_tot          + sum(sum(ws.^2,1))/size(ws,1);
   
    T1 = T0;
end

function T1 = incrStat(CONFIG,T0,iter)

    n                      = CONFIG.Main.numPatches;
    T0.fobj_avg     (iter) = T0.fobj_total       / n;
    T0.fresidue_avg (iter) = T0.fresidue_total   / n;
    T0.fsparsity_avg(iter) = T0.fsparsity_total  / n;
    T0.var_avg      (iter) = T0.var_tot          / n;
    T1                     = T0;

end

function [fobj, fresidue, fsparsity] = getObjective(A, S, X, sparsity, noise_var, beta, sigma, epsilon)


    if ~strcmp(sparsity, 'log') && ~strcmp(sparsity, 'huberL1') && ~strcmp(sparsity,'epsL1') && ...
            ~strcmp(sparsity,'FS') && ~strcmp(sparsity, 'L1') && ~strcmp(sparsity,'LARS') && ...
            ~strcmp(sparsity, 'trueL1') && ~strcmp(sparsity, 'logpos')
        error('sparsity function is not properly specified!\n');
    end

    if strcmp(sparsity, 'huberL1') || strcmp(sparsity, 'epsL1')
        if ~exist('epsilon','var') || isempty(epsilon) || epsilon==0
            error('epsilon was not set properly!\n')
        end
    end

    E         = A*S - X;
    lambda    = 1/noise_var;
    fresidue  = 0.5*lambda*sum(sum(E.^2));

    if strcmp(sparsity, 'log')
        fsparsity = beta*sum(sum(log(1+(S/sigma).^2)));
    elseif strcmp(sparsity, 'huberL1') 
        fsparsity = beta*sum(sum(huber_func(S/sigma, epsilon)));
    elseif strcmp(sparsity, 'epsL1')
        fsparsity = beta*sum(sum(sqrt(epsilon+(S/sigma).^2)));
    elseif strcmp(sparsity, 'L1') | strcmp(sparsity, 'LARS') | strcmp(sparsity, 'trueL1') | strcmp(sparsity, 'FS')        
        fsparsity = beta*sum(sum(abs(S/sigma)));
    elseif strcmp(sparsity, 'logpos')
        fsparsity = beta*sum(sum(log(1+(S/sigma))));
    end

    fobj = fresidue + fsparsity;


    % E         = A*S - X;
    % lambda    = 1/noise_var;
    % fresidue  = 0.5*lambda*sum(sum(E.^2));
    % fsparsity = beta*sum(sum(sqrt(epsilon+(S/sigma).^2)));
    % fobj      = fresidue + fsparsity;

end

