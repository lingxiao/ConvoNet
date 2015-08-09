% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------
% Author : Xiao Ling
% Module : Week Four Part I - train MNIST handwriting images for 2nd layer
%  							  brute force sensitivity analysis wrt all parameters
% Date   : August 18th, 2014
% Bash   : /Applications/MATLAB_R2014a.app/bin/matlab -nodesktop -nosplash
% Sftp   : sftp xiaoling@hawkeye.eatonlab.org
% SSH    : /Applications/MATLAB_R2013b.app/bin/    ==> ./matlab
% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------

clear all
clc;

Pr    = Prelude();
Ut    = DBNUtils();
RBM   = ConvBoltzman();

% -----------------------------------------------------------------------------------
% Load data
% -----------------------------------------------------------------------------------

% Visible layer
Data 		   = load('../Data/MnistTrain.mat');
H0 			   = Data.MnistTrain.Images;

% Hidden layer one
H1 			   = load('../Results/Layers/MNIST_MnistRbm1_40_Aug19.mat');
H1 			   = H1.H1;

% -----------------------------------------------------------------------------------
% CON1figuration Sensitivity analysis
% -----------------------------------------------------------------------------------

% Base Base
CON1.numTrial     = 200; 				% Low num trials to speed up training

CON1.patchSize    = 70;					% Size of patch sample from image

CON1.nv1          = 12;					% number of rows in visible layer
CON1.nv2          = 12; 				% number of columns in visible layer
CON1.K 	          = 40;		 			% number of basis in each hidden layer 
CON1.spacing      = 2;					% pooling factor

CON1.cdk 		  = 2; 					% burn in period for gibbs sampling
CON1.bias_mode    = 'simple';
CON1.pbias        = 0.002;
CON1.pbias_lb     = 0.002;
CON1.pbias_lambda = 5;

CON1.std_gaussian = 0.1;
CON1.sigma_start  = 0.2;
CON1.sigma_stop   = 0.1;
CON1.epsilon      = 0.01;

CON1.batchSize    = 2;
CON1.l2reg        = 0.01;  			


% Vary nv
CON2      			= CON1;
CON2.nv1  			= 6;
CON2.nv2  			= 6;

CON3      			= CON1;
CON3.nv1  			= 15;
CON3.nv2  			= 15;

% Vary K
CON4      			= CON1;
CON4.K    			= 60;

CON5      			= CON1;
CON5.K    			= 20;

% Vary spacing
CON6          		= CON1;
CON6.spacing  		= 1;

CON7          		= CON1;
CON7.spacing  		= 4;

% Vary cdk
CON8 		  		= CON1;
CON8.cdk      		= 10;

% vary pbias
CON9                = CON1;
CON9.pbias          = 0.0002;

CON10               = CON1;
CON10.pbias         = 0.02;

% vary pbias_lb
CON11           	= CON1;
CON11.pbias_lb  	= 0.0002;

CON12           	= CON1;
CON12.pbias_lb  	= 0.02;

% vary pbias_labmda
CON13               = CON1;
CON13.pbias_lambda  = 1;

CON13               = CON1;
CON13.pbias_lambda  = 10;

% vary std_guassian
CON14 			   = CON1;
CON14.std_gaussian = 1;

CON15 			   = CON1;
CON15.std_gaussian = 0.01;

% vary sigma_start and stop
CON16 			   = CON1;
CON16.sigma_start  = 0.02;
CON16.sigma_stop   = 0.01;

% vary epsilon
CON17 			   = CON1;
CON17.epsilon      = 0.001;

CON18 			   = CON1;
CON18.epsilon      = 0.1;

% vary l2reg 	
CON19 			   = CON1;
CON19.l2reg	       = 0.001;

CON20 			   = CON1;
CON20.l2reg 	   = 0.1;


% CONs = [CON1,CON2,CON3,CON4,CON5,CON6,CON7,CON8,CON9,CON10,CON11,CON12,...
CONs = [CON3,CON4,CON5,CON6,CON7,CON8,CON9,CON10,CON11,CON12,...
        CON13,CON14,CON15,CON16,CON17,CON18,CON19,CON20];

% -----------------------------------------------------------------------------------
% Train layers
% -----------------------------------------------------------------------------------


title = 'MNIST Layer 2'

fprintf(strcat('------------------------ Begin Training', title, ' ----------------------\n'))


Rbms = {};

for k = 1:length(CONs)

	CON = CONs(k);

	fprintf('------------------------ Train CONfigurations:      ----------------------\n')
	CON
	fprintf('\n');

	% train RBM
	mnistRbm2 = RBM.train(CON,H1);

	% Display RBM
	key       = strcat('nv: '  , num2str(CON.nv1)  ,{' '}, 'Basis: '  , num2str(CON1.K)           , {' '},...
					   'burn:' , num2str(CON.cdk)  ,{' '}, 'l2reg: '  , num2str(CON1.l2reg)       , {' '},...
					   'pbias:', num2str(CON.pbias),{' '}, 'pbias_lb:', num2str(CON1.pbias_lambda), {' '},...
					   'std_gaussian:', num2str(CON.std_gaussian), {' '}, ...
					   'sigma_start :', num2str(CON.sigma_start ), {' '}, ...
					   'sigma_stop  :', num2str(CON.sigma_stop  ), {' '}, ...
					   'epsilon     :', num2str(CON.epsilon     ), {' '}  ...
					   );


	% Ut.show(mnistRbm2.Rbm.W            ,strcat(title,{' '},key ) )
	% figure; plot(mnistRbm2.Ferr)       ; title('Ferror' 		 )
	% figure; plot(mnistRbm2.SparsityErr); title('Sparsity Error'  )

	Rbms{k} = mnistRbm2;

	save mnist_SensitivityAnalsysis Rbms;

end





