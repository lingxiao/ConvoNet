% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------
% Author : Xiao Ling
% Module : Week Three Part I - train 4thth layer on liver
% Date   : August 4th, 2014
% Bash   : /Applications/MATLAB_R2014a.app/bin/matlab -nodesktop -nosplash
% Sftp   : sftp xiaoling@hawkeye.eatonlab.org
% SSH    : /Applications/MATLAB_R2013b.app/bin/    ==> ./matlab
% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------

% clear all
clc;

Pr    = Prelude();
Ut    = DBNUtils();
RBM   = ConvBoltzman();

% -----------------------------------------------------------------------------------
% Load data
% -----------------------------------------------------------------------------------

% Visible layer
Liver3 		   = load('../Results/Layers/Liver_liverRbm3_80_Aug12.mat')
Liver3 		   = Liver3.liver3;

liverRbm_ 	   = load('../Results/LiverRbm4_Aug15.mat');
liverRbm_      = liverRbm_.liverRbm4;

% -----------------------------------------------------------------------------------
% Configurations
% -----------------------------------------------------------------------------------

CON.numTrial     = 10;

CON.patchSize    = 70;					% Size of patch sample from image

CON.nv1          = 10;					% number of rows in visible layer
CON.nv2          = 10; 					% number of columns in visible layer
CON.K 	         = 120;		 			% number of basis in each hidden layer 
CON.spacing      = 2;					% pooling factor

CON.bias_mode    = 'simple';
CON.pbias        = 0.002;
CON.pbias_lb     = 0.002;
CON.pbias_lambda = 5;

CON.std_gaussian = 0.1;
CON.sigma_start  = 0.2;
CON.sigma_stop   = 0.1;
CON.epsilon      = 0.01;

CON.batchSize    = 2;
CON.l2reg        = 0.01;

% -----------------------------------------------------------------------------------
% Train layers
% -----------------------------------------------------------------------------------

fprintf('------------------------ Begin Training liver RBM 4 ----------------------\n')
fprintf('------------------------ Train Configurations:      ----------------------\n')
CON
fprintf('\n');

% train current layer
liverRbm4 = liverRbm_; 		 									% : State Rbm Rbm
totalIter = 320   				 ;								% : State Int Int
save Current_liverRbm4 liverRbm4 ; 								% : IO ()

for k = 1:(500-totalIter)/CON.numTrial

	liverRbm4  = RBM.trainWith(CON,Liver3,liverRbm4); 			% : State Rbm Rbm
	totalIter  = totalIter + CON.numTrial; 						% : State Int Int

	fprintf(strcat('------------- Total Iteration: ',...
	num2str(totalIter), ' ---------------\n')); 				% : IO ()
	save Current_liverRbm4 liverRbm4; 							% : IO ()

end

% Ut.show(liverRbm1.W ,'liver Basis Layer One 18 basis' );
% Ut.show(liverRbm2.W ,'liver Basis Layer two 40 basis' );








