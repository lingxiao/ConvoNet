% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------
% Author : Xiao Ling
% Module : Week Two Part I - train 3rd layer on river
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
River2 		   = load('../Results/Layers/River_riverRbm2_40_Aug3.mat');
River2 		   = River2.River2;

% Load current RBM
% riverRbm_ 	   = load('../Results/RiverRBM3_80_398Trial_Aug6.mat');
% riverRbm_      = riverRbm_.riverRbm3;

% -----------------------------------------------------------------------------------
% Configurations
% -----------------------------------------------------------------------------------

CON.numTrial     = 10;

CON.patchSize    = 70;					% Size of patch sample from image

CON.nv1          = 10;					% number of rows in visible layer
CON.nv2          = 10; 					% number of columns in visible layer
CON.K 	         = 80;		 			% number of basis in each hidden layer 
CON.spacing      = 4;					% pooling factor

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

fprintf('------------------------ Begin Training River RBM 3 ----------------------\n')
fprintf('------------------------ Train Configurations:      ----------------------\n')
CON
fprintf('\n');

% train current layer
riverRbm3 = RBM.train(CON,River2);								% : State Rbm Rbm
totalIter = 0      ; 											% : State Int Int
save Current_riverRbm3 riverRbm3; 								% : IO ()

for k = 1:(500-totalIter)/CON.numTrial

	riverRbm3  = RBM.trainWith(CON,River2,riverRbm3); 			% : State Rbm Rbm
	totalIter  = totalIter + CON.numTrial; 						% : State Int Int

	fprintf(strcat('------------- Total Iteration: ',...
	num2str(totalIter), ' ---------------\n')); 				% : IO ()
	save Current_riverRbm3 riverRbm3; 							% : IO ()

end

% Ut.show(riverRbm1.W ,'River Basis Layer One 18 basis' );
% Ut.show(riverRbm2.W ,'River Basis Layer two 40 basis' );








