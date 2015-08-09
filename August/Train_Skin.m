% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------
% Author : Xiao Ling
% Module : Week Three Part I - train 3rd layer on skin
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
Skin2 		   = load('../Results/Layers/Skin_skinRbm2_40_Aug13.mat')
Skin2 		   = Skin2.skin2;

% -----------------------------------------------------------------------------------
% Configurations
% -----------------------------------------------------------------------------------

CON.numTrial     = 10;

CON.patchSize    = 70;					% Size of patch sample from image

CON.nv1          = 10;					% number of rows in visible layer
CON.nv2          = 10; 					% number of columns in visible layer
CON.K 	         = 80;		 			% number of basis in each hidden layer 
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

fprintf('------------------------ Begin Training Skin RBM 3 ----------------------\n')
fprintf('------------------------ Train Configurations:      ----------------------\n')
CON
fprintf('\n');

% train current layer
skinRbm3 = RBM.train(CON,Skin2);								% : State Rbm Rbm
totalIter = 10    			   ;								% : State Int Int
save Current_skinRbm3 skinRbm3 ; 								% : IO ()

for k = 1:(500-totalIter)/CON.numTrial

	skinRbm3  = RBM.trainWith(CON,Skin2,skinRbm3); 			% : State Rbm Rbm
	totalIter  = totalIter + CON.numTrial; 						% : State Int Int

	fprintf(strcat('------------- Total Iteration: ',...
	num2str(totalIter), ' ---------------\n')); 				% : IO ()
	save Current_skinRbm3 skinRbm3; 							% : IO ()

end

% Ut.show(liverRbm1.W ,'liver Basis Layer One 18 basis' );
% Ut.show(liverRbm2.W ,'liver Basis Layer two 40 basis' );








