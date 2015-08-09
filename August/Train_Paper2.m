% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------
% Author : Xiao Ling
% Module : Week Two Part II - train 4th layer on paper
% Date   : August 10th, 2014
% Bash   : /Applications/MATLAB_R2014a.app/bin/matlab -nodesktop -nosplash
% Sftp   : sftp xiaoling@hawkeye.eatonlab.org
% SSH    : /Applications/MATLAB_R2013b.app/bin/    ==> ./matlab
% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------


clear all
clc;

Ut    = DBNUtils();
RBM   = ConvBoltzman();

% -----------------------------------------------------------------------------------
% Load data
% -----------------------------------------------------------------------------------

% Visible layer
Paper3 		   = load('../Results/Layers/Paper_paperRbm3_80_Aug10.mat');
Paper3 		   = Paper3.Paper3;

% current RBM3
paperRbm_ 	   = load('../Results/PaperRbm4_120_320Iter_Aug12.mat');
paperRbm_      = paperRbm_.paperRbm4;

% -----------------------------------------------------------------------------------
% Configurations
% -----------------------------------------------------------------------------------

CON.numTrial     = 10;

CON.patchSize    = 70;					% Size of patch sample from image

CON.nv1          = 10;					% number of rows in visible layer
CON.nv2          = 10; 					% number of columns in visible layer
CON.K 	         = 120;		 			% number of basis in each hidden layer 
CON.spacing      = 2;					% pooling factor four 

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
% Train current layer in 10 trial increments
% -----------------------------------------------------------------------------------

fprintf('------------------------ Begin Training Paper RBM 4 ----------------------\n')

% train current layer
paperRbm4 = paperRbm_; 											% : State Rbm Rbm
totalIter = 320	                 ;   							% : State Int Int
save Current_paperRbm4 paperRbm4 ; 								% : IO ()

for k = 1:18

	paperRbm4  = RBM.trainWith(CON,Paper3,paperRbm4); 			% : State Rbm Rbm
	totalIter  = totalIter + CON.numTrial; 						% : State Int Int

	fprintf(strcat('------------- Total Iteration: ',...
	num2str(totalIter), ' ---------------\n')); 				% : IO ()
	save Current_paperRbm4 paperRbm4; 							% : IO ()

end
