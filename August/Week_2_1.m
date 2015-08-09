% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------
% Author : Xiao Ling
% Module : Week Two Part I - train 3rd layer on tree
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

% load visible layer
Tree2      = load('../Results/Layers/Tree_treeRbm2_40_Aug5.mat');
Tree2      = Tree2.Tree2;

% load current RBM
treeRbm_   = load('../Results/TreeRBM3_80_Aug6_90Iter.mat');
treeRbm_   = treeRbm_.treeRbm3;

% -----------------------------------------------------------------------------------
% Configurations
% -----------------------------------------------------------------------------------

CON.numTrial     = 41;

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

treeRbm3 = treeRbm_; 								   % : State Rbm Rbm 

for k = 1:10
	treeRbm3  = RBM.trainWith(CON,Tree2,treeRbm3);     % : State Rbm Rbm 
	save Current_treeRbm3 treeRbm3 					   % : IO ()
end








