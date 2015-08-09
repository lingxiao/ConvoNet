% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------
% Author : Xiao Ling
% Module : Week One Part I - run 1st layer with 18 basis, 2nd layer with 40 basis
% Date   : August 2nd, 2014
% Bash   : /Applications/MATLAB_R2014a.app/bin/matlab -nodesktop -nosplash
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

% load patches
Tree  		   = load('../Data/Tree1.mat');
Tree	   	   = Tree.Tree.Processed;

% load layer one basis
treeRbm1 	   = load('../Results/TreeRBM1_18_Aug3.mat');
treeRbm1       = treeRbm1.treeRbm1;

% -----------------------------------------------------------------------------------
% Configurations
% -----------------------------------------------------------------------------------

CON.numTrial     = 500;

CON.patchSize    = 70;					% Size of patch sample from image

CON.nv1          = 10;					% number of rows in visible layer
CON.nv2          = 10; 					% number of columns in visible layer
CON.K 	         = 18;		 			% number of basis in each hidden layer 
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
CON.batch_ws     = 70;
CON.l2reg        = 0.01;


% -----------------------------------------------------------------------------------
% Train layers
% -----------------------------------------------------------------------------------

% tranform layer one
Tree2      = RBM.runAll(treeRbm1,Tree);

% train second layer
treeRbm2   = RBM.train(CON,Tree2);

Ut.show(treeRbm1.W , 'Tree Basis Layer One 18 basis' );
Ut.show(treeRbm2.W , 'Tree Basis Layer two 40 basis' );








