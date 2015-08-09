% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------
% Author : Xiao Ling
% Module : Week One Part III - run on provided image
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
Paper 		  = load('../Data/Paper.mat');						
Paper 	      = Paper.Paper;

% load basis
paperRbm1     = load('../Results/PaperRBM1_18_Aug2.mat');
paperRbm1     = paperRbm1.paperRbm1;

% -----------------------------------------------------------------------------------
% Configurations
% -----------------------------------------------------------------------------------

CON.numTrial     = 500;

CON.nv1          = 10;
CON.nv2          = 10;
CON.K 	         = 40;
CON.spacing      = 2;

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

% second layer
Paper2 	         = RBM.runAll(paperRbm1,Paper);

% train second layer
paperRbm1        = RBM.train(CON,Paper2);

Ut.show(paperRbm1.W ,'Paper Basis Layer One 18 basis' );
Ut.show(paperRbm2.W ,'Paper Basis Layer Two 40 basis' );







