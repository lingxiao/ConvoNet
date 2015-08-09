% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------
% Author : Xiao Ling
% Module : Week One Part V - run 1st layer with 18 basis, 2nd layer with 40 basis on organ
% Date   : August 3rd 2014
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
Train  		   = load('../Data/Surgery.mat');
Train 		   = Train.Train.large;
% Skin 	       = Pr.mapCell(@(m) Ut.whiten(m), Train.skin );
% Fat 		   = Pr.mapCell(@(m) Ut.whiten(m), Train.fat);

Liver 		   = Pr.mapCell(@(m) Ut.whiten(m), Train.liver);

% load RBM
liverRbm1 	   = load('../Results/LiverRBM1_18_Aug3.mat');
liverRbm1 	   = liverRbm1.liverRbm1;


% -----------------------------------------------------------------------------------
% Configurations
% -----------------------------------------------------------------------------------

CON.numTrial     = 500;

CON.patchSize    = 70;					% Size of patch sample from image

CON.nv1          = 10;					% number of rows in visible layer
CON.nv2          = 10; 					% number of columns in visible layer
CON.K 	         = 40;		 			% number of basis in each hidden layer 
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

% transform
Liver2    = RBM.runAll(liverRbm1,Liver);

% train 
liverRbm2 = RBM.train(CON,Liver2);

% plot
Ut.show(liverRbm2.W , 'Liver Basis Layer Two 40 Basis');









