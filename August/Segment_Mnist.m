% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------
% Author : Xiao Ling
% Module : Week Four Part II - Use SVM to segment MNIST 
% Date   : August 21sh, 2014
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

% Load river and Tree
train = load('../Data/MnistTrain.mat');
train = train.MnistTrain.Images;

test  = load('../Data/MnistTest.mat' );
test  = test.MnistTest.Images;

% -----------------------------------------------------------------------------------
% Load RBM
% -----------------------------------------------------------------------------------

rbm1 = load('../Results/MnistRbm1_40_Aug19.mat');
rbm2 = load('../Results/MnistRbm2_40_Aug19.mat');
rbm3 = load('../Results/MnistRbm3_40_Aug19.mat');

rbm1 = rbm1.mnistRbm1;
rbm2 = rbm2.mnistRbm2;
rbm3 = rbm3.mnistRbm3;

% -----------------------------------------------------------------------------------
% Reconstruct
% -----------------------------------------------------------------------------------

mnist1   = {};
err1     = [];

mnist2   = {};
err2     = [];

mnist3   = {};
err3     = [];

for k = 1:length(train)
	
	h0  = train{k};

	% one layer up and down
	h1  = RBM.run    (rbm1,h0 );
	h01 = RBM.down   (rbm1,h1 );

	mnist1{k}   = h01;
	err1  (k)   = sum(sum(abs(h0-h01)));

	% two layer up and down
	h2  = RBM.run    (rbm2,h1 );
	h12 = RBM.down   (rbm2,h2 );
	h02 = RBM.down   (rbm1,h12);

	mnist2{k}   = h02;
	err2  (k)   = sum(sum(abs(h0-h02)));

	% three layer reconstruction
	h3  = RBM.run    (rbm3,h2 );

	h23 = RBM.down   (rbm3,h3 );
	h13 = RBM.down   (rbm2,h23);
	h03 = RBM.down   (rbm1,h13);

	mnist3{k}   = h03;
	err3  (k)   = sum(sum(abs(h0-h03)));

	fprintf('--------------- Reconstructed Train Image -----------------------\n');
end

tmnist1   = {};
terr1     = [];

tmnist2   = {};
terr2     = [];

tmnist3   = {};
terr3     = [];

for k = 1:length(test)
	
	h0  = test{k};

	% one layer up and down
	h1  = RBM.run    (rbm1,h0 );
	h01 = RBM.down   (rbm1,h1 );

	tmnist1{k}   = h01;
	terr1  (k)   = sum(sum(abs(h0-h01)));

	% two layer up and down
	h2  = RBM.run    (rbm2,h1 );
	h12 = RBM.down   (rbm2,h2 );
	h02 = RBM.down   (rbm1,h12);

	tmnist2{k}   = h02;
	terr2  (k)   = sum(sum(abs(h0-h02)));

	% three layer reconstruction
	h3  = RBM.run    (rbm3,h2 );

	h23 = RBM.down   (rbm3,h3 );
	h13 = RBM.down   (rbm2,h23);
	h03 = RBM.down   (rbm1,h13);

	tmnist3{k}   = h03;
	terr3  (k)   = sum(sum(abs(h0-h03)));

	fprintf('--------------- Reconstructed Test Image -----------------------\n');
end


ReconMnist.Train.recon1 = mnist1;
ReconMnist.Train.recon2 = mnist2;
ReconMnist.Train.recon3 = mnist3;
ReconMnist.Train.err1   = err1;
ReconMnist.Train.err2   = err2;
ReconMnist.Train.err3   = err3;

ReconMnist.Test.recon1 = tmnist1;
ReconMnist.Test.recon2 = tmnist2;
ReconMnist.Test.recon3 = tmnist3;
ReconMnist.Test.err1   = terr1;
ReconMnist.Test.err2   = terr2;
ReconMnist.Test.err3   = terr3;

save ReconMnist ReconMnist

% -----------------------------------------------------------------------------------
% Segment Using SVM
% -----------------------------------------------------------------------------------






