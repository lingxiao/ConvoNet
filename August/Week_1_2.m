% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------
% Author : Xiao Ling
% Module : Week Five Part II - run on 
% Date   : August 2nd, 2014
% Bash   : /Applications/MATLAB_R2014a.app/bin/matlab -nodesktop -nosplash
% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------

% clear all
clc;

Pr    = Prelude();
Ut    = DBNUtils();
RBM   = ConvBoltzman();

-----------------------------------------------------------------------------------
Load data
-----------------------------------------------------------------------------------

% load patches
Tree   		   = load('../Data/Tree.mat');
trees   	   = Tree.Data.large;

% load basis
treeRbm1  	   = load('../Results/TreeRBM1_18_Aug2.mat');
treeRbm1 	   = treeRbm1.treeRbm1;

% -----------------------------------------------------------------------------------
% Preprocess training data
% -----------------------------------------------------------------------------------

normalize      = @(m) m/max(max(m));
pBw    		   = @(m) normalize(double(imsharpen(rgb2gray(m), 'radius',0.5,'amount',3)));
trees          = Pr.mapCell(pBw, trees);

tree1  		   = trees{1};

% -----------------------------------------------------------------------------------
% Transform image by first layer
% -----------------------------------------------------------------------------------


spacing        = treeRbm1.Config.spacing;

treeh 		   = tohidden(treeRbm1.Config,treeRbm1.W,treeRbm1.B,tree1);

% resize treeh so it's divisible by 2
treeh		   = treeh(1:98,1:616,:);

[a, b]         = pool    (treeRbm1.Config,treeh);
[a_,b_]        = pool_o  (treeh,treeRbm1.Config.spacing);


% conclusion: pool is ok, although in some cases flips between all 0 vs all 1
for k = 1:18
	figure; colormap('gray'); imagesc(a(:,:,k));
	figure; colormap('gray'); imagesc(a_(:,:,k));
end





























