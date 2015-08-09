% -----------------------------------------------------------------------------------
% -----------------------------------------------------------------------------------
% Author : Xiao Ling
% Module : Week Four Part II - Use SVM to segment nature scenes
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
River = load('../Data/River1.mat');
River = River.River.Processed;

Tree  = load('../Data/Tree1.mat' );
Tree  = Tree.Tree.Processed;

% -----------------------------------------------------------------------------------
% Load RBM
% -----------------------------------------------------------------------------------

riverRbm1 = load('../Results/RiverRBM1_18_Aug2.mat' 		 );
riverRbm2 = load('../Results/RiverRBM2_40_Aug3.mat' 		 );
riverRbm3 = load('../Results/RiverRBM3_80_500Trial_Aug13.mat');

treeRbm1  = load('../Results/TreeRBM1_18_Aug3.mat'		 	 );
treeRbm2  = load('../Results/TreeRBM2_40_Aug3.mat'			 );
treeRbm3  = load('../Results/TreeRBM3_80_Aug7_500Iter.mat'   );

riverRbm1 = riverRbm1.riverRbm1;
riverRbm2 = riverRbm2.riverRbm2;
riverRbm3 = riverRbm3.riverRbm3;
treeRbm1  = treeRbm1.treeRbm1  ;
treeRbm2  = treeRbm2.treeRbm2  ;
treeRbm3  = treeRbm3.treeRbm3  ;

% -----------------------------------------------------------------------------------
% Reconstruct
% -----------------------------------------------------------------------------------

River_   = {};
riverErr = [];

for k = 1:length(River);
	
	h0  = River{k};
	h1  = RBM.run    (riverRbm1,h0 );
	h2  = RBM.run    (riverRbm2,h1 );
	h3  = RBM.run    (riverRbm3,h2 );

	h2_ = RBM.down   (riverRbm3,h3 );
	h1_ = RBM.down   (riverRbm2,h2_);
	h0_ = RBM.down   (riverRbm1,h1_);

	riverErr(k) = sum(sum(abs(h0-h0_)))
	River_{k}   = h0_;

	fprintf('--------------- Reconstructed Image -----------------------');
end


Tree_   = {};
treeErr = [];

for k = 1:length(Tree);
	
	h0  = Tree{k};
	h1  = RBM.run    (TreeRbm1,h0 );
	h2  = RBM.run    (TreeRbm2,h1 );
	h3  = RBM.run    (TreeRbm3,h2 );

	h2_ = RBM.down   (TreeRbm3,h3 );
	h1_ = RBM.down   (TreeRbm2,h2_);
	h0_ = RBM.down   (TreeRbm1,h1_);

	treeErr(k) = sum(sum(abs(h0-h0_)))
	Tree_{k}   = h0_;

	fprintf('--------------- Reconstructed Image -----------------------');
	
end

SegmentNature.River    = River;
SegmentNature.RiverErr = riverErr;

SegmentNature.Tree     = Tree;
SegmentNature.TreeErr  = treeErr;

save SegmentNature SegmentNature














