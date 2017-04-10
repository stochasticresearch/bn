%% Test the tree based inference method
clear;
clc;

M_train = 500;
M_test = 500;

% generate a simple DAG w/ 4 nodes, 2 discrete, 2 continuous
nodeNames = {'A','B','C','D'};
discreteNodes = {'C','D'};
dag = zeros(4,4);
dag(1,3) = 1;   % A --> C
dag(3,2) = 1;   % C --> B
dag(2,4) = 1;   % B --> D

K = 25;

% generate the data

% copula for A->C
rho1 = 0.8;
U_C1 = copularnd('Gaussian', 0.8, M_train+M_test);
uu = U_C1(:,1);
alpha_c2 = 3; p = rand(M_train+M_test,1);
% copula for C-->B
U_C2_2 = -log((exp(-alpha_c2.*uu).*(1-p)./p + exp(-alpha_c2))./(1 + exp(-alpha_c2.*uu).*(1-p)./p))./alpha_c2;

uu = U_C2_2;
alpha_c3 = 3; p = rand(M_train+M_test,1);
% copula for B-->D
U_C3_2 = -log((exp(-alpha_c3.*uu).*(1-p)./p + exp(-alpha_c3))./(1 + exp(-alpha_c3.*uu).*(1-p)./p))./alpha_c3;

% combine all into one large matrix of dependencies
U = [U_C1 U_C2_2 U_C3_2];

% convert pseudo-observations to data
b_dist_probs = [0.2 0.3 0.4 0.1];
d_dist_probs = [0.1 0.2 0.2 0.5];
b_dist = makedist('Multinomial','Probabilities',b_dist_probs);
d_dist = makedist('Multinomial','Probabilities',d_dist_probs);
X = zeros(size(U));
X(:,1) = norminv(U(:,1), 3, 5);
X(:,2) = b_dist.icdf(U(:,3));
X(:,3) = norminv(U(:,2), -2, 2);
X(:,4) = d_dist.icdf(U(:,4));

hcbnObj = hcbn_k1(X,nodeNames,discreteNodes,K,dag);
save('/tmp/test_tree_inference.mat');

%% Test the computations of pairwise-joints
load('/tmp/test_tree_inference.mat');

a_idx = 1; b_idx = 2; c_idx = 3; d_idx = 4;

% Get all the probabilities 
[ac_joint_modelselect,ac_modelselect_xy] = hcbnObj.computePairwiseJoint(c_idx, a_idx);
[cb_joint_modelselect,cb_modelselect_xy] = hcbnObj.computePairwiseJoint(b_idx, c_idx);
[bd_joint_modelselect,bd_modelselect_xy] = hcbnObj.computePairwiseJoint(d_idx, b_idx);

% change the copula family objects to known copula dependencies to compare
% the error
hcbnObjCpy = copy(hcbnObj);
u = linspace(0,1,hcbnObjCpy.K); [U,V] = ndgrid(u,u);

ac_copula_pdf = reshape(copulapdf('Gaussian',[U(:),V(:)], rho1),hcbnObj.K,hcbnObj.K);
hcbnObjCpy.copulaFamilies{c_idx}.c_model_name = 'Gaussian';
hcbnObjCpy.copulaFamilies{c_idx}.c_model_params = rho1;
hcbnObjCpy.copulaFamilies{c_idx}.C_discrete_integrate = []; % doesn't matter since both A/C are continuous

cb_copula_pdf = reshape(copulapdf('Frank',[U(:),V(:)], alpha_c2),hcbnObj.K,hcbnObj.K);
hcbnObjCpy.copulaFamilies{b_idx}.c_model_name = 'Frank';
hcbnObjCpy.copulaFamilies{b_idx}.c_model_params = alpha_c2;
cb_discrete_integrate = cumtrapz(u,cb_copula_pdf,2);
hcbnObjCpy.copulaFamilies{b_idx}.C_discrete_integrate = cb_discrete_integrate;

bd_copula_pdf = reshape(copulapdf('Frank',[U(:),V(:)], alpha_c3),hcbnObj.K,hcbnObj.K);
hcbnObjCpy.copulaFamilies{d_idx}.c_model_name = 'Frank';
hcbnObjCpy.copulaFamilies{d_idx}.c_model_params = alpha_c2;
% both dimensions are discrete so we integrate both
bd_discrete_integrate = cumtrapz(u,cumtrapz(u,bd_copula_pdf,1),2);
hcbnObjCpy.copulaFamilies{d_idx}.C_discrete_integrate = bd_discrete_integrate;

[ac_joint_modelknown,ac_known_xy] = hcbnObjCpy.computePairwiseJoint(c_idx, a_idx);
[cb_joint_modelknown,cb_known_xy] = hcbnObjCpy.computePairwiseJoint(b_idx, c_idx);
[bd_joint_modelknown,bd_known_xy] = hcbnObjCpy.computePairwiseJoint(d_idx, b_idx);

% now generate the reference -- and compute the error

%% Test the inference computations