%% Test the tree based inference method
clear;
clc;

M_train = 500;
M_test = 500;

% generate a simple DAG w/ 4 nodes, 2 discrete, 2 continuous
nodeNames = {'A','B','C','D'};
discreteNodes = {'B','C'};
dag = zeros(4,4);
dag(1,2) = 1;   % A --> B
dag(2,3) = 1;   % B --> C
dag(3,4) = 1;   % C --> D

K = 25;

% generate the data
rho1 = 0.8;
U_C1 = copularnd('Gaussian', 0.8, M_train+M_test);
uu = U_C1(:,1);
alpha_c2 = 3; p = rand(M_train+M_test,1);
U_C2_2 = -log((exp(-alpha_c2.*uu).*(1-p)./p + exp(-alpha_c2))./(1 + exp(-alpha_c2.*uu).*(1-p)./p))./alpha_c2;

uu = U_C2_2;
alpha_c3 = 3; p = rand(M_train+M_test,1);
U_C3_2 = -log((exp(-alpha_c3.*uu).*(1-p)./p + exp(-alpha_c3))./(1 + exp(-alpha_c3.*uu).*(1-p)./p))./alpha_c3;

% combine all into one large matrix of dependencies
U = [U_C1 U_C2_2 U_C3_2];

% convert pseudo-observations to data
b_dist = makedist('Multinomial','Probabilities',[0.2 0.3 0.4 0.1]);
c_dist = makedist('Multinomial','Probabilities',[0.1 0.2 0.2 0.5]);
X = zeros(size(U));
X(:,1) = norminv(U(:,1), 3, 5);
X(:,2) = b_dist.icdf(U(:,2));
X(:,3) = c_dist.icdf(U(:,3));
X(:,4) = norminv(U(:,4), -2, 2);

hcbnObj = hcbn_k1(X,nodeNames,discreteNodes,K,dag);

%% Test the computations of pairwise-joints

%% Test the inference computations