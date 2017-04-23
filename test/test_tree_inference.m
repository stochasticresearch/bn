%% Test the tree based inference method
clear;
clc;
dbstop if error;

M_train = 500;
M_test = 500;

% generate a simple DAG w/ 4 nodes, 2 discrete, 2 continuous
nodeNames = {'A','B','C','D'};
discreteNodes = {'B','D'};
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
b_dist_obj = makedist('Multinomial','Probabilities',b_dist_probs);
d_dist_obj = makedist('Multinomial','Probabilities',d_dist_probs);
a_dist_obj = makedist('Normal','mu',3,'sigma',5);
c_dist_obj = makedist('Normal','mu',-2,'sigma',2);

X = zeros(size(U));
X(:,1) = norminv(U(:,1), 10, 2);
X(:,2) = b_dist_obj.icdf(U(:,3));
X(:,3) = norminv(U(:,2), -10, 2);
X(:,4) = d_dist_obj.icdf(U(:,4));

verboseFlag = 1;
hcbnObj = hcbn_k1(X,nodeNames,discreteNodes,K,verboseFlag,dag);

if ispc
    saveDir = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\bn';
elseif ismac
    % do something else
    saveDir = '/Users/Kiran/ownCloud/PhD/sim_results/bn';
else
    % do a third thing
    saveDir = '/home/kiran/ownCloud/PhD/sim_results/bn';
end
fileName = 'test_tree_inference.mat';
save(fullfile(saveDir,fileName));

%% Test the computations of pairwise-joints
clear;
clc;
dbstop if error;

if ispc
    saveDir = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\bn';
elseif ismac
    % do something else
    saveDir = '/Users/Kiran/ownCloud/PhD/sim_results/bn';
else
    % do a third thing
    saveDir = '/home/kiran/ownCloud/PhD/sim_results/bn';
end
fileName = 'test_tree_inference.mat';
load(fullfile(saveDir,fileName));

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
cb_discrete_integrate = cumtrapz(u,cb_copula_pdf,1);    % dim-1 = B (discrete)
hcbnObjCpy.copulaFamilies{b_idx}.C_discrete_integrate = cb_discrete_integrate;

bd_copula_pdf = reshape(copulapdf('Frank',[U(:),V(:)], alpha_c3),hcbnObj.K,hcbnObj.K);
hcbnObjCpy.copulaFamilies{d_idx}.c_model_name = 'Frank';
hcbnObjCpy.copulaFamilies{d_idx}.c_model_params = alpha_c2;
% both dimensions are discrete so we integrate both
bd_discrete_integrate = cumtrapz(u,cumtrapz(u,bd_copula_pdf,1),2);  % dim-1 & dim-2 are discrete here
hcbnObjCpy.copulaFamilies{d_idx}.C_discrete_integrate = bd_discrete_integrate;

[ac_joint_modelknown,ac_known_xy] = hcbnObjCpy.computePairwiseJoint(c_idx, a_idx);
[cb_joint_modelknown,cb_known_xy] = hcbnObjCpy.computePairwiseJoint(b_idx, c_idx);
[bd_joint_modelknown,bd_known_xy] = hcbnObjCpy.computePairwiseJoint(d_idx, b_idx);

% ensure the domains of calculation are the same (they should be)
if(~isequal(ac_modelselect_xy,ac_known_xy) || ...
   ~isequal(cb_modelselect_xy,cb_known_xy) || ...
   ~isequal(bd_modelselect_xy,bd_known_xy))
    error('Domains of calculation seem different!');
end
if(any(isnan(ac_joint_modelselect(:))) || any( ac_joint_modelselect(:) < 0) )
    error('ac_joint_modelselect NaN or Negative!');
end
if(any(isnan(ac_joint_modelknown(:))) || any( ac_joint_modelknown(:) < 0) )
    error('ac_joint_modelknown NaN or Negative!');
end
if(any(isnan(cb_joint_modelselect(:))) || any( cb_joint_modelselect(:) < 0) )
    error('cb_joint_modelselect NaN or Negative!');
end
if(any(isnan(cb_joint_modelknown(:))) || any( cb_joint_modelknown(:) < 0) )
    error('cb_joint_modelknown NaN or Negative!');
end
if(any(isnan(bd_joint_modelselect(:))) || any( bd_joint_modelselect(:) < 0) )
    error('bd_joint_modelselect NaN or Negative!');
end
if(any(isnan(bd_joint_modelknown(:))) || any( bd_joint_modelknown(:) < 0) )
    error('bd_joint_modelknown NaN or Negative!');
end

% now generate the reference -- and compute the error
ac_joint_modelselect_err = zeros(size(ac_joint_modelselect));
ac_joint_modelknown_err = zeros(size(ac_joint_modelknown));
for ii=1:numel(ac_modelselect_xy)   % use linear indexing
    xy = ac_modelselect_xy{ii};
    
    model_select_prob = ac_joint_modelselect(ii);
    model_forced_prob = ac_joint_modelknown(ii);
    
    % get the "true" model probability
    % h(x,y) = c(F(x),G(y))*f(x)*g(y)
    F_c = a_dist_obj.cdf(xy(1));    % child is stored as x-coordinate
    F_a = c_dist_obj.cdf(xy(2));    % parent is stored as y-coordinate
    f_c = a_dist_obj.pdf(xy(1));
    f_a = c_dist_obj.pdf(xy(2));
    true_prob = copulapdf('Gaussian',[F_c F_a],rho1)*f_c*f_a;
    
    % compute error & store
    ac_joint_modelselect_err(ii) = (true_prob-model_select_prob).^2;
    ac_joint_modelknown_err(ii) = (true_prob-model_forced_prob).^2;
end

cb_joint_modelselect_err = zeros(size(cb_joint_modelselect));
cb_joint_modelknown_err = zeros(size(cb_joint_modelknown));
for ii=1:numel(cb_modelselect_xy)   % use linear indexing
    xy = cb_modelselect_xy{ii};
    
    model_select_prob = cb_joint_modelselect(ii);
    model_forced_prob = cb_joint_modelknown(ii);
    
    % get the "true" model probability
    % h(x,y) = [C(F(x),G(y)) - C(F(x),G(y-))]* f(x)
    F_c = c_dist_obj.cdf(xy(2));    % parent is stored as y-coordinate
    f_c = c_dist_obj.pdf(xy(2));
    % B is a discrete random variable, so we handle it differently
    b_val = xy(1);
    F_b = b_dist_obj.cdf(b_val);
    F_b_minus = b_dist_obj.cdf(b_val-1);
    
    true_prob = (copulacdf('Frank',[F_b F_c],alpha_c2) - ...
                 copulacdf('Frank',[F_b_minus F_c],alpha_c2))*f_c;
    
    % compute error & store
    cb_joint_modelselect_err(ii) = (true_prob-model_select_prob).^2;
    cb_joint_modelknown_err(ii) = (true_prob-model_forced_prob).^2;
end

bd_joint_modelselect_err = zeros(size(bd_joint_modelselect));
bd_joint_modelknown_err = zeros(size(bd_joint_modelknown));
for ii=1:numel(bd_modelselect_xy)   % use linear indexing
    xy = bd_modelselect_xy{ii};
    
    model_select_prob = bd_joint_modelselect(ii);
    model_forced_prob = bd_joint_modelknown(ii);
    
    % get the "true" model probability
    % h(x,y) = [C(F(x),G(y)) - C(F(x),G(y-)) - C(F(x-),G(y) + C(F(x-)+G(y-))]
    b_val = xy(2);
    F_b = b_dist_obj.cdf(b_val);    % parent is stored as y-coordinate
    F_b_minus = b_dist_obj.pdf(b_val-1);
    
    d_val = xy(1);
    F_d = d_dist_obj.cdf(d_val);
    F_d_minus = d_dist_obj.cdf(d_val-1);
    
    true_prob = (copulacdf('Frank',[F_d F_b],alpha_c3) - ...
                 copulacdf('Frank',[F_d_minus F_b],alpha_c3) - ...
                 copulacdf('Frank',[F_d F_b_minus],alpha_c3) + ...
                 copulacdf('Frank',[F_d_minus F_b_minus],alpha_c3) );
    
    % compute error & store
    bd_joint_modelselect_err(ii) = (true_prob-model_select_prob).^2;
    bd_joint_modelknown_err(ii) = (true_prob-model_forced_prob).^2;
end

% plot the errors
subplot(2,3,1); surf(ac_joint_modelselect_err); title('A->C (ModelSelect) [C]'); zlabel('MSE'); grid on;
subplot(2,3,4); surf(ac_joint_modelknown_err);  title('A->C (ModelKnown) [C]'); zlabel('MSE'); grid on;

subplot(2,3,2); surf(cb_joint_modelselect_err); title('C->B (ModelSelect) [H]'); zlabel('MSE'); grid on;
subplot(2,3,5); surf(cb_joint_modelknown_err);  title('C->B (ModelKnown) [H]'); zlabel('MSE'); grid on;

subplot(2,3,3); surf(bd_joint_modelselect_err); title('B->D (ModelSelect) [D]'); zlabel('MSE'); grid on;
subplot(2,3,6); surf(bd_joint_modelknown_err);  title('B->D (ModelKnown) [D]'); zlabel('MSE'); grid on;

%% Test the computations of the full joint probability
clear;
clc;
dbstop if error;

if ispc
    saveDir = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\bn';
elseif ismac
    % do something else
    saveDir = '/Users/Kiran/ownCloud/PhD/sim_results/bn';
else
    % do a third thing
    saveDir = '/home/kiran/ownCloud/PhD/sim_results/bn';
end
fileName = 'test_tree_inference.mat';
load(fullfile(saveDir,fileName));

a_idx = 1; b_idx = 2; c_idx = 3; d_idx = 4;

givenNodesIdx = [];
givenNodesValues = [];
normalizeProb = 0;

% Get all the pairwise
[ac_joint_modelselect,ac_modelselect_xy] = hcbnObj.computePairwiseJoint(c_idx, a_idx);
[cb_joint_modelselect,cb_modelselect_xy] = hcbnObj.computePairwiseJoint(b_idx, c_idx);
[bd_joint_modelselect,bd_modelselect_xy] = hcbnObj.computePairwiseJoint(d_idx, b_idx);

% marginalize from the joint and see if they equal the pairwise
requestedNodesIdx = [a_idx c_idx];
[ac_joint_inference,ac_inference_xy] = hcbnObj.inference(requestedNodesIdx,givenNodesIdx,givenNodesValues,normalizeProb);
ac_joint_inference_normalized = hcbnObj.inference(requestedNodesIdx,givenNodesIdx,givenNodesValues,1);
requestedNodesIdx = [c_idx b_idx];
[cb_joint_inference,cb_inference_xy] = hcbnObj.inference(requestedNodesIdx,givenNodesIdx,givenNodesValues,normalizeProb);
cb_joint_inference_normalized = hcbnObj.inference(requestedNodesIdx,givenNodesIdx,givenNodesValues,1);
requestedNodesIdx = [b_idx d_idx];
[bd_joint_inference,bd_inference_xy] = hcbnObj.inference(requestedNodesIdx, givenNodesIdx,givenNodesValues,normalizeProb);
bd_joint_inference_normalized = hcbnObj.inference(requestedNodesIdx, givenNodesIdx,givenNodesValues,1);

ac_joint_modelselectT = ac_joint_modelselect'; ac_modelselect_xyT = ac_modelselect_xy';
ac_inference_xyT = cellfun(@fliplr, ac_inference_xy,'UniformOutput',0);
ac_domain_match = isequal(ac_modelselect_xyT, ac_inference_xyT);
ac_prob_match = isequal(ac_joint_modelselectT, ac_joint_inference);
fprintf('AC ** Domain_Match=%d | Prob_Match=%d\n', ...
        ac_domain_match, ac_prob_match);

cb_joint_modelselectT = cb_joint_modelselect'; cb_modelselect_xyT = cb_modelselect_xy';
cb_inference_xyT = cellfun(@fliplr, cb_inference_xy,'UniformOutput',0);
cb_domain_match = isequal(cb_modelselect_xyT, cb_inference_xyT);
cb_prob_match = isequal(cb_joint_modelselectT, cb_joint_inference);
fprintf('CB ** Domain_Match=%d | Prob_Match=%d\n', ...
        cb_domain_match, cb_prob_match);
    
bd_joint_modelselectT = bd_joint_modelselect'; bd_modelselect_xyT = bd_modelselect_xy';
bd_inference_xyT = cellfun(@fliplr, bd_inference_xy,'UniformOutput',0);
bd_domain_match = isequal(bd_modelselect_xyT, bd_inference_xyT);
bd_prob_match = isequal(bd_joint_modelselectT, bd_joint_inference);
fprintf('BD ** Domain_Match=%d | Prob_Match=%d\n', ...
        bd_domain_match, bd_prob_match);

ac_marginalization_err = (ac_joint_inference_normalized-ac_joint_modelselect').^2;
cb_marginalization_err = (cb_joint_inference_normalized-cb_joint_modelselect').^2;
bd_marginalization_err = (bd_joint_inference_normalized-bd_joint_modelselect').^2;

subplot(1,3,1); surf(ac_marginalization_err); title('A->C [C]'); zlabel('MSE'); grid on;
subplot(1,3,2); surf(cb_marginalization_err); title('C->B [C]'); zlabel('MSE'); grid on;
subplot(1,3,3); surf(bd_marginalization_err); title('B->D [C]'); zlabel('MSE'); grid on;

%% Test the marginalization of nuisance variables
clear;
clc;
dbstop if error;

if ispc
    saveDir = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\bn';
elseif ismac
    % do something else
    saveDir = '/Users/Kiran/ownCloud/PhD/sim_results/bn';
else
    % do a third thing
    saveDir = '/home/kiran/ownCloud/PhD/sim_results/bn';
end
fileName = 'test_tree_inference.mat';
load(fullfile(saveDir,fileName));

a_idx = 1; b_idx = 2; c_idx = 3; d_idx = 4;

requestedNodesIdx = [a_idx b_idx];
givenNodesIdx = [];     % force the inference engine to integrate out the C node
[ab_joint_inference,ab_inference_xy] = hcbnObj.inference(requestedNodesIdx,givenNodesIdx);
% how to verify ab is actually correct? there is no reference distribution
% to compare against

% ensure we are integrating out the correct dimension of the domain by
% checking the x and y domains, and seeing if they match what we expect
x_coord_list = zeros(1,numel(ab_inference_xy));
for ii=1:numel(ab_inference_xy)
    z = ab_inference_xy{ii};
    x_coord_list(ii) = z(1);
end
x_domain = unique(x_coord_list);        % should be all positive w/ a center around 10
                                        % based on the way we generated the data

requestedNodesIdx = [c_idx d_idx];
givenNodesIdx = [];
[cd_joint_inference,cd_inference_xy] = hcbnObj.inference(requestedNodesIdx,givenNodesIdx);

% ************** TODO ****************
% How to verify the inference?

%% Test how we deal w/ "given" variables
clear;
clc;
dbstop if error;

if ispc
    saveDir = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\bn';
elseif ismac
    % do something else
    saveDir = '/Users/Kiran/ownCloud/PhD/sim_results/bn';
else
    % do a third thing
    saveDir = '/home/kiran/ownCloud/PhD/sim_results/bn';
end
fileName = 'test_tree_inference.mat';
load(fullfile(saveDir,fileName));

a_idx = 1; b_idx = 2; c_idx = 3; d_idx = 4;

requestedNodesIdx = [a_idx b_idx];
givenNodesIdx = [c_idx];
givenNodesValues = [2];
[ab_joint_inference,ab_inference_xy] = hcbnObj.inference(requestedNodesIdx,givenNodesIdx,givenNodesValues);
