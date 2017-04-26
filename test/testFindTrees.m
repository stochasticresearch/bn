%% test the find trees algorithm
clear;
clc;
dbstop if error;

if(ispc)
    outFolder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\bn\\kdd1999';
elseif(ismac)
    outFolder = '/Users/kiran/ownCloud/PhD/sim_results/bn/kdd1999';
elseif(isunix)
    outFolder = 1;
end

dag_file = fullfile(outFolder, 'kddcup.learned.dag.mat');
load(dag_file);
hcbnObj.setVerbose(1);

trees = findTrees(hcbnObj.dag_topoSorted);

%% test the find paths

clear;
clc;
dbstop if error;

if(ispc)
    outFolder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\bn\\kdd1999';
elseif(ismac)
    outFolder = '/Users/kiran/ownCloud/PhD/sim_results/bn/kdd1999';
elseif(isunix)
    outFolder = 1;
end

dag_file = fullfile(outFolder, 'kddcup.learned.dag.mat');
load(dag_file);
hcbnObj.setVerbose(1);

g = digraph(hcbnObj.dag_topoSorted, hcbnObj.nodeNames);
plot(g);

% test case 1 (all in tree-1)
requestedNodesIdxs = [3,6];
givenNodesIdxs = [];
paths = findPaths(hcbnObj.dag, requestedNodesIdxs, givenNodesIdxs);
pathsExpected = {[3,4,5,6]};
test1 = isequal(paths,pathsExpected);

requestedNodesIdxs = [3];
givenNodesIdxs = [6];
paths = findPaths(hcbnObj.dag, requestedNodesIdxs, givenNodesIdxs);
pathsExpected = {[3,4,5,6]};
test11 = isequal(paths,pathsExpected);

% test case 2 (all in tree-2)
requestedNodesIdxs = [11,16];
givenNodesIdxs = [];
paths = findPaths(hcbnObj.dag, requestedNodesIdxs, givenNodesIdxs);
pathsExpected = {[11,12,13,14,15,16]};
test2 = isequal(paths,pathsExpected);

% test case 3 (in tree-1 and tree-2)
requestedNodesIdxs = [3,6];
givenNodesIdxs = [11,16];
paths = findPaths(hcbnObj.dag, requestedNodesIdxs, givenNodesIdxs);
pathsExpected = {[3,4,5,6],[11,12,13,14,15,16]};
test3 = isequal(paths,pathsExpected);

% print results of testing
fprintf('Test_1 --> %d\n', test1);
fprintf('Test_2 --> %d\n', test2);
fprintf('Test_3 --> %d\n', test3);
fprintf('Test_4 --> %d\n', test11);