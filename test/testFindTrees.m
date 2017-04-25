% test the find trees algorithm
clear;
clc;
dbstop if error;

if(ispc)
    outFolder = 'C:\\Documents\\Kiran\\ownCloud\\PhD\sim_results\\bn\\kdd1999';
elseif(ismac)
    outFolder = '/Users/kiran/ownCloud/PhD/sim_results/bn/kdd1999';
elseif(isunix)
    outFolder = 1;
end

dag_file = fullfile(outFolder, 'kddcup.learned.dag.mat');
load(dag_file);
hcbnObj.setVerbose(1);

dag = hcbnObj.dag;
bg = digraph(dag, hcbnObj.nodeNames);
plot(bg);

[trees, subgraphs] = findTrees(hcbnObj.dag);