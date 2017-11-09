%**************************************************************************
%* 
%* Copyright (C) 2017  Kiran Karra <kiran.karra@gmail.com>
%*
%* This program is free software: you can redistribute it and/or modify
%* it under the terms of the GNU General Public License as published by
%* the Free Software Foundation, either version 3 of the License, or
%* (at your option) any later version.
%*
%* This program is distributed in the hope that it will be useful,
%* but WITHOUT ANY WARRANTY; without even the implied warranty of
%* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%* GNU General Public License for more details.
%*
%* You should have received a copy of the GNU General Public License
%* along with this program.  If not, see <http://www.gnu.org/licenses/>.
%* 
%**************************************************************************
%% Process the Wine-Quality Dataset
clear;
clc;
dbstop if error;

% Read in a data-set
dataFolder = '/data/winequality';
train_dataFile = fullfile(dataFolder, 'winequality-red.csv');

% ensure dataset is all numeric, and convert categorical data to numeric
x_train = importdata(train_dataFile,';');

% learn structure
discreteNodes = [];
K = 100;
verboseFlag = 1;

hcbnObj = hcbn_k1(x_train.data, x_train.colheaders, [], K, verboseFlag, 'learn');

% now -- compute the likelihood of the model, given the data
totalLL = hcbnObj.dataLogLikelihood(x_train.data);
fprintf('Total LL=%0.02f\n', totalLL);

%% KDD-1999 Learning
clear;
clc;
dbstop if error;

% Read in a data-set
if(ispc)
    dataFolder = 'C:\\Users\\Kiran\\Documents\\data\\kdd1999';
    outFolder = 'C:\\Users\\Kiran\\ownCloud\\PhD\sim_results\\bn\\kdd1999';
elseif(ismac)
    dataFolder = '/Users/kiran/Documents/data/kdd1999';
    outFolder = '/Users/kiran/ownCloud/PhD/sim_results/bn/kdd1999';
elseif(isunix)
    dataFolder = 1;
end

train_dataFile = fullfile(dataFolder, 'kddcup.preprocess.data.train');
test_dataFile = fullfile(dataFolder, 'kddcup.preprocess.data.test');
discreteIdxsFile = fullfile(dataFolder, 'kddcup.preprocess.data.discrete');
dag_file = fullfile(outFolder, 'kddcup.learned.dag.mat');

% ensure dataset is all numeric, and convert categorical data to numeric
x_train = importdata(train_dataFile,',');
x_test  = importdata(test_dataFile,',');

% the first column of x.data is just the pandas index, we ignore it
train_data = x_train.data(:,2:end);
test_data  = x_test.data(:,2:end);
% get the discrete indices
discreteIdxs = importdata(discreteIdxsFile)' + 1;   % +1 to reindex w/ python
                                               
colheaders = x_train.colheaders(2:end);
K = 25;
verboseFlag = 1;

discreteNodesNames = cell(1,length(discreteIdxs));
for ii=1:length(discreteIdxs)
    discreteNodesNames{ii} = colheaders{discreteIdxs(ii)};
end

hcbnObj = hcbn_k1(train_data, colheaders, discreteNodesNames, K, verboseFlag, 'learn');

% save the learned DAG
save(dag_file);

%% KDD-1999 Testing
clear;
clc;
close all;
dbstop if error;

% Read in a data-set
if(ispc)
    outFolder = 'C:\\Users\\Kiran\\ownCloud\\PhD\sim_results\\bn\\kdd1999';
elseif(ismac)
    outFolder = '/Users/kiran/ownCloud/PhD/sim_results/bn/kdd1999';
elseif(isunix)
    outFolder = 1;
end

dag_file = fullfile(outFolder, 'kddcup.learned.dag.mat');
load(dag_file);
hcbnObj.setVerbose(1);
normalizeFlag = 1;

% % plot the grpah
bg = biograph(hcbnObj.dag_topoSorted,hcbnObj.nodeNames);
dolayout(bg);
view(bg)

% % % now -- compute the likelihood of the model, given the data
% % totalLL = hcbnObj.dataLogLikelihood(test_data);
% % fprintf('Total LL=%0.02f\n', totalLL);

% test out the application of the topological ordering 
%requestedNodes = {colnames{3}, colnames{4}}
%givenNodes = {colnames{6}, colnames{8}}

% within 1 tree
% requestedNodes = [3,4];
% givenNodes = [];
% givenValues = [1];
% [inference1,domain1] = hcbnObj.inference(requestedNodes, givenNodes, givenValues, normalizeFlag);
% 
% % across 2 trees
% requestedNodes = [3,12];
% givenNodes = [];
% [inference2,domain2] = hcbnObj.inference(requestedNodes, givenNodes, givenValues, normalizeFlag);

requestedNodes = {'class_label','count'};
givenNodes = []; givenValues = [];
[inference1,domain1,names1] = hcbnObj.inference(requestedNodes, givenNodes, givenValues, normalizeFlag);
hcbn_heatmap(inference1,domain1,names1);

requestedNodes = {'count','srv_count'};
givenNodes = []; givenValues = [];
[inference2,domain2,names2] = hcbnObj.inference(requestedNodes, givenNodes, givenValues, normalizeFlag);
hcbn_heatmap(inference2,domain2,names2);

requestedNodes = {'srv_count','dst_host_same_src_port_rate'};
givenNodes = []; givenValues = [];
[inference,domain,names] = hcbnObj.inference(requestedNodes, givenNodes, givenValues, normalizeFlag);
hcbn_heatmap(inference,domain,names);

requestedNodes = {'dst_host_same_src_port_rate','src_bytes'};
givenNodes = []; givenValues = [];
[inference,domain,names] = hcbnObj.inference(requestedNodes, givenNodes, givenValues, normalizeFlag);
hcbn_heatmap(inference,domain,names);

requestedNodes = {'src_bytes','dst_host_diff_srv_rate'};
givenNodes = []; givenValues = [];
[inference,domain,names] = hcbnObj.inference(requestedNodes, givenNodes, givenValues, normalizeFlag);
hcbn_heatmap(inference,domain,names);

requestedNodes = {'dst_host_diff_srv_rate','dst_host_same_srv_rate'};
givenNodes = []; givenValues = [];
[inference,domain,names] = hcbnObj.inference(requestedNodes, givenNodes, givenValues, normalizeFlag);
hcbn_heatmap(inference,domain,names);

requestedNodes = {'dst_host_same_srv_rate','dst_host_srv_count'};
givenNodes = []; givenValues = [];
[inference,domain,names] = hcbnObj.inference(requestedNodes, givenNodes, givenValues, normalizeFlag);
hcbn_heatmap(inference,domain,names);

requestedNodes = {'dst_host_srv_count','diff_srv_rate'};
givenNodes = []; givenValues = [];
[inference,domain,names] = hcbnObj.inference(requestedNodes, givenNodes, givenValues, normalizeFlag);
hcbn_heatmap(inference,domain,names);
