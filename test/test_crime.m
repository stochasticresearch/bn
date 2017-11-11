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
%% Process the Crime Dataset
clear;
clc;
dbstop if error;

% Read in a data-set
% Read in a data-set
if(ispc)
    dataFolder = 'C:\\Users\\Kiran\\ownCloud\\PhD\sim_results\\crime\\';
elseif(ismac)
    dataFolder = '/Users/kiran/ownCloud/PhD/sim_results/crime';
elseif(isunix)
    dataFolder = '/home/kiran/ownCloud/PhD/sim_results/crime';
end
dataFile = fullfile(dataFolder, 'communities.mat');

% ensure dataset is all numeric, and convert categorical data to numeric
load(dataFile);

rng(12345);
trainRatio = 0.7; valRatio = 0.00; testRatio = 0.3;
[trainInd,valInd,testInd] = dividerand(size(X,1),trainRatio,valRatio,testRatio);

x_train = X(trainInd,:);
x_test  = X(testInd,:);

numFeatures = size(X,2);
colHeaders = cell(1,numFeatures);
for ii=1:numFeatures
    colHeaders{ii} = sprintf('F_%d',ii);
end

% learn structure
discreteNodes = [];
K = 100;
verboseFlag = 1;

srhoProxy = 'srho';
cimProxy  = 'cim';

%% run the srho proxy
hcbnSrhoObj = hcbn_k1(x_train, colHeaders, [], K, verboseFlag, 'learn', srhoProxy);
srhoLL = hcbnSrhoObj.copulaLogLikelihood(x_test);

save(fullfile(dataFolder,'crime_results_srho.mat'));

%% run the cim proxy
hcbnCimObj = hcbn_k1(x_train, colHeaders, [], K, verboseFlag, 'learn', cimProxy);
cimLL = hcbnCimObj.copulaLogLikelihood(x_test);

save(fullfile(dataFolder,'crime_results_cim.mat'));

%% Compare the LL's
load(fullfile(dataFolder,'crime_results_srho.mat'));
load('/home/kiran/ownCloud/PhD/sim_results/crime/crime_results_cim.mat');

fprintf('LL[sRho]=%0.02f LL[CIM]=%0.02f\n', srhoLL, cimLL);
