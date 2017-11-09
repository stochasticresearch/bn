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
%% Process the Red Wine-Quality Dataset
clear;
clc;
dbstop if error;

% Read in a data-set
dataFolder = '/data/winequality';
dataFile = fullfile(dataFolder, 'winequality-red.csv');
dataFile = fullfile(dataFolder, 'winequality-white.csv');

% ensure dataset is all numeric, and convert categorical data to numeric
x = importdata(dataFile,';');

rng(12345);
trainRatio = 0.7; valRatio = 0.00; testRatio = 0.3;
[trainInd,valInd,testInd] = dividerand(size(x.data,1),trainRatio,valRatio,testRatio);

x_train = x.data(trainInd,:);
x_test  = x.data(testInd,:);

% learn structure
discreteNodes = [];
K = 100;
verboseFlag = 1;

srhoProxy = 'srho';
cimProxy  = 'cim';

hcbnSrhoObj = hcbn_k1(x_train, x.colheaders, [], K, verboseFlag, 'learn', srhoProxy);
srhoLL = hcbnSrhoObj.copulaLogLikelihood(x_test);

hcbnCimObj = hcbn_k1(x_train, x.colheaders, [], K, verboseFlag, 'learn', cimProxy);
cimLL = hcbnCimObj.copulaLogLikelihood(x_test);

fprintf('LL[sRho]=%0.02f LL[CIM]=%0.02f\n', ...
    srhoLL, cimLL);

%% Process the White Wine-Quality Dataset
clear;
clc;
dbstop if error;

% Read in a data-set
dataFolder = '/data/winequality';
dataFile = fullfile(dataFolder, 'winequality-white.csv');

% ensure dataset is all numeric, and convert categorical data to numeric
x = importdata(dataFile,';');

rng(12345);
trainRatio = 0.7; valRatio = 0.00; testRatio = 0.3;
[trainInd,valInd,testInd] = dividerand(size(x.data,1),trainRatio,valRatio,testRatio);

x_train = x.data(trainInd,:);
x_test  = x.data(testInd,:);

% learn structure
discreteNodes = [];
K = 100;
verboseFlag = 1;

srhoProxy = 'srho';
cimProxy  = 'cim';

hcbnSrhoObj = hcbn_k1(x_train, x.colheaders, [], K, verboseFlag, 'learn', srhoProxy);
srhoLL = hcbnSrhoObj.copulaLogLikelihood(x_test);

hcbnCimObj = hcbn_k1(x_train, x.colheaders, [], K, verboseFlag, 'learn', cimProxy);
cimLL = hcbnCimObj.copulaLogLikelihood(x_test);

fprintf('LL[sRho]=%0.02f LL[CIM]=%0.02f\n', ...
    srhoLL, cimLL);