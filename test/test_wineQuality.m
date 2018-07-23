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

% ensure dataset is all numeric, and convert categorical data to numeric
x = importdata(dataFile,';');

trainRatio = 0.7; valRatio = 0.00; testRatio = 0.3;

% learn structure
discreteNodes = [];
K = 100;
verboseFlag = 0;

srhoProxy = 'srho';
tauProxy  = 'ktau';
tauklProxy = 'taukl';
cimProxy   = 'cim';

numMCSims = 10;

rng(12345);
dispstat('','init'); % One time only initialization
llVec = zeros(4,numMCSims);
for mcSimNum=1:numMCSims
    [trainInd,~,testInd] = dividerand(size(x.data,1),trainRatio,valRatio,testRatio);
    x_train = x.data(trainInd,:);
    x_test  = x.data(testInd,:);

    hcbnSrhoObj  = hcbn_k1(x_train, x.colheaders, [], K, verboseFlag, 'learn', srhoProxy);
    hcbnKTauObj  = hcbn_k1(x_train, x.colheaders, [], K, verboseFlag, 'learn', tauProxy);    
    hcbnTauKLObj = hcbn_k1(x_train, x.colheaders, [], K, verboseFlag, 'learn', tauklProxy);
    hcbnCIMObj = hcbn_k1(x_train, x.colheaders, [], K, 1, 'learn', cimProxy);
    
%     hcbnSrhoObj.dag_topoSorted
%     hcbnKTauObj.dag_topoSorted
%     hcbnTauKLObj.dag_topoSorted
    
    srhoLL  = hcbnSrhoObj.copulaLogLikelihood(x_test);
    kTauLL  = hcbnKTauObj.copulaLogLikelihood(x_test);
    tauklLL = hcbnTauKLObj.copulaLogLikelihood(x_test);
    cimLL   = hcbnCIMObj.copulaLogLikelihood(x_test);

    llVec(1,mcSimNum) = srhoLL;
    llVec(2,mcSimNum) = kTauLL;
    llVec(3,mcSimNum) = tauklLL;
    llVec(4,mcSimNum) = cimLL;
    
    dispstat(sprintf('{%d} LL[sRho]=%0.02f LL[tau]=%0.02f LL[tau_KL]=%0.02f LL[CIM]=%0.02f', ...
                      mcSimNum, srhoLL, kTauLL, tauklLL, cimLL),'keepthis','timestamp');
end


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
tauProxy  = 'ktau';
tauklProxy  = 'taukl';

hcbnSrhoObj = hcbn_k1(x_train, x.colheaders, [], K, verboseFlag, 'learn', srhoProxy);
srhoLL = hcbnSrhoObj.copulaLogLikelihood(x_test);

hcbnKTauObj = hcbn_k1(x_train, x.colheaders, [], K, verboseFlag, 'learn', tauProxy);
kTauLL = hcbnKTauObj.copulaLogLikelihood(x_test);

hcbnTauKLObj = hcbn_k1(x_train, x.colheaders, [], K, verboseFlag, 'learn', tauklProxy);
tauklLL = hcbnTauKLObj.copulaLogLikelihood(x_test);

fprintf('LL[sRho]=%0.02f LL[tau]=%0.02f LL[tau_KL]=%0.02f\n', ...
    srhoLL, kTauLL, tauklLL);
