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
dataFolder = '/Users/kiran/Documents/data/wine_quality';
train_dataFile = fullfile(dataFolder, 'winequality-red.csv');

% ensure dataset is all numeric, and convert categorical data to numeric
x_train = importdata(train_dataFile,';');

% learn structure
discreteNodes = [];
K = 100;
verboseFlag = 1;

hcbnObj = hcbn_k1(x_train.data, x_train.colheaders, [], K, 'learn', verboseFlag);

% now -- compute the likelihood of the model, given the data
totalLL = hcbnObj.dataLogLikelihood(x_train.data);
fprintf('Total LL=%0.02f\n', totalLL);

%% KDD-1999 Learning
clear;
clc;
dbstop if error;

% Read in a data-set
dataFolder = '/Users/kiran/Documents/data/kdd1999';
train_dataFile = fullfile(dataFolder, 'kddcup.preprocess.data.train');
test_dataFile = fullfile(dataFolder, 'kddcup.preprocess.data.test');
dag_file = fullfile(dataFolder, 'kddcup.learned.dag');

% ensure dataset is all numeric, and convert categorical data to numeric
x_train = importdata(train_dataFile,',');
x_test  = importdata(test_dataFile,',');

% the first column of x.data is just the pandas index, we ignore it
train_data = x_train.data(:,2:end);
test_data  = x_test.data(:,2:end);
discreteIdxs = [1, 2, 3, 6, 11, 20, 21, 41]+1; % +1 to go from python indexing (0 based)
                                               % to Matlab indexing (1 based)
                                               
% find features we should exclude b/c there is not any variation in the
% data
exclude_features = [];
for ii=1:size(train_data,2)
    if( (len(unique(train_data(:,ii)))<2) || (len(unique(test_data(:,ii)))<2) )
        exclude_features = [exclude_features ii];
    end
end
                                               
colheaders = x_train.colheaders(2:end);
K = 100;
verboseFlag = 1;

train_data(:,exclude_features) = [];
test_data(:,exclude_features) = [];
colheaders(exclude_features) = [];

hcbnObj = hcbn_k1(train_data, colheaders, [], K, 'learn', verboseFlag);

% save the learned DAG
save(dag_file);

%% KDD-1999 Testing
dataFolder = '/Users/kiran/Documents/data/kdd1999';
dag_file = fullfile(dataFolder, 'kddcup.learned.dag');

load(dag_file);

% now -- compute the likelihood of the model, given the data
totalLL = hcbnObj.dataLogLikelihood(test_data);
fprintf('Total LL=%0.02f\n', totalLL);