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
clear;
clc;
dbstop if error;

% Read in a data-set
dataFolder = '/Users/kiran/Documents/data/wine_quality';
redWineDataFile = fullfile(dataFolder, 'winequality-red.csv');

% ensure dataset is all numeric, and convert categorical data to numeric
x = importdata(redWineDataFile,';');

% learn structure
discreteNodes = [];
K = 100;
verboseFlag = 1;

hcbnObj = hcbn_k1(x.data, x.colheaders, [], K, 'learn', verboseFlag);

% now -- compute the likelihood of the model, given the data
totalLL = hcbnObj.dataLogLikelihood(x.data);
fprintf('Total LL=%0.02f\n', totalLL);
