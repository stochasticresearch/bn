function [] = learnstruct_hc_llproxy(dagModelObj, seeddag, verboseFlag)
%LEARNSTRUCT_HC learn the structure of the HCBN network using
%               the hill climbing algorithm as described in 
%               Koller and Friedman (2009).  The code for this 
%               is based off the structure learning toolbox 
%               within BNT.  See BNT/SLP/learn_struct_hc.m.
%               This will use the X dataset that was passed to 
%               the constructor to learn the structure.
%
% Inputs:
%  dagModelObj - should be either a HCBN, CLG, or MTE object.
%                See copula/hcbn/hcbn.m, copula/clg/clg.m, copula/mte/mte.m
%  seeddag - a DAG which can be used as a reference DAG as a
%            starting point for the search process.  This is
%            optional, and can be an empty array if no seed is
%            desired.
%
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

if(nargin<2)
    seeddag = [];
    verboseFlag = 0;
end
if(nargin<3)
    verboseFlag = 0;
end

estimateCop = 0;    % we don't need to estimate the copula while structure learning

% ensure that the seeddag is acyclic
if(isempty(seeddag) || ~acyclic(seeddag))
    warning('Specified seeddag is NOT acyclic!');
	dagModelObj.setDag(zeros(dagModelObj.D,dagModelObj.D), estimateCop);
else
	dagModelObj.setDag(seeddag, estimateCop);
end

% get the baseline score
bestScore = dagModelObj.dataLogLikelihood_proxy();
if(verboseFlag)
    dispstat('','init'); % One time only initialization
    dispstat(sprintf('Begining Structure Learning...'),'keepthis','timestamp');
end
done = 0;
iter = 0;
while ~done
	% make dag's which are addition, reversal, and subtraction
	% of edges
	[candidateDags,~,~] = mk_nbrs_of_dag_K1(dagModelObj.dag);
	
	% score all the dags
	scores = -Inf*ones(1,length(candidateDags));
	for ii=1:length(candidateDags)
		dagModelObj.setDag(candidateDags{ii}, estimateCop);
		scores(ii) = dagModelObj.dataLogLikelihood_proxy();
        if(verboseFlag)
            dispstat(sprintf('%d/%d-Dag Score=%0.02f',ii,length(candidateDags),scores(ii)));
        end
	end
	
	% find the maximum scoring DAG, and see if it is better
	% than the current best
	maxScore = max(scores);
	% see if multiple candidate dag's had the same maximum
	% score, and if so, choose randomely among those dag's
	new = find(scores == maxScore );
	% update best candidate dag as new dag and continue search
	if ( ~isempty(new) && (maxScore > bestScore) )
		p = sample_discrete(normalise(ones(1, length(new))));
		bestScore = maxScore;
		dagModelObj.setDag(candidateDags{new(p)},estimateCop);
	else
		done = 1;
    end 
    
    iter = iter + 1;
    if(verboseFlag)
        dispstat(sprintf('iter=%d Max Score=%0.02f',iter, maxScore),'keepthis', 'timestamp');
    end
end
dispstat('Structure Learning complete!','keepthis','timestamp');

% TODO: topo-sort the DAG, and sort the names cell array to
% match the topologically sorted DAG

% TODO: print out DAG structure

% now that structure learning is complete, estimate the copula family
% objects
dispstat('Learning Copula Families and Parameters!', 'keepthis', 'timestamp');
dagModelObj.estFamilyCopula();
dispstat('Learning complete!', 'keepthis', 'timestamp');

end