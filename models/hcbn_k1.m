classdef hcbn_k1 < handle & matlab.mixin.Copyable
    %HCBN Definition of a Hybrid Copula Bayesian Network w/ maximum
    %in-degree and out-degree to be 1, i.e. a tree structure 
    %
    %**********************************************************************
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
    %* along with this program.  If not, see <http://www.gnu.org/licenses/>
    %* 
    %**********************************************************************

    properties
        dag;        % the Directed Acyclic Graph structure.  The format of
                    % the DAG is an adjancency matrix.  Element (i,j)
                    % represents a connection from the ith random variable
                    % to the jth random variable
                    % TODO: consider changing this to a sparse matrix
                    % representation
        dag_topoSorted; % a topologically sorted verison of the DAG
        topoOrder;  % ordering of the nodes topologically by index
        rg;         % the reachability graph
        topoSortedFlag; % flag indicating whether topological sort was conducted
        nodeNames;  % a cell array of the list of node names  
        nodeVals;   % an array of the topological order of the nodes w.r.t
                    % the DAG
        D;          % the number of nodes in the graph
        empInfo;    % a cell array of the empirical information for each
                    % node.  empInfo{i} is the empirical information for
                    % the node nodeNames{nodeVals{i}}.  Empirical
                    % information consits of 3 things, 1.) the domain of
                    % the random variable, the empirical marginal density 
                    % function for node i, and the empirical marginal 
                    % distribution function for node i
        copulaFamilies; % a cell array of the copula of the family associated 
                        % with nodeNames{nodeVals{i}}
        X;          % an M x D matrix of the observable data.  This can be
                    % considered the "training" data.
        X_xform;    % a transformed M x D matrix of the observable data,
                    % where we "continue" the discrete random variables, as
                    % described in Neslehova's paper, and Denuit and
                    % Lambert's paper
        K;          % the number of points over the unit-hypercube for
                    % which to calculate the empirical copula.  The higher
                    % the value of K, the greater the accuracy, but also
                    % the higher the memory and processing time
                    % requirements
        discreteNodes;  % the names of the nodes which should be modeled as
                        % discrete random variables
        discNodeIdxs;   % the indices of of the nodes of the discrete
                        % random variables
        PROXY_COMPUTATION_METHOD; % variable which indicates which measure
                                  % we use to compute LL proxy
                        
        LOG_CUTOFF;
        
        PSEUDO_OBS_CALC_METHOD;     % should be either RANK or ECDF
        VERBOSE_MODE;

        MIN_PROB;
    end
    
    methods
        function obj = hcbn_k1(X, nodeNames, discreteNodes, K, varargin)
            % HCBN - Constructs a HCBN object
            %  Inputs:
            %   X - a M x D matrix of the the observable data which the
            %       HCBN will model
            %   nodes - a cell array of names which represent the columns,
            %           where the ith 
            %  Optional Inputs:
            %   dag - an adjacency matrix representing the DAG structure.
            %         If this is not specified, it is assumed that the user
            %         will learn the structure through an available
            %         learning algorithm
            %
            %  TODO
            %   [ ] - 
            %
            nVarargs = length(varargin);
            if(nVarargs>0)
                obj.VERBOSE_MODE = varargin{1};
                if(obj.VERBOSE_MODE)
                    dispstat('','init'); % One time only initialization
                    dispstat(sprintf('Begining HCBN Initialization...'),'keepthis','timestamp');
                end
            end
            obj.LOG_CUTOFF = 10^-5;
            obj.MIN_PROB = .001;
            
            obj.PSEUDO_OBS_CALC_METHOD = 'RANK';    % can be either RANK or ECDF
            obj.PROXY_COMPUTATION_METHOD = 'srho';  % can be either srho, tau or cim
            
            obj.D = size(X,2);      
            
            obj.nodeNames = nodeNames;
            obj.nodeVals = 1:obj.D;
            obj.copulaFamilies = cell(1,obj.D);
            obj.empInfo = cell(1,obj.D);
            
            obj.dag = zeros(obj.D,obj.D);
            obj.dag_topoSorted = [];
            obj.rg = [];
            
            obj.K = K;
            
            obj.X = X;
            obj.X_xform = X;
            obj.topoSortedFlag = 0;

            obj.discreteNodes = discreteNodes;
            obj.discNodeIdxs = zeros(1,length(obj.discreteNodes));
            for ii=1:length(obj.discNodeIdxs)
                nodeName = obj.discreteNodes{ii};
                for jj=1:length(obj.nodeNames)
                    if(isequal(nodeName,obj.nodeNames{jj}))
                        obj.discNodeIdxs(ii) = jj;
                        break;
                    end
                end
            end
            
            % continue the discrete random variables.  X* = X + (U - 1)
            for idx=obj.discNodeIdxs
                obj.X_xform(:,idx) = continueRv(obj.X(:,idx));
            end
            
            % calculate empirical distributions.  We need to do this before
            % estimating structure b/c after structure estimation, we
            % estimate copula families, which requires the empirical
            % distributions to be estimated.  The topological sort will
            % rearrange the cell-array produced by this function
            % appropriately
            obj.calcEmpInfo();

            if(nVarargs>2)
                obj.PROXY_COMPUTATION_METHOD = varargin{3};
            end
            if(nVarargs>1)
                if(ischar(varargin{2}))
                    % assume it is just a flag and we attempt to perform
                    % speedy structure learning
                    seedDag = [];       % start w/ an empty seed
                    learnstruct_hc_llproxy(obj, seedDag, obj.VERBOSE_MODE);
                else
                    candidateDag = varargin{2};
                    if(~acyclic(candidateDag) || isGraphDegG1(candidateDag))
                        error('Specified DAG is not acyclic or In/Out Degree > 1!\n');
                    end
                    % ensure that the DAG dimensions correspond w/ the data
                    if(size(candidateDag,1)~=size(candidateDag,2) || size(candidateDag,1)~=obj.D)
                        error('DAG must be a square matrix of size D!');
                    end
                    estimateCop = 1;    % flag allows us to toposort & estimate the copula
                    obj.setDag(candidateDag, estimateCop);
                end
            end
            if(obj.VERBOSE_MODE)
                dispstat(sprintf('HCBN Initialization complete!'),'keepthis','timestamp');
            end
        end
        
        function [] = setVerbose(obj, mode)
            obj.VERBOSE_MODE = mode;
        end
        
        function [] = calcEmpInfo(obj)
            %CALCEMPINFO - calculates the empirical distribution function
            %              and the empirical density function via kernel
            %              based methods of the marginal distributions of
            %              the dataset (i.e. each column of the inputted
            %              data matrix X)
            for ii=1:obj.D
                if(obj.VERBOSE_MODE)
                    dispstat(sprintf('Learning Node %d/%d empirical distribution...',ii,obj.D),'timestamp');
                end
                % check if this is a discrete or continuous node for
                % density estimation, we handle these separately
                isdiscrete = 0;
                if(any(obj.discNodeIdxs==ii))
                    isdiscrete = 1;
                end
                [F,x] = empcdf(obj.X(:,ii), isdiscrete, obj.K);
                f = emppdf(obj.X(:,ii), isdiscrete, obj.K);
                empInfoObj = rvEmpiricalInfo(x, f, F, isdiscrete);
                obj.empInfo{ii} = empInfoObj;
            end
        end
        
        function [parentIdxs, parentNames] = getParents(obj, node)
            %GETPARENTS - returns the indices and the names of a nodes
            %                 parents
            %
            % Inputs:
            %  node - the node index or name of the node for which the
            %         parents are desired
            %
            % Outputs:
            %  parentIdxs - a vector of the indices of all the parents of
            %               this node
            %  parentNames - a cell array of the names of the parents of
            %               this node
            if(ischar(node))
                % node name was provided
                nodeName = node;
                for ii=1:obj.D
                    if(isequal(nodeName, obj.nodeNames{ii}))
                        nodeIdx = ii;
                        break;
                    end
                end
            else
                % assume node index was provided
                nodeIdx = node;
            end
            
            % find the node's parents
            if(obj.topoSortedFlag)
                parentIdxs = find(obj.dag_topoSorted(:,nodeIdx))';
            else
                parentIdxs = find(obj.dag(:,nodeIdx))';
            end
            parentNames = obj.nodeNames(parentIdxs);
        end
        
        function [] = doToposorting(obj)
            g1 = digraph(obj.dag,obj.nodeNames);
            [n,g2] = toposort(g1,'Order','stable');
            obj.dag_topoSorted = adjacency(g2);
            obj.rg = reachability_graph(obj.dag_topoSorted);
            
            % reorder all the data and arrays according to the
            % topologically sorted order
            obj.nodeNames = obj.nodeNames(n);
            obj.nodeVals = obj.nodeVals(n);
            obj.empInfo = obj.empInfo(n);
            
            % reorder the internal data
            obj.X_xform = obj.X_xform(:,n);
            obj.X = obj.X(:,n);
            
            % fix the discNodeIdxs after re-ordering the nodes
            obj.discNodeIdxs = zeros(1,length(obj.discreteNodes));
            for ii=1:length(obj.discNodeIdxs)
                nodeName = obj.discreteNodes{ii};
                for jj=1:length(obj.nodeNames)
                    if(isequal(nodeName,obj.nodeNames{jj}))
                        obj.discNodeIdxs(ii) = jj;
                        break;
                    end
                end
            end
            obj.topoSortedFlag = 1;
        end

        function [] = setDag(obj, candidateDag, estimateCop)
            obj.dag = candidateDag;
            if(estimateCop)
                obj.doToposorting();
                obj.estFamilyCopula();
            end
        end
        
        function [] = estFamilyCopula(obj)
            %ESTFAMILYCOPULA - estimates the copulas for each family (child
            %                  and parents), based on the current definition
            %                  of the dag, which is stored in obj.dag
            
            % for each node, estimate the copula of that node and it's
            % parents
            for ii=1:obj.D
                if(obj.VERBOSE_MODE)
                    dispstat(sprintf('Learning Copula Family for Node %d/%d...',ii,obj.D),'timestamp');
                end
                
                node = obj.nodeNames{ii};
                nodeIdx = obj.nodeVals(ii);
                [parentIdxs, parentNames] = obj.getParents(nodeIdx);
                
                if(isempty(parentIdxs))
                    % no parents situation
                    obj.copulaFamilies{nodeIdx} = [];
                else
                    % grab the appropriate values 
                    X_in = zeros(size(obj.X_xform,1), 1+length(parentNames));
                    X_in(:,1) = obj.X_xform(:,nodeIdx);
                    kk = 2:2+length(parentIdxs)-1;
                    X_in(:,kk) = obj.X_xform(:,parentIdxs);
                    
                    % fit all continuous and hybrid models w/ an empirical
                    % copula
                    if(strcmpi(obj.PSEUDO_OBS_CALC_METHOD, 'ecdf'))
                        U_in = pobs(X_in, 'ecdf', 100);
                    else
                        U_in = pobs(X_in);
                    end

                    % perform model selection on copula
                    [c_model_name,c_model_params] = copmodelsel_helm(U_in);
                    
                    % compute c so we can integrate
                    u = linspace(0,1,obj.K);
                    [U,V] = ndgrid(u,u);
                    c = reshape(copulapdf(c_model_name,[U(:),V(:)], c_model_params),obj.K,obj.K);
                    
                    allIdxs = [nodeIdx parentIdxs];
                    % find which dimensions are discrete, and integrate
                    % those out of c.  This is the equivalent of takign the
                    % partial derivative of the copula function w.r.t. only
                    % the continuous variables
                    [~,discreteDimensions,~] = intersect(allIdxs,obj.discNodeIdxs); discreteDimensions = discreteDimensions';
                    if(~isempty(discreteDimensions) && length(discreteDimensions)<length(allIdxs))
                        % hybrid scenario, where we manually integrate out
                        % the dimensions that are discrete
                        C_discrete_integrate = c;
                        for discreteDimension=discreteDimensions
                            C_discrete_integrate = cumtrapz(u,C_discrete_integrate,discreteDimension);
                        end
                    else
                        C_discrete_integrate = [];
                        % works for both all-continuous and all-discrete
                        % scenarios because we can use the pdf and cdf
                        % models directly.
                    end

                    copFam = hcbnk1family(node, nodeIdx, parentNames, parentIdxs, ...
                            c_model_name, c_model_params, C_discrete_integrate);
                    obj.copulaFamilies{nodeIdx} = copFam;
                end
            end
        end
        
        function [X] = genFamilySamples(obj, node, M)
            %GENFAMILYSAMPLES - generates samples from a specified family
            % Inputs:
            %  node - the child node, whose family is sampled
            %  M - the number of samples to generate
            % Outputs:
            %  X - the samples from the family associated with the input
            %      node.  The columns of X are [node, parent1, parent2 ...]
            
            if(ischar(node))
                % node name was provided
                nodeName = node;
                for ii=1:obj.D
                    if(isequal(nodeName, obj.nodeNames{ii}))
                        nodeIdx = ii;
                        break;
                    end
                end
            else
                % assume node index was provided
                nodeIdx = node;
            end
            
            % generate U from the copula that was "learned"
            copFam = obj.copulaFamilies{nodeIdx};
            U = copularnd(copFam.c_model_name, copFam.c_model_params, M);
            D_family = size(U,2);
            X = zeros(size(U));
            % invert each U appropriately to generate X
            allIdxs = [nodeIdx copFam.parentNodeIdxs];
            for ii=1:M
                for dd=1:D_family
                    X(ii,dd) = obj.empInfo{allIdxs(dd)}.icdf(U(ii,dd));
                end
            end
        end
        
        function [mixedProbability] = computeMixedJointProbability_(obj, X, idxs, nodeNum, parentsFlag)
            %COMPUTEMIXEDPROBABILITY - computes the joint probability of a
            %mixed probability distribution for the indices given by idxs.
            % Inputs:
            %  X - the data point, should be the entire dimension obj.D,
            %      this function picks the appropriate points from that
            %      according to idxs
            %  idxs - the indices of the nodes for which to calculate the
            %         mixed joint probability.
            %  parentsFlag - 1 if we are calculating the mixed prob of the
            %                parents
            
            if(length(idxs)==1)
                mixedProbability = obj.empInfo{idxs}.pdf(X(idxs));
            else
                [~,discreteIdxs,~] = intersect(idxs,obj.discNodeIdxs); discreteIdxs = discreteIdxs';
                continuousIdxs = setdiff(1:length(idxs),discreteIdxs);

                u = zeros(1,length(idxs));

                % fill the u w/ the continuous values, since we
                % already performed the partial derivative w.r.t.
                % the continuous variables for the copula
                for continuousIdx=continuousIdxs
                    continuousNodeNum = idxs(continuousIdx);
                    % query that node's distribution and insert into u
                    u(continuousIdx) = obj.empInfo{continuousNodeNum}.cdf(X(continuousNodeNum));
                end

                % compute the coupla value for the discrete
                % portions through rectangle differencing
                vals = 0:2^length(discreteIdxs) - 1;
                rectangleDiffStates = dec2bin(vals)-'0';
                mixedProbability = 0;
                for ii=1:size(rectangleDiffStates,1)
                    rectangleDiffState = rectangleDiffStates(ii,:);
                    if(~isempty(discreteIdxs))
                        diffStateIdx = 1;
                        for diffState=rectangleDiffState
                            discreteIdx = discreteIdxs(diffStateIdx);
                            discreteNodeNum = idxs(discreteIdx);
                            u(discreteIdx) = obj.empInfo{discreteNodeNum}.cdf(X(discreteNodeNum)-diffState);
                            diffStateIdx = diffStateIdx + 1;
                        end
                    end
                    if(parentsFlag)
                        tmp = (-1)^(sum(rectangleDiffState))*empcopulaval(obj.copulaFamilies{nodeNum}.C_parents_discrete_integrate, u, 1/obj.K);
                    else
                        tmp = (-1)^(sum(rectangleDiffState))*empcopulaval(obj.copulaFamilies{nodeNum}.C_discrete_integrate, u, 1/obj.K);
                    end

                    mixedProbability = mixedProbability + tmp;
                end

                % multiply w/ the marginal distributions of the
                % continuous variables
                for continuousIdx=continuousIdxs
                    continuousNodeNum = idxs(continuousIdx);
                    mixedProbability = mixedProbability * obj.empInfo{continuousNodeNum}.pdf(X(continuousNodeNum));
                end
            end
        end
        
        function [conditionalProb] = computeMixedConditionalProbability_(obj, X, idxs, nodeNum)
            jointProbAllNodes = obj.computeMixedJointProbability_(X, idxs, nodeNum, 0);
            jointProbParentNodes = obj.computeMixedJointProbability_(X, idxs(2:end), nodeNum, 1);
            conditionalProb = jointProbAllNodes/jointProbParentNodes;
        end
        
        function [ ll_proxy_val ] = dataLogLikelihood_proxy(obj)
            % Computes a Proxy for the log-likelihood!  From the paper:
            % Speedy Model Selection for CBN's by Tenzer & Elidan:
            %  https://arxiv.org/abs/1309.6867
            % we know that we can use any DPI satisfying measure as a proxy
            % to log-likelihood.  This includes CIM (for continuous
            % networks), CIM_S (for hybrid networks), tau, srho (for
            % continuous networks), and tau_s (for hybrid networks)
            
            % WARNING!! - Should only be used with DAG's that are limited
            % to nodes w/ only 1 parent (K=1).
            
            % For each node, find its parent, and compute the likelihood
            % proxy
            ll_proxy_val = 0;
            for dd=1:obj.D
                parentIdx = obj.getParents(dd);
                if(~isempty(parentIdx))
                    if(length(parentIdx)>1)
                        error('Not a Tree w/ 1 parent!!');
                    else
                        dataX = obj.X_xform(:,dd);
                        dataY = obj.X_xform(:,parentIdx);
                        
                        if(strcmpi(obj.PROXY_COMPUTATION_METHOD,'srho'))
                            score_proxy = abs(corr(dataX,dataY,'type','Spearman'));
                        elseif(strcmpi(obj.PROXY_COMPUTATION_METHOD,'ktau'))
                            score_proxy = abs(corr(dataX,dataY,'type','Kendall'));
                        elseif(strcmpi(obj.PROXY_COMPUTATION_METHOD,'taukl'))
                            [u,v] = pobs_sorted_cc(obj.X(:,dd),obj.X(:,parentIdx));
                            score_proxy = abs(taukl_cc(u,v,0,0,1));
                        elseif(strcmpi(obj.PROXY_COMPUTATION_METHOD,'cim'))
                            score_proxy = abs(cim(dataX,dataY,0.015625,0.2,0,0,0));
                        else
                            error('Invalid Proxy Computation Method!');
                        end
                        ll_proxy_val = ll_proxy_val + score_proxy;
                    end
                end
            end
        end
        
        function [ ll_val, totalProbVec ] = dataLogLikelihood(obj, X)
            M = size(X,1);
            if(size(X,2)~=obj.D)
                error('Input data for LL calculation must be the same dimensions as the BN!');
            end      
            
            if(nargout>1)
                totalProbVec = zeros(1,M);
            end
            
            ll_val = 0;
            for mm=1:M
                % for each D-dimensional data point, compute the likelihood
                totalProb = 1;
                for dd=1:obj.D
                    % get the parents for this node
                    parentIdxs = obj.getParents(dd);
                    if(isempty(parentIdxs))
                        tmp = obj.empInfo{dd}.pdf(X(mm,dd));
                    else
                        allIdxs = [dd parentIdxs];
                        tmp = obj.computeMixedConditionalProbability_(X(mm,:),allIdxs, dd);
                    end
                    totalProb = totalProb * tmp;
                end
                
                if(isnan(totalProb))
                    1;      % for DEBUGGING :D
                end
                if(totalProb<=obj.LOG_CUTOFF)
                    totalProb = obj.LOG_CUTOFF;
                end
                ll_val = ll_val + log(totalProb);
                
                if(nargout>1)
                    totalProbVec(mm) = totalProb;
                end
            end
        end
        
        function [ll_val] = copulaLogLikelihood(obj, X)
            % WARNING - this method should *ONLY* be used if you are
            % dealing w/ all continuous nodes (i.e. using the hcbn as a
            % non-parametric CBN)
            M = size(X,1);
            if(size(X,2)~=obj.D)
                error('Input data for LL calculation must be the same dimensions as the BN!');
            end      
            ll_val = 0;
            for mm=1:M
                for dd=1:obj.D
                    R_ci = obj.computeCopulaRatio(dd, X(mm,:));
                    
                    if(isinf(R_ci) || isnan(R_ci))
                        error('R_ci is inf/nan!');
                    end
                    
                    if(R_ci < obj.LOG_CUTOFF)
                        R_ci = obj.LOG_CUTOFF;
                    end
                    
                    ll_val = ll_val + log(R_ci);
                    if(~isreal(ll_val))
                        % for debugging purposes :)
                        error('LL Value imaginary!');
                    end
                end
            end
        end
        
        function [rcVal] = computeCopulaRatio(obj, nodeIdx, x)
            % WARNING - this method should *ONLY* be used if you are
            % dealing w/ all continuous nodes (i.e. using the hcbn as a
            % non-parametric CBN)
            copFam = obj.copulaFamilies{nodeIdx};
            if(isempty(copFam))
                rcVal = 1;
            else
                idxs_all = [nodeIdx copFam.parentNodeIdxs];
                x_all = x(idxs_all);                
                u_all = zeros(1,length(x_all));
                % convert to pseudo-observations via ECDF
                for ii=1:length(x_all)
                    u_all(ii) = obj.empInfo{idxs_all(ii)}.cdf(x_all(ii));
                end                
                
                % We don't have to compute the copula ratio b/c in the k1
                % version, we are limiting ourselves to a graph structure
                % that has atmost in-degree=1 and out-degree=1
                rcVal = copulapdf(copFam.c_model_name, u_all, copFam.c_model_params);
            end
        end
        
        function [prob, xy] = computePairwiseJoint(obj, childNode, parentNode)
            % determine if we have continuous, hybrid-1, hybrid-2, or
            % discrete scenarios
            childDiscrete = any(childNode==obj.discNodeIdxs);
            parentDiscrete = any(parentNode==obj.discNodeIdxs);
            
            if(~childDiscrete && ~parentDiscrete)
                probabilityType='all-continuous';
            elseif(childDiscrete && parentDiscrete)
                probabilityType='all-discrete';
            elseif(childDiscrete && ~parentDiscrete)
                probabilityType='hybrid-1';
            else
                probabilityType='hybrid=2';
            end
            
            % retrieve the domain of the child and parent, over which we
            % will compute the probabilities
            childObj = obj.empInfo{childNode};
            parentObj = obj.empInfo{parentNode};
            copFamilyObj = obj.copulaFamilies{childNode};
            childDomain = childObj.domain;
            parentDomain = parentObj.domain;
            prob = zeros(length(childDomain), length(parentDomain));
            xy   = cell(length(childDomain), length(parentDomain));
            
            if(strcmpi(probabilityType, 'all-continuous'))
                % f(x,y) = f(x)*f(y)*c(F(x),F(y))
                for ii=1:length(childDomain)
                    xVal = childDomain(ii);
                    for jj=1:length(parentDomain)
                        yVal = parentDomain(jj);
                        uVec = [childObj.cdf(xVal) parentObj.cdf(yVal)];
                        probVal = childObj.pdf(xVal)*...
                                  parentObj.pdf(yVal)*...
                                  copulapdf(copFamilyObj.c_model_name, uVec, copFamilyObj.c_model_params);
                        prob(ii,jj) = probVal;
                        xy{ii,jj} = [xVal yVal];
                    end
                end
            elseif(strcmpi(probabilityType, 'all-discrete'))
                % f(x,y) = sum(sum( [(-1)^(i+j) C(u_i,v_j)], 1, 2), 1, 2)
                %  u_1 = F(x-), u_2 = F(x)
                %  v_1 = F(y-), v_2 = F(y)
                for ii=1:length(childDomain)
                    xVal = childDomain(ii);
                    for jj=1:length(parentDomain)
                        yVal = parentDomain(jj);
                        if(ii==1)
                            u1 = 0;
                        else
                            u1 = childObj.cdf(childDomain(ii-1));
                        end
                        u2 = childObj.cdf(xVal);
                        
                        if(jj==1)
                            v1 = 0;
                        else
                            v1 = parentObj.cdf(parentDomain(jj-1));
                        end
                        v2 = parentObj.cdf(yVal);
                        % compute the C-Volume
                        model_name = copFamilyObj.c_model_name;
                        model_params = copFamilyObj.c_model_params;
                        probVal = copulacdf(model_name, [u2 v2], model_params) - ...
                                  copulacdf(model_name, [u2 v1], model_params) - ...
                                  copulacdf(model_name, [u1 v2], model_params) + ...
                                  copulacdf(model_name, [u1 v1], model_params);
                        prob(ii,jj) = probVal;
                        xy{ii,jj} = [xVal yVal];
                    end
                end
                
            elseif(strcmpi(probabilityType, 'hybrid-1'))
                % f(x,y) = sum( [(-1)^(i+j) C(u_i,F(y))], 1, 2) * f(y)
                %  u_1 = F(x-), u_2 = F(x)
                for ii=1:length(childDomain)
                    xVal = childDomain(ii);
                    for jj=1:length(parentDomain)
                        yVal = parentDomain(jj);
                        if(ii==1)
                            u1 = 0;
                        else
                            u1 = childObj.cdf(childDomain(ii-1));
                        end
                        u2 = childObj.cdf(xVal);
                        v = parentObj.cdf(yVal);
                        probVal = parentObj.pdf(yVal)*...
                                  (empcopulaval(copFamilyObj.C_discrete_integrate, [u2, v], 0) - ...
                                   empcopulaval(copFamilyObj.C_discrete_integrate, [u1, v], 0) );
                        prob(ii,jj) = probVal;
                        xy{ii,jj} = [xVal yVal];
                    end
                end
            else
                % f(x,y) = sum( [(-1)^(i+j) C(F(x),v_i)], 1, 2) * f(x)
                %  v_1 = F(y-), v_2 = F(y)
                for ii=1:length(childDomain)
                    xVal = childDomain(ii);
                    for jj=1:length(parentDomain)
                        yVal = parentDomain(jj);
                        if(jj==1)
                            v1 = 0;
                        else
                            v1 = parentObj.cdf(parentDomain(jj-1));
                        end
                        v2 = parentObj.cdf(yVal);
                        u = childObj.cdf(xVal);
                        probVal = childObj.pdf(xVal)*...
                                  (empcopulaval(copFamilyObj.C_discrete_integrate, [u, v2], 0) - ...
                                   empcopulaval(copFamilyObj.C_discrete_integrate, [u, v1], 0) );
                        prob(ii,jj) = probVal;
                        xy{ii,jj} = [xVal yVal];
                    end
                end
            end
            
            % renormalize the probability matrix
            prob = prob/sum(prob(:));
        end
        
        function [probOut, domainOut, nodeNamesOut] = ...
            inference(obj, requestedNodes, givenNodes, givenVals, normalizeProbFlag)
            % Performs inference on the HCBN, given a certain number of
            % nodes for the requested nodes.  Requested Nodes and Given
            % Nodes can both be provided as either indices, or names.
            
            if(~obj.topoSortedFlag)
                error('Must Topologically sort the graph before performing inference!');
            end
            
            % determine whether we have indices or we need to get the
            % indices of the requested/given nodes
            if(iscell(requestedNodes))
                requestedNodesIdxs = zeros(1,length(requestedNodes));
                for ii=1:length(requestedNodes)
                    for jj=1:obj.D
                        nodeName = obj.nodeNames{jj};
                        if(isequal(requestedNodes{ii},nodeName))
                            requestedNodesIdxs(ii) = jj;
                            break;
                        end
                    end
                end
            else
                requestedNodesIdxs = requestedNodes;
            end
            if(iscell(givenNodes))
                givenNodesIdxs = zeros(1,length(requestedNodes));
                for ii=1:length(givenNodesIdxs)
                    for jj=1:obj.D
                        nodeName = obj.nodeNames{jj};
                        if(isequal(givenNodes{ii},nodeName))
                            givenNodesIdxs(ii) = jj;
                            break;
                        end
                    end
                end
            else
                givenNodesIdxs = givenNodes;
            end
            
            nodeNamesOut = cell(1,length(requestedNodes));
            nodeNamesOutIdx = 1;
            % compute the maximum number of nodes we need to compute the
            % joint probability for
            paths = findPaths(obj.dag_topoSorted, requestedNodesIdxs, givenNodesIdxs);
            numPaths = length(paths);
            probCell = cell(1,numPaths); domainCell = cell(1,numPaths);
            for kk=1:numPaths
                if(obj.VERBOSE_MODE)
                    dispstat(sprintf('Processing branch %d/%d.', kk, numPaths));
                end
                
                dagChainVec = paths{kk};
                numNodes = length(dagChainVec); 
                requestedNodesIdxsPathSubset = intersect(dagChainVec,requestedNodesIdxs);
                givenNodesIdxsPathSubset = intersect(dagChainVec,givenNodesIdxs);
                
                % perform inference for this path
                szArr = zeros(1,numNodes);
                % make the DAG chain & get create the tensor dimensions
                for ii=1:numNodes
                    szArr(ii) = length(obj.empInfo{dagChainVec(ii)}.domain);
                    if(any(ismember(requestedNodes,obj.nodeNames{dagChainVec(ii)})))
                        nodeNamesOut{nodeNamesOutIdx} = obj.nodeNames{dagChainVec(ii)};
                        nodeNamesOutIdx = nodeNamesOutIdx + 1;
                    end
                end
                numPairwiseProbsToCompute = numNodes-1;

                if(numPairwiseProbsToCompute==0)
                    % if the dagChain is not empty, then this means that we
                    % can just extract the empirical distribution for this
                    if(~isempty(dagChainVec))
                        probCell{kk} = obj.empInfo{dagChainVec(1)}.density;
                        domainCell{kk} = num2cell(obj.empInfo{dagChainVec(1)}.domain);
                    else
                        error('Path does not have a chain!');
                    end
                else
                    pairwiseJointProbsCell = cell(1,numPairwiseProbsToCompute);
                    xyCell = cell(1,numPairwiseProbsToCompute);
                    % compute pairwise joint probabilities
                    if(obj.VERBOSE_MODE)
                        dispstat(sprintf('Computing %d pairwise probabilities.', numPairwiseProbsToCompute));
                    end
                    for ii=1:numPairwiseProbsToCompute
                        childNode = dagChainVec(ii+1);
                        parentNode = dagChainVec(ii);
                        [pairwiseJointProbsCell{ii}, xyCell{ii}] = obj.computePairwiseJoint(childNode, parentNode);
                    end

                    % compute overall joint
                    % TODO: is there a more efficient way to do this w/ matrix
                    % operations?  I would think so ...
                    if(obj.VERBOSE_MODE)
                        dispstat('Computing overall joint probability distribution.');
                    end
                    fullJointProbTensor = zeros(szArr);
                    domain = cell(szArr);
                    idxsOutput = cell(1,numel(szArr));
                    for ii=1:prod(szArr)
                        [idxsOutput{:}] = ind2sub(szArr,ii);
                        idxsMat = cell2mat(idxsOutput);
                        prob = 1;
                        domainVec = zeros(1,numPairwiseProbsToCompute+1);
                        for jj=1:numPairwiseProbsToCompute
                            pairwiseJoint = pairwiseJointProbsCell{jj};
                            xy = xyCell{jj};
                            accessIdxVec = fliplr(idxsMat(jj:jj+1));    % flip because the pairwise joints are stored as [child/parent]
                                                                        % and idxsMat is stored as parent/child
                            prob = prob*pairwiseJoint(accessIdxVec(1),accessIdxVec(2));
                            xyVec = xy{accessIdxVec(1),accessIdxVec(2)};
                            domainVec(jj:jj+1) = fliplr(xyVec);
                        end
                        fullJointProbTensor(ii) = prob;
                        domain{ii} = domainVec;
                    end

                    % apply the given variables
                    if(obj.VERBOSE_MODE)
                        dispstat('Applying given variables.');
                    end
                    outputTensor = fullJointProbTensor;
                    idxsToRemove = zeros(1,length(givenNodesIdxsPathSubset));
                    for ii=1:length(givenNodesIdxsPathSubset)
                        givenNodeIdx = givenNodesIdxsPathSubset(ii);
                        givenNodeVal = givenVals(ii);
                        % find ii in the dag-chain vec to determine which index in
                        % our fullJoint we need to access .. this determines which
                        % of the (:,:,:,...:) into which we need to put a specific
                        % number is
                        fullJointSliceDim = find(dagChainVec==givenNodeIdx);

                        % find which index coresponds to the given value 
                        % this determines what the number is that we need to put
                        % into the indexing
                        withinSliceIdx = obj.empInfo{givenNodeIdx}.findClosestDomainVal(givenNodeVal);

                        %%%%%%% EXAMPLE %%%%%
                        % suppose fullJointProbTensor = [25 x 25 x 25 x 25]
                        % fullJointSliceIdx = 2
                        % withinSliceIdx = 15, then, we whould slice as:
                        %  fullJointProbTensor(:,15,:,:) to get the
                        %  probability given a variable

                        % slice out the correct index
                        outputTensor = slice(outputTensor, withinSliceIdx, fullJointSliceDim);

                        % remove dimensions from the domain cell-array to maintain
                        % consistency between data and the corresponding reference
                        % point
                        domain = slice(domain, withinSliceIdx, fullJointSliceDim);
                        idxsToRemove(ii) = fullJointSliceDim;
                    end

                    % integrate out nuisance variables
                    if(obj.VERBOSE_MODE)
                        dispstat('Integrating out nuisance variables.');
                    end
                    nodesToIntegrateOut = setdiff(dagChainVec,requestedNodesIdxsPathSubset);
                    nodesToIntegrateOut = setdiff(nodesToIntegrateOut, givenNodesIdxsPathSubset);
                    idxsToRemove = [idxsToRemove zeros(1,length(nodesToIntegrateOut))];
                    for ii=1:length(nodesToIntegrateOut)
                        nodeToIntegrateOut = nodesToIntegrateOut(ii);

                        % the ii-1 is to allow for the fact that every-time we sum
                        % across a dimension, we reduce the dimension of the tensor
                        % by 1
                        dimToIntegrate = find(dagChainVec==nodeToIntegrateOut);
                        outputTensor = sum(outputTensor,dimToIntegrate);

                        % remove dimensions from the domain cell-array to maintain
                        % consistency between data and the corresponding reference
                        % point
                        domain = slice(domain, 1, dimToIntegrate);
                        idxsToRemove(ii+length(givenNodesIdxsPathSubset)) = dimToIntegrate;
                    end

                    % squeeze dimensions
                    % we squeeze at the end so that we can use the same indexing
                    % values above
                    prob = squeeze(outputTensor);
                    idxToKeepVec = setdiff(1:length(dagChainVec),idxsToRemove);
                    domain = cellfun(@(x) x(idxToKeepVec), domain, 'UniformOutput', false);
                    domain = squeeze(domain);

                    % don't allow zero probability
                    prob(prob==0) = obj.MIN_PROB;

                    probCell{kk} = prob;
                    domainCell{kk} = domain;
                end
            end
            
            if(numPaths==1)
                probOut = probCell{1};
                domainOut = domainCell{1};
            else
                % flatten the probCell and domainCell into one
                % probability tensor and one domain cell tensor
                
                outputShape = cell2mat(cellfun(@(x) mysizefun(x), probCell, 'UniformOutput', 0));
                numIters = prod(outputShape);
                probOut = zeros(outputShape);
                domainOut = cell(outputShape);
                I = cell(1,length(outputShape));
                probVec = zeros(1,numPaths);
                domainVec = zeros(1,length(outputShape));
                for ii=1:numIters
                    % compute the index for each path's probability
                    [I{:}] = ind2sub(outputShape,ii);
                    II = cell2mat(I);
                    % get the probabilities from each path, multiply, and store
                    accessIdxStart= 1;
                    storeIdxStart = 1;
                    for jj=1:numPaths
                        numIdxsToGrab = length(mysizefun(probCell{jj}));
                        idxsToGrab = accessIdxStart:numIdxsToGrab+accessIdxStart-1;
                        idxs = II(idxsToGrab);
                        domainStoreIdxs = storeIdxStart:numIdxsToGrab+storeIdxStart-1;

                        mm = probCell{jj};
                        probVec(jj) = mm(idxs);
                        dd = domainCell{jj};
                        domainVec(domainStoreIdxs) = dd{idxs};

                        % reset for the next path
                        accessIdxStart = idxsToGrab(end) + 1;
                        storeIdxStart = domainStoreIdxs(end)+1;
                    end
                    probOut(ii) = prod(probVec);
                    domainOut{ii} = domainVec;
                end
            end
            
            % normalize probability
            if((nargin>4 && normalizeProbFlag) || nargin<4)
                if(obj.VERBOSE_MODE)
                    dispstat('Re-normalizing probabilities.');
                end
                probOut = probOut/sum(probOut(:));
            end
            if(obj.VERBOSE_MODE)
                dispstat(' ');
            end
        end
    end
end

function y = mysizefun(x)
y = size(x);
if(y(1)==1)
    y = y(2:end);
end
end