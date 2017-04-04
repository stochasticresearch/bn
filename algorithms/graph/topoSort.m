%TOPOSORT   Performs topological sort of a directed graph.
%
% [n] = TOPOSORT(G) returns an index vector with the order of the nodes 
% sorted topologically. G is a directed graph represented by an n-by-n 
% adjacency matrix in which 1 indicates the presence of an edge and 0 
% indicates the absence of the edge. Note that the graph is not required to
% be acyclic and that the graph G may have two or more valid topological 
% orderings, even though only the first one is returned.
%
% [n, level] = TOPOSORT(G) also returns a vector with the index of the
% layer, into which the nodes were placed.
%
% [n, level, dag] = TOPOSORT(G) also returns a directed acyclic graph 
% produced from the graph G. If the graph G is acyclic, the produced dag
% is identical with the graph G.
% 
% The function is an implementation of an algorithm described in: 
%   "A Simple Algorithm for Automatic Layout of BPMN Processes"
%
% Assumptions:
%   1) The graph is directed.
%   2) The graph has at least one node without any incoming link.
%   3) The edges are not weighted.
%
% Example:
%   binary_adjacency_matrix = [0 0 1; 1 0 0; 0 0 0]; % A->C, B->A.
%   n = topoSort(binary_adjacency_matrix)
%
% See also: GRAPHTOPOORDER, BIOGRAPH.

%   Contributed by Jan Motl (jan@motl.us)
%   $Revision: 1.0 $  $Date: 2016/02/21 13:41:01 $

function [L, level, dag] = topoSort(graph)
    % Parameter test 
    validateattributes(graph, {'numeric', 'logical'}, {'2d'}); % 2d matrix with values {0, 1}
    if size(graph, 1) ~= size(graph, 2)
        error('topoSort:noStart','The graph have to be a square matrix.')
    end
    if min(sum(graph)) > 0 
        error('topoSort:noStart','The graph have to have at least one node without any incoming link.')
    end
        
    % Initialization
    indegree = sum(graph); % Incoming link counter
    indegree0 = indegree; % Initial incoming link counter

    % Topological sort
    G = 1:length(graph);   % Set of nodes to sort
    L = [];     % Empty list for the sorted elem.
    S = [];     % Empty Set for nodes with no incoming edges
    level = ones(size(G));  % Level of the node in L
    dag = graph;    % Directed Acyclic Graph

    while ~isempty(G) % G is non-empty

        % Search for free nodes
        for n = G
            if indegree(n)==0 % Incoming link counter of n = 0
                S = [S, n];
            end
        end

        if ~isempty(S) % S is non-empty, perform ordinary top-sort
            n = S(1);  % Remove a node n from S and G
            S = S(S~=n); 
            G = G(G~=n);
            L = [L, n]; % Insert n into L
            for m = find(graph(n,:)) % Foreach node m with an edge e from n to m
                graph(n,m) = 0; % Remove e
                level(m) = max(level(m), level(n)+1);
                indegree(m) = indegree(m) -1; % Decrement incoming link counter from m
            end
        else % Cycle found
            % Find loop entry
            J=-1;
            for j = G
                if indegree(j) < indegree0(j);
                    J = j;
                    break;
                end
            end
            % Process loop entry
            for e = find(graph(:,J)) % e = nodes with transition to J
                graph(J, e) = 1;  % Change the orientation of the edge
                graph(e, J) = 0;
                dag(J, e) = 1;
                dag(e, J) = 0;
                indegree(e) = indegree(e) + 1;
                indegree(J) = indegree(J) - 1;
            end
        end
    end
    
end
