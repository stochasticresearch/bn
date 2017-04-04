%ALLTOPOST  Generates all topological sorting arrangements.
%   For a binary adjacency matrix M, which represents a directed graph,
%   ALLTOPOSORT(M) returns a matrix with all plausible topological sorting
%   arrangements.
%
% Example: 
%   % We have a graph with 3 nodes: {A,B,C} and a constrain: A<C
%   graph = [0 0 1; 0 0 0; 0 0 0]   
%   alltoposort(graph)
%
% References:
%   http://comjnl.oxfordjournals.org/content/24/1/83.abstract (article)
%   http://www.sitc.net.au/topologicalsorting/  (an online implementation)

%   Contributed by Jan Motl (jan@motl.us)
%   $Revision: 1.1 $  $Date: 2016/04/24 12:31:01 $

function result = alltoposort(graph)
    % Parameter control
    validateattributes(graph, {'numeric', 'logical'}, {'nonnan'});
    if size(graph, 1) ~= size(graph, 2) || size(graph, 1) == 1
        error('The input must be a square matrix');
    end

    % Initialization
    N = length(graph);              % Count of nodes
    ordering = topoSort(graph);     % Find one solution
    m = graph(ordering, ordering);  % Perform the required substitution 
    M = [m ones(N, 1)];             % Matrix with constrains after the substitution
    LOC = 1:N;                      % A single solution after the substitution
    P = 1:N+1;      
    result = LOC;                   % The output with all the orderings
    I = 1;                          % Index
    
    % Get all orderings
    while I < N 
        K = LOC(I);
        K1 = K+1;
        OBJ_K = P(K);
        OBJ_K1 = P(K1);

        if M(I, OBJ_K1)             % Is swapping forbidden?
            for L = K:-1:(I+1)          
                P(L) = P(L-1);
            end
            P(I) = OBJ_K;
            LOC(I) = I;
            I = I+1;    
        else                        % Swap the objects
            P(K) = OBJ_K1;
            P(K1) = OBJ_K;
            LOC(I) = K1;
            I = 1;
            result = [result; P(1:N)]; 
        end
    end
    
    % Assembly the output
    result = ordering(result(:,1:N));  % Reverse the substition
end
