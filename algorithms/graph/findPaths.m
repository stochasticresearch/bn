function [paths] = findPaths(dag, requestedNodesIdxs, givenNodesIdxs)
%FINDPATHS - finds all the independent paths in the dag that we need to
%process in order compute the inference of request

trees = findTrees(dag);
numTrees = length(trees);

paths = cell(1,numTrees);
for ii=1:numTrees
    % find which requested & given nodes exist in this tree
    treeNodes = intersect(trees{ii}, [requestedNodesIdxs givenNodesIdxs]);
    
    % maximize the distance, and store this as a path that needs to be
    % processed
    path = min(treeNodes):max(treeNodes);
    if(~isempty(path))
        paths{ii} = path;
    end
end

emptyCells = cellfun('isempty', paths); 
paths(emptyCells) = [];

end