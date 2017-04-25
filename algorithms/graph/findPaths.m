function [paths] = findPaths(dag, requestedNodesIdxs, givenNodesIdxs)
%FINDPATHS - finds all the independent paths in the dag that we need to
%process in order compute the inference of request

if(~isempty(requestedNodesIdxs) && ~isempty(givenNodesIdxs))
    allCombos1 = combvec(requestedNodesIdxs,givenNodesIdxs)';
    allCombos2 = combvec(requestedNodesIdxs,requestedNodesIdxs)';
    allCombos3 = combvec(givenNodesIdxs,givenNodesIdxs)';
    allCombos = [allCombos1; allCombos2; allCombos3];
elseif(isempty(requestedNodesIdxs) && ~isempty(givenNodesIdxs))
    allCombos = combvec(givenNodesIdxs,givenNodesIdxs)';
elseif(~isempty(requestedNodesIdxs) && isempty(givenNodesIdxs))
    allCombos = combvec(requestedNodesIdxs,requestedNodesIdxs)';
else
    error('Must provide some indices for processing!');
end

[trees, subgraphs] = findTrees(dag);
numTrees = length(trees);

for ii=1:size(allCombos,1)
    combo = allCombos(ii,:);
    fromIdx = combo(1); toIdx = combo(2);
    ij_reach = obj.rg(fromIdx,toIdx);
    ji_reach = obj.rg(toIdx,fromIdx);
    if(ij_reach)
        [numNodesTmp,dagChainVecTmp] = shortestpath(bg,fromIdx,toIdx,'Method','Acyclic');
    elseif(ji_reach)
        [numNodesTmp,dagChainVecTmp] = shortestpath(bg,toIdx,fromIdx,'Method','Acyclic');
    else
        numNodesTmp = -999;
    end
    numNodesTmp = numNodesTmp + 1;
    if(numNodesTmp>numNodes)
        numNodes = numNodesTmp;
        dagChainVec = dagChainVecTmp;
    end
end

end