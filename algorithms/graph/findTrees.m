function [trees] = findTrees(topoSortedDAG)
%FINDTREES - finds independent trees, given a topologically DAG

treeBreakIdxs = find(sum(topoSortedDAG,2)==0);
numTrees = length(treeBreakIdxs);

trees = cell(1,numTrees);
for ii=1:numTrees
    if(ii==1)
        start = 1;
    else
        zz = trees{ii-1};
        start = zz(end)+1;
    end
    trees{ii} = start:treeBreakIdxs(ii);
end

end