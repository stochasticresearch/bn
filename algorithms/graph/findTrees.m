function [trees, subgraphs] = findTrees(dag)
%FINDTREES - finds independent trees, given a DAG

% g1 = digraph(dag,nodeNames);
g1 = digraph(dag);
[~,g2] = toposort(g1);
dag_toposort = adjacency(g2);

numTrees = size(dag,1)-size(g2.Edges,1);
trees = cell(1,numTrees);

nodes = 1:size(dag,1);
for ii=1:numTrees
    chain = [];
    
    chain = nodes(1); chainIdx=chain; nodes(nodes==chainIdx)=[];
    quitFlag = 0;
    while(~quitFlag)
        chainIdx = find(dag_toposort(chainIdx,:));
        if(~isempty(chainIdx))
            chain = [chain chainIdx];
            nodes(nodes==chainIdx) = [];
        else
            quitFlag = 1;
        end
    end
    trees{ii} = chain;
    subgraphs{ii} = subgraph(g2,chain);
end

end