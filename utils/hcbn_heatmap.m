function [] = hcbn_heatmap(inference,domain,namesCell)

figure;
xx = cellfun(@(x) x(1),domain);
x = xx(:,1);
yy = cellfun(@(x) x(2),domain);
y = yy(1,:);
imagesc(x,y,inference);
xlabel(namesCell{1});
ylabel(namesCell{2});

end