% TEst the TOPOSORT algorithm

clear;
clc;

x = zeros(6,6);
x(1,2) = 1; x(1,4) = 1;
x(2,3) = 1; x(2,4) = 1; x(2,5) = 1;
x(3,4) = 1;
x(5,3) = 1;
x(6,3) = 1; x(6,5) = 1;

[n, level, dag] = topoSort(x);

n_expect = [1,6,2,5,3,4];
isequal(n,n_expect)
level