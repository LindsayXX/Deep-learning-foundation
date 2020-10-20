function [eta, rho, ConvNet] = Initialize2(d, n1, n2, k1, k2, fsize)
% n1:nr of filters applied at layer 1
% n2:nr of filters applied at layer 2
% k1: width of the filters applied at layer 1(with size d*k1)
% K2: width of the filters applied at layer 2(with size n1*k2)
% fsize: number of elements in X(2) in eq(4)
% set sig1~sig3 by using He Initialization
eta = 0.001; % learning rate
rho = 0.9; % momentum
sig1 = sqrt(2);
sig2 = sqrt(2 / n1);
sig3 = sqrt(2 / fsize);
ConvNet.F{1} = randn(d, k1, n1)*sig1;
ConvNet.F{2} = randn(n1, k2, n2)*sig2;
K = 18;
ConvNet.W = randn(K, fsize)*sig3;
end