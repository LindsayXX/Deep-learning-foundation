% X_batch: nlen*d x n matrix
% Ys_batch:nlen 
% MFs: MF for each layerv  [MakeMFMatrix(ConvNet.F{1}]
% W: weight matrices last fully connected layer
function [X_batch_1, X_batch_2, P_batch] = ForwardPass(X_batch, MFs, W)
% forward pass 
X_batch_1 = max(MFs{1} * X_batch, 0);
X_batch_2 = max(MFs{2} * X_batch_1, 0);
S_batch = W * X_batch_2;
P_batch = softmax(S_batch);
end

function loss = ComputeLoss(X_batch, Ys_batch, MFs, W)
% forward pass
[~, ~, P_batch] = ForwardPass(X_batch, MFs, W);
% loss
[~, n] = size(Ys_batch);
loss = 0;
for i=1:n
    loss = loss - (Ys_batch(:,i).' * P_batch(:,i));
end
loss = loss/n;
end

%X_1 = max(MFs{1} * X, 0);
function Grad = ComputeGradient(X_batch, Ys_batch, MFs, ConvNet, X_1)
[~, n] = size(Ys_batch);
%grad_W = zeros();
%grad_F1 = zeros();
%grad_F2 = zeros();
% forward pass 
X_batch_1 = max(MFs{1} * X_batch, 0);
X_batch_2 = max(MFs{2} * X_batch_1, 0);
S_batch = ConvNet.W * X_batch_2;
P_batch = softmax(S_batch);
% backward pass
% MX = MakeMXMatrix(x_input, d, k, nf), 
% k: width of filter
% nf: number of filter
% X: d(28)*nlen
G_batch = P_batch - Ys_batch;
grad_W = 1/n * (G_batch * X_batch_2.');%Grad.W

G_batch = W.'* G_batch;
indX2 = max(X_batch_2, 0);
% h=X_batch_2~=0;
G_batch = G_batch .* spones(indX2);
[~, c] = size(G_batch);
[d1, k1, nf1] = size(ConvNet.F{1});
[d2, k2, nf2] = size(ConvNet.F{2});
for j=1:c
    g_j = G_batch(:,j);
    x_j = X_1(:,j);
    MX = MakeMXMatrix(x_j, d2, k2, nf2);%(nlen-k2+1)*nf2, k2*nf2*d2
    v = g_j.' * MX;
    grad_F2 = grad_F2 + v/n;
end

G_batch = MFs{2}.' * G_batch;
indX1 = max(X_batch_1, 0);
G_batch = G_batch .* spones(indX1);
for j=1:n
    g_j = G_batch(:,j);%nf2, ?
    x_j = X_batch(:,j);%X_batch(:,:,j)(:)
    MX = MakeMXMatrix(x_j, d1, k1, nf1);%(nlen-k1+1)*nf1, k1*nf1*d1
    v = g_j.' * MX;
    % v = 
    grad_F1 = grad_F1 + v/n;
end
% for mini batch: grad_F1 = reshape(grad_vecF1, [d, k1, n1]);
Grad.F1 = grad_F1;
Grad.F2 = grad_F2;
Grad.W = grad_W;
end