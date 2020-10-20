load('names_data.mat', 'trainXx', 'validXx');%trainX: 28*19 x 19798, validX: 28*19 x 252
load('names.mat', 'trainY', 'validY');%trainY 18 * 19798, validY 18*252

d = 28;%dim of one-hot vector, existed alphabet
n_len = 19;%19 maximum length of name
K = 18;%18 classes

n1 = 20;% n1:nr of filters applied at layer 1 
n2 = 20;% n2:nr of filters applied at layer 2
k1 = 5;% k1: width of the filters applied at layer 1(with size d*k1)
k2 = 3;% layer 2 filter n1*k2
fsize = (n_len - k1 - k2 + 2) * n2;% fsize: number of elements in X(2) in eq(4)
[eta, rho, ConvNet] = Initialize(d, n1, n2, k1, k2, fsize);

%% gradient check
%X_input: 28*19 x N matrix
n = 3;
X_batch = trainXx(:, 1:n); %trainXx(:,1:n);
Ys_batch = trainY(:, 1:n); %trainY(:,1:n);
%
n_len1 = n_len - k1 + 1;%15
MFs{1} = MakeMFMatrix(ConvNet.F{1}, n_len);
MFs{2} = MakeMFMatrix(ConvNet.F{2}, n_len1);

%
% [d, k, nf] = size(ConvNet.F{1});
% MX = MakeMXMatrix(X_batch, d, k, nf);
% vecF = ConvNet.F{1}(:);
% s1 = MX * vecF;
% s2 = MFs{1} * X_batch;
% sdiff = max(s1-s2);


Gs = NumericalGradient(X_batch, Ys_batch, ConvNet, 1e-5);
save('Gs.mat', 'Gs')
Grad = NewComputeGradients(X_batch, Ys_batch, ConvNet,MFs);
diff_1 = Gs{1} - Grad{1};
max(max(max(diff_1)))
diff_2 = Gs{2} - Grad{2};
max(max(max(diff_2)))
diff_3 = Gs{3} - Grad{3};
max(max(diff_3))
%diff1 = ComputeDiff(Gs{1}, Grad{1}, 0);
%diff2 = ComputeDiff(Gs{2}, Grad{2}, 0);
%diff3 = ComputeDiff(Gs{3}, Grad{3}, 1);


function [X_batch_1, X_batch_2, P_batch] = ForwardPass(X_batch, W, MFs)
% forward pass
X_batch_1 = max(MFs{1} * X_batch, 0);
X_batch_2 = max(MFs{2} * X_batch_1, 0);
S_batch = W * X_batch_2;
P_batch = softmax(S_batch);
end

function loss = ComputeLoss(X_batch, Ys_batch, ConvNet, MFs)
% forward pass
[~, ~, P_batch] = ForwardPass(X_batch, ConvNet, MFs);
% loss
[~, n] = size(Ys_batch);
loss = 0;
for i=1:n
    loss = loss - log(Ys_batch(:,i).' * P_batch(:,i));
end
loss = loss/n;
end


function Grad = NewComputeGradients(X_batch, Ys_batch, ConvNet, MFs)%, MX_1)
[~, n] = size(Ys_batch);
grad_W = 0;
grad_vF1 = 0;
grad_vF2 = 0;
% forward pass 
%[X_batch_1, X_batch_2, P_batch] = ForwardPass(X_batch, ConvNet.W, MFs);
X_batch_1 = max(MFs{1} * X_batch, 0);
X_batch_2 = max(MFs{2} * X_batch_1, 0);
S_batch = ConvNet.W * X_batch_2;
P_batch = softmax(S_batch);
% backward pass
% MX = MakeMXMatrix(x_input, d, k, nf), 
% X: d(28)*nlen
G_batch = P_batch - Ys_batch;
grad_W = 1/n * (G_batch * X_batch_2.'); % 18*260

% propagate 1
G_batch = ConvNet.W.'* G_batch;%260 * n
h = X_batch_2~=0;
G_batch = G_batch .* h;%260*n
%indX2 = max(X_batch_2, 0);
%G_batch = G_batch .* spones(indX2);

[d1, k1, nf1] = size(ConvNet.F{1});% 28 5 20
[d2, k2, nf2] = size(ConvNet.F{2});% 20 3 20
nlen1 = size(X_batch_1,1) / d2;
nlen2 = nlen1 - k2 + 1;
for j=1:n
    g_j = G_batch(:,j);%260*1
    x_j = X_batch_1(:,j);% 300*1
    %MX = MakeMXMatrix(x_j, d2, k2, nf2);%(nlen-k2+1)*nf2, k2*nf2*d2  260*1200
    %v = g_j.' * MX;% 1* 1200
    % speed up
    %MX = MakeMX(x_j, d2, k2);
    %MG = MakeMG(g_j, n2);
    X_j = reshape(x_j, [d2, nlen1]);
    for i=1:nlen2
        x = X_j(:, i:i+k2-1);%d*k
        vx = x(:).';
        jj = (i-1)*nf2 + 1;
        vg = g_j(jj:jj+nf2-1).';
        if i==1
            MX = vx;
            MG = vg;
        else
            MX = [MX;vx];
            MG = [MG;vg];
        end
    end
    V = MX.' * MG;
    v = V(:);
    grad_vF2 = grad_vF2 + v/n;
end

% propagate 2
G_batch = MFs{2}.' * G_batch;
h = X_batch_1 ~= 0;
G_batch = G_batch .* h;
for j=1:n
    g_j = G_batch(:,j);
    x_j = X_batch(:,j);
    MX = MakeMXMatrix(x_j, d1, k1, nf1);%(nlen-k1+1)*nf1, k1*nf1*d1 300*2800
    v = g_j.' * MX;% 1 * 2800
    % MX can be pre-computated
    %v = g_j.' * MX_1{j};
    grad_vF1 = grad_vF1 + v/n;% 1 * 2800
end

Grad{1}= reshape(grad_vF1, [d1, k1, nf1]);%grad_F1;
Grad{2} = reshape(grad_vF2, [d2, k2, nf2]);%grad_F2;
Grad{3} = grad_W;
end

%% backward pass
function Grad = ComputeGradient(X_batch, Ys_batch, ConvNet, MFs)
[~, n] = size(Ys_batch);
grad_W = 0;
grad_vF1 = 0;
grad_vF2 = 0;
% forward pass 
[X_batch_1, X_batch_2, P_batch] = ForwardPass(X_batch, ConvNet, MFs);
% backward pass
G_batch = P_batch - Ys_batch;
grad_W = (G_batch * X_batch_2.') / n; 

% propagate 1
G_batch = ConvNet.W.'* G_batch;
h = X_batch_2~=0;
G_batch = G_batch .* h;
%indX2 = max(X_batch_2, 0);
%G_batch = G_batch .* spones(indX2);

[d1, k1, nf1] = size(ConvNet.F{1});% 28 5 20
[d2, k2, nf2] = size(ConvNet.F{2});% 20 3 20
for j=1:n
    g_j = G_batch(:,j);
    x_j = X_batch_1(:,j);
    MX = MakeMXMatrix(x_j, d2, k2, nf2);%(nlen-k2+1)*nf2, k2*nf2*d2  260*1200
    v = g_j.' * MX;% 1* 1200
    grad_vF2 = grad_vF2 + v/n;
end

% propagate 2
%MFs{2} = MakeMFMatrix(ConvNet.F{2}, 15);
G_batch = MFs{2}.' * G_batch;
h = X_batch_1 ~= 0;
G_batch = G_batch .* h;
%indX1 = max(X_batch_1, 0);
%G_batch = G_batch .* spones(indX1);

for j=1:n
    g_j = G_batch(:,j);
    x_j = X_batch(:,j);
    MX = MakeMXMatrix(x_j, d1, k1, nf1);%(nlen-k1+1)*nf1, k1*nf1*d1 300*2800
    % MX can be pre-computated
    v = g_j.' * MX;% 1 * 2800
    grad_vF1 = grad_vF1 + v/n;% 1 * 2800
end
Grad{1}= reshape(grad_vF1, [d1, k1, nf1]);%grad_F1;
Grad{2} = reshape(grad_vF2, [d2, k2, nf2]);%grad_F2;
Grad{3} = grad_W;
end

%% tools
function [eta, rho, ConvNet] = Initialize(d, n1, n2, k1, k2, fsize)
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

function MF = MakeMFMatrix(F, nlen)
[dd, k, nf] = size(F);% nf: number of filters
% VF: nf * (dd*k) matrix
VF = zeros(nf, dd*k);
for j=1:nf
    f = F(:,:,j);
    f = f(:).';
    VF(j,:) = f;
end
MF = zeros((nlen-k+1)*nf, nlen*dd);
for i=1:(nlen-k+1)
    %i_s = (i-1)*nf+1;
    MF((i-1)*nf+1:i*nf, (i-1)*dd+1: (i+k-1)*dd) = VF;
end
MF = sparse(MF);
end

function MX = MakeMXMatrix(x_input, d, k, nf)
% X_input has size d * nlen
% can get nlen by dividing the length of x_input by d
nlen = size(x_input,1) / d;
X = reshape(x_input, [d, nlen]);
%MX = zeros((nlen-k+1)*nf, k*nf*d);% * 10080
I = eye(nf);
for i=1:(nlen-k+1)
    x = X(:, i:i+k-1);%d*k
    vx = x(:).';%d*k*1
    %M1 = kron(I, vx);
    %if i==1
       % MX = M1;
    %else
        %MX = [MX; M1];
    %end
    for j=1:nf
        MX((i-1)*nf + j, (j-1)*d*k+1:j*d*k) = vx;
    end
end
MX = sparse(MX);
end

function error = ComputeDiff(ga, gn, flag)
eps = 1e-5;
error = zeros(size(ga));
if flag==1
[ii, jj] = size(ga);
for i=1: ii
    for j=1:jj
        error(i, j) = abs(ga(i, j) - gn(i, j))/max(eps, (abs(ga(i, j)) + abs(gn(i, j))));
    end
end
end

if flag~=1
[ii, jj, kk] = size(ga);
for i=1: ii
    for j=1:jj
        for k=1:kk
        error(i, j, k)= abs(ga(i, j, k) - gn(i, j, k))/max(eps, (abs(ga(i, j, k)) + abs(gn(i, j, k))));
        end
    end
end
end
end
