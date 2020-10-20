load('names_data.mat', 'validXx','trainXx','trainY','trainYs', 'validYs', 'validY');%trainX: 28*19 x 19798, validX: 28*19 x 252
load('classind.mat', 'Class')

d = 28;%dim of one-hot vector, existed alphabet
n_len = 19;%19 maximum length of name
K = 18;%18 classes

n1 = 20;% n1:nr of filters applied at layer 1 
n2 = 20;% n2:nr of filters applied at layer 2
k1 = 5;% k1: width of the filters applied at layer 1(with size d*k1)
k2 = 3;% layer 2 filter n1*k2
n_len1 = n_len - k1 + 1;
fsize = (n_len - k1 - k2 + 2) * n2;% fsize: number of elements in X(2) in eq(4)

%% mini batch training
%profile on
%ConvNet = MiniBatch(trainXx, trainY, validXx, validY, validYs.', 100, 500, 25000, ConvNet, eta, rho, n_len1);
%load('model3.mat','ConvNet');

eta=0.0006561;%0.00081;
%[eta, rho, ConvNet] = Initialize(d, n1, n2, k1, k2, fsize);
%[ConvNet,valid_loss] = MiniBatchBalance(trainXx, trainY, validXx, validY, validYs.', 100, 500, 400, ConvNet, eta, 0.9, n_len1);
%save('model4.mat', 'ConvNet','valid_loss')
%profile viewer
load('model4.mat','ConvNet');
names = {'xiong', 'gagos', 'corbetta','tump','vincent','alquezar','miyazaki'};
[C, MP] = Test(names, d, n_len, n_len1, ConvNet);
%[C, P] = Test('gagos', d, n_len, n_len1, ConvNet);


%% MiniBatch
function [ConvNet,valid_loss,valid_acc]  = MiniBatch(X, Ys, X_val, Y_val, Ys_val, n_batch, n_update, n_steps, ConvNet, eta, rho, n_len1)
N = size(X,2);
nb = floor(N/n_batch);
n_len = 19;
MFs{1} = MakeMFMatrix(ConvNet.F{1}, n_len);
MFs{2} = MakeMFMatrix(ConvNet.F{2}, n_len1);
valid_loss = ComputeLoss(X_val, Y_val, ConvNet.W, MFs);
iup = 2;
[d1, k1, nf1] = size(ConvNet.F{1});
%X_1 = PreMX(X, d1, k1, nf1);
%save('X_1_1.mat', 'X_1');
load('X_1_1.mat', 'X_1');
for i=1:n_steps
    if i<nb
        j = i;
    else
        j = rem(i, nb) + 1;
    end
    j_start = (j-1)*n_batch+1;
    j_end = j * n_batch;
    X_batch = X(:,j_start:j_end);
    Y_batch = Ys(:,j_start:j_end);
    MX_1 = X_1(j_start:j_end);
    Grad = ComputeGradients(X_batch, Y_batch, ConvNet, MFs, MX_1);
    ConvNet.F{1} = ConvNet.F{1} - eta * Grad{1};
    ConvNet.F{2} = ConvNet.F{2} - eta * Grad{2};
    ConvNet.W = ConvNet.W - eta * Grad{3};
    MFs{1} = MakeMFMatrix(ConvNet.F{1}, n_len);
    MFs{2} = MakeMFMatrix(ConvNet.F{2}, n_len1);
    if rem(i,n_update) == 0
        iter = i/n_update
        valid_loss(iup) = ComputeLoss(X_val, Y_val, ConvNet.W, MFs);
        %[valid_acc(iup), M] = ConfusionMatrix(X_val, Ys_val, ConvNet.W, MFs);
        %figure
        %imagesc(M)
        %title(['Confustion Matrix, Acc=',num2str(valid_acc(iup))])
        iup = iup+1;
        if rem(i/n_update,5) == 0
            eta = eta * 0.9;
        end
    end
end
x = linspace(0, size(valid_loss,2)-1,size(valid_loss,2)) * n_update;
[valid_acc, M] = ConfusionMatrix(X_val, Ys_val, ConvNet.W, MFs);
figure
subplot(1,2,1)
plot(x, valid_loss);
ylabel('loss')
xlabel('steps')
title('Validation loss')
subplot(1,2,2)
imagesc(M)
title(['Acc=',num2str(valid_acc)])
end


function [trainX, trainY] = RandomSample(X, Y, num, d1, k1, nf1)
load('classind.mat', 'Class');
for i=1:18
    classind = Class{i};
    inds = randsample(classind(2)-classind(1)+1, num) + classind(1)-1;
    trainX(:,(i-1)*num+1:i*num) = X(:,inds);
    trainY(:,(i-1)*num+1:i*num) = Y(:,inds);
end
% shuffle
n = size(trainY,2);
p = randperm(n);
trainX = trainX(:,p);
trainY = trainY(:,p);
X_1 = PreMX(trainX, d1, k1, nf1);
save('X_1_rs.mat', 'X_1');
end

function [ConvNet,valid_loss]  = MiniBatchBalance(X, Y, X_val, Y_val, Ys_val, n_batch, n_update, n_epochs, ConvNet, eta, rho, n_len1)
n_len = 19;
MFs{1} = MakeMFMatrix(ConvNet.F{1}, n_len);
MFs{2} = MakeMFMatrix(ConvNet.F{2}, n_len1);
valid_loss = ComputeLoss(X_val, Y_val, ConvNet.W, MFs);
iup = 2;
[d1, k1, nf1] = size(ConvNet.F{1});
N = 56*18;
nb = floor(N/n_batch);
for i=1:n_epochs
    [trainX, trainY] = RandomSample(X, Y, 56, d1, k1, nf1);
    load('X_1_rs.mat', 'X_1');
    for j=1:nb
        j_start = (j-1)*n_batch+1;
        j_end = j * n_batch;
        X_batch = trainX(:,j_start:j_end);
        Y_batch = trainY(:,j_start:j_end);
        MX_1 = X_1(j_start:j_end);
        Grad = ComputeGradients(X_batch, Y_batch, ConvNet, MFs, MX_1);
        ConvNet.F{1} = ConvNet.F{1} - eta * Grad{1};
        ConvNet.F{2} = ConvNet.F{2} - eta * Grad{2};
        ConvNet.W = ConvNet.W - eta * Grad{3};
        MFs{1} = MakeMFMatrix(ConvNet.F{1}, n_len);
        MFs{2} = MakeMFMatrix(ConvNet.F{2}, n_len1);
    end
    if rem(i*nb, n_update) == 0
        iter = i
        valid_loss(iup) = ComputeLoss(X_val, Y_val, ConvNet.W, MFs);
        %[valid_acc(iup), M] = ConfusionMatrix(X_val, Ys_val, ConvNet.W, MFs);
        %figure
        %imagesc(M)
        %title(['Confustion Matrix, Acc=',num2str(valid_acc(iup))])
        iup = iup+1;
        if rem(i/n_update,5) == 0
            eta = eta * 0.9;
        end
    end
end
x = linspace(0, size(valid_loss,2)-1,size(valid_loss,2)) * n_update;
[valid_acc, M] = ConfusionMatrix(X_val, Ys_val, ConvNet.W, MFs);
save('Mm.mat','M')
figure
subplot(1,2,1)
plot(x, valid_loss);
ylabel('loss')
xlabel('steps')
title('Validation loss')
subplot(1,2,2)
imagesc(M)
title(['Acc=',num2str(valid_acc)])
end


%% confusion matrix
function [acc, M] = ConfusionMatrix(X, Ys, W, MFs)
K=18;
M = zeros(K, K);% K:#classes
% Mij: number of examples with label i that are classified as class j
[~, ~, P] = ForwardPass(X, W, MFs);
for i=1:size(Ys,2)
    a = find(P(:, i) == max(P(:, i))) ;
    M(Ys(i), a) = M(Ys(i), a) + 1;
end
acc = trace(M) / sum(sum(M));
end

function loss = ComputeLoss(X_batch, Ys_batch, W, MFs)
% forward pass
[~, ~, P_batch] = ForwardPass(X_batch, W, MFs);
% loss
[~, n] = size(Ys_batch);
loss = 0;
for i=1:n
    loss = loss - log(Ys_batch(:,i).' * P_batch(:,i));
end
loss = loss/n;
end

function X_1 = PreMX(X, d, k, nf)
% X_input has size d * nlen
% can get nlen by dividing the length of x_input by d
nlen = size(X,1) / d;
N = size(X, 2);
for ii=1:N
    X_ii = reshape(X(:,ii), [d, nlen]);
    MX = zeros((nlen-k+1)*nf, k*nf*d);
    for i=1:(nlen-k+1)
        x = X_ii(:, i:i+k-1);%d*k
        vx = x(:).';%d*k*1
        for j=1:nf
            MX((i-1)*nf + j, (j-1)*d*k+1:j*d*k) = vx;
        end
    end
    X_1{ii} = sparse(MX);
end
end

%% Gradient update
function Grad = ComputeGradients(X_batch, Ys_batch, ConvNet, MFs, MX_1)
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
    %x_j = X_batch(:,j);
    %MX = MakeMXMatrix(x_j, d1, k1, nf1);%(nlen-k1+1)*nf1, k1*nf1*d1 300*2800
    %v = g_j.' * MX;% 1 * 2800
    % MX can be pre-computated
    v = g_j.' * MX_1{j};
    grad_vF1 = grad_vF1 + v/n;% 1 * 2800
end
Grad{1}= reshape(grad_vF1, [d1, k1, nf1]);%grad_F1;
Grad{2} = reshape(grad_vF2, [d2, k2, nf2]);%grad_F2;
Grad{3} = grad_W;
end

function [X_batch_1, X_batch_2, P_batch] = ForwardPass(X_batch, W, MFs)
% forward pass
X_batch_1 = max(MFs{1} * X_batch, 0);
X_batch_2 = max(MFs{2} * X_batch_1, 0);
S_batch = W * X_batch_2;
P_batch = softmax(S_batch);
end

%% tools
function MF = MakeMFMatrix(F, nlen)
[dd, k, nf] = size(F);% nf: number of filters
% VF: nf * (dd*k) matrix
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

% M_{x, k, nf}
function MX = MakeMXMatrix(x_input, d, k, nf)
% X_input has size d * nlen
% can get nlen by dividing the length of x_input by d
nlen = size(x_input,1) / d;
X = reshape(x_input, [d, nlen]);
MX = zeros((nlen-k+1)*nf, k*nf*d);% * 10080
for i=1:(nlen-k+1)
    x = X(:, i:i+k-1);%d*k
    vx = x(:).';%d*k*1
    %vx = x_input((i-1)*d+1:(i-1)*d+d*k);
    %MX((i-1) + j, (j-1)*d*k+1:j*d*k) = vx;
    for j=1:nf
        MX((i-1)*nf + j, (j-1)*d*k+1:j*d*k) = vx;
    end
end
%MX = sparse(MX);
end

% M_{x,k}
function MX = MakeMX(x_input, d, k)
%input n_1 x n_len1, filters of width k2
nlen = size(x_input,1) / d;
X = reshape(x_input, [d, nlen]);
%MX = zeros((nlen-k+1)*nf, k*nf*d);% * 10080
for i=1:(nlen-k+1)
    x = X(:, i:i+k-1);%d*k
    vx = x(:).';
    if i==1
        MX = vx;
    else
        MX = [MX;vx];
    end
end
end

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

function [C, MP] = Test(names, d, n_len, n_len1, ConvNet)
load('c2i.mat','char_to_ind');
load('classind.mat','country')
N = size(names,2);
X = zeros(d * n_len, N);
MP = zeros(18, N);
for i=1:N
    name = names{i};
    x_input = name(:);
    x = zeros(1, n_len);
    for j=1:length(x_input)
        x(1, j) = char_to_ind(x_input(j));   
    end
    onehot = bsxfun(@eq, x(:), 1:d).';%28*19
    %inds = (i-1)*d+1 : i*d;
    X(:, i) = onehot(:);
    MFs{1} = MakeMFMatrix(ConvNet.F{1}, n_len);
    MFs{2} = MakeMFMatrix(ConvNet.F{2}, n_len1);
    X_1 = max(MFs{1} * X(:,i), 0);
    X_2 = max(MFs{2} * X_1, 0);
    S = ConvNet.W * X_2;
    P = softmax(S);
    MP(:,i) = P.';
    c = find(P == max(P));
    C = country{c};
    text = "The country of " + name + " is " + C
end
end


%% PrepareData
% load('assignment3_names.mat')
% % all_names: 1*20050 cell
% % ys: 20050*1 double
% %% get a vector containing the unique characters
% C = unique(cell2mat(all_names));
% d = numel(C);%28 dim of one-hot vector, existed alphabet
% n_len = max(strlength(all_names));%19 maximum length of name
% CC = unique(ys);
% k = numel(CC);%18 classes
% 
% %char_to_ind = containers.Map('KeyType','char','ValueType','int32');
% Key = num2cell(C);
% Val = int32(1:d);
% char_to_ind = containers.Map(Key, Val);
% y = bsxfun(@eq, ys(:), 1:max(ys)).';%k*N
% [~, N] = size(y);
% 
% %% encode your input name
% % d * n_len input matrix X
% % has at most 1 non-zeros element per column
% % N: #names in the datasets
% % d: dim of one-hot vector (Background 1)
% %X = cell(N, 1);%(d * n_len, N);
% Xx = zeros(d*n_len, N);
% X = zeros(d, n_len, N);
% for i=1:N
%     name = all_names{i};
%     name = name(:);
%     x = zeros(1, n_len);
%     for j=1:length(name)
%         x(1, j) = char_to_ind(name(j));   
%     end
%     onehot = bsxfun(@eq, x(:), 1:d).';%28*19
%     %inds = (i-1)*d+1 : i*d;
%     Xx(:, i) = onehot(:);
%     X(:, :, i) = onehot;
% end
% 
% 
% %% split into train and valid set
% valid = fopen('Validation_Inds.txt', 'r');
% V = fscanf(valid, '%u');%252*1
% validX = X(:,:,V);%252*1cell, 28*19
% validY = y(:,V);%19*252
% validYs = ys(V);
% validXx = Xx(:, V);
% 
% T = 1:N;
% T(V) = 0;
% T = nonzeros(T.');
% trainX = X(:,:,T);%19798*1cell, 28*19
% trainXx = Xx(:, T);
% trainY = y(:,T);%19 * 19798
% trainYs = ys(T);
% 
% save('names_data.mat', 'validXx','trainXx','trainY', 'trainYs', 'validYs', 'validY');
% %save('names.mat', 'trainY', 'validY');
% save('c2i.mat','char_to_ind')
% 
% Class{1} = [1 1986];
% Class{2} = [1987 2240];
% Class{3} = [2241 2745];
% Class{4} = [2746 3028];
% Class{5} = [3029 6682];
% Class{6} = [6683 6945];
% Class{7} = [6946 7655];
% Class{8} = [7656 7844];
% Class{9} = [7845 8062];
% Class{10} = [8063 8757];
% Class{11} = [8758 9734];
% Class{12} = [9735 9814];
% Class{13} = [9815 9939];
% Class{14} = [9940 9999];
% Class{15} = [10000 19369];
% Class{16} = [19370 19455];
% Class{17} = [19456 19739];
% Class{18} = [19740 19798];
% 
% country{1} = 'Arabic';
% country{2} = 'Chinese';
% country{3} = 'Czech';
% country{4} = 'Dutch';
% country{5} = 'English';
% country{6} = 'French';
% country{7} = 'German';
% country{8} = 'Greek';
% country{9} = 'Irish';
% country{10} = 'Italian';
% country{11} = 'Japanese';
% country{12} = 'Korean';
% country{13} = 'Polish';
% country{14} = 'Portuguese';
% country{15} = 'Russian';
% country{16} = 'Scottish';
% country{17} = 'Spanish';
% country{18} = 'Vietnamese';
% 
% save('classind.mat', 'Class','country')