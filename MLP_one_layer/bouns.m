addpath DirName/Datasets/cifar-10-batches-mat/

[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');%d*N, K*N, N*1
[validX, validY, validy] = LoadBatch('data_batch_2.mat');%d*N, K*N, N*1
[testX, testY, testy] = LoadBatch('test_batch.mat');%d*N, K*N, N*1
% initial W and b
%W = randn(10,3072);%K*d
%b = randn(10,1);%K*1
[d, N] = size(trainX);
W = normrnd(0, 0.01, [10,d]);%K*d
b = normrnd(0, 0.01, [10,1]);%K*1

%[Wstar, bstar] = MiniBatchGD(trainX, trainY, validX, validY, testX, testy, 100, 0.01, 40, W, b, 0);
%% for ex2.1
rng(400)
%method_b(trainX, trainY, validX, validY, testX, testy, 100, 0.01, 200, W, b, 0)
%method_d(trainX, trainY, validX, validY, testX, testy, 100, 0.01, 40, W, b, 0)
%method_g(trainX, trainY, validX, validY, testX, testy, 100, 0.01, 40, W, b, 0)

%% for ex2.2
%exercise2(trainX, trainY, trainy, validX, validY, validy, testX, testy, 100, 0.01, 50, W, b, 0)
%exercise2(trainX, trainY, trainy, validX, validY, validy, testX, testy, 100, 0.01, 50, W, b, 0.01)
%exercise2(trainX, trainY, trainy, validX, validY, validy, testX, testy, 100, 0.01, 50, W, b, 0.1)
%exercise2(trainX, trainY, trainy, validX, validY, validy, testX, testy, 100, 0.01, 50, W, b, 1)
exercise2(trainX, trainY, trainy, validX, validY, validy, testX, testy, 100, 0.1, 50, W, b, 0)
method(trainX, trainY, trainy, validX, validY, validy, testX, testy, 100, 0.1, 50, W, b, 0)

%% visualize the W
function visualization(Wstar)
figure
title("The learnt W matrix")
for i=1:10
    im = reshape(Wstar(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
for i=1:10
    subplot(4,5,i);
    imshow(s_im{i})
end
end

%% read data from CIFAR-10
function [X, Y, y] = LoadBatch(filename)
A = load(filename);
X = double(A.data');%d*N
X = X/255;
y = A.labels+1;%N*1
Y = bsxfun(@eq, y(:), 1:max(y)).';%K*N
end

%% evaluate the network
function P = EvaluateClassifier(X, W, b)
[~, n_b] = size(X);
s = W*X + b * ones(1,n_b);
P = softmax(s);%K*n(N)
end

%% cost function
function [J, loss] = ComputeCost(X, Y, W, b, lambda)
P = EvaluateClassifier(X, W, b);%K*n 
[~,n] = size(Y);
% % sum the diagonal of the loss matrix
% loss = -log(Y.'*P);%n*n
% J = 1/n * sum(diag(loss)) + lambda * sum(sum(W.^2));
% calculate it sample by sample
loss = 0;
for i=1:n
    loss = loss - log(Y(:,i).' * P(:,i));
end
J = loss / n + lambda * sum(sum(W.^2));

end

%% compute accuarcy
function acc = ComputeAccuracy(X, y, W, b)
[~,n] = size(X);
P = EvaluateClassifier(X, W, b);%K*n
[~, K] = max(P);%K:1*n
S = K.'- double(y);
acc = nnz(~S)/n;
end

%% gradients of cost function for a mini-batch
function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
G = - Y + P;%batch K*N
[~, n_b] = size(X);%3072*10000(100)
grad_W = 1/n_b * G * X.'+ 2 * lambda * W; %K*d
grad_b = 1/n_b * G * ones(n_b,1);% eye(n_b); %K*1
end

%% difference between numerically and analytically computed gradient
function error = ComputeDiff(ga, gn)
eps = 1e-5;
[ii, jj] = size(ga);
error = zeros(size(ga));
for i=1: ii
    for j=1:jj
        error(i, j)= abs(ga(i, j) - gn(i, j))/max(eps, (abs(ga(i, j)) + abs(gn(i, j))));
    end
end
end

%% numerical gradient computation(based on the finite difference method)
function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

c = ComputeCost(X, Y, W, b, lambda);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c) / h;
end

for i=1:numel(W)   
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c) / h;
end
end

%% numerical gradient computation(based on the centered difference formula)
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) - h;
    c1 = ComputeCost(X, Y, W, b_try, lambda);
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c1) / (2*h);
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    c1 = ComputeCost(X, Y, W_try, b, lambda);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c1) / (2*h);
end
end

%% mini-batch gradient descent algorithm
%function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)
function [Wstar, bstar] = MiniBatchGD(X, Y, X_val, Y_val, X_test, y_test, n_batch, eta, n_epochs, W, b, lambda)
train_loss = zeros(size(n_epochs));
valid_loss = zeros(size(n_epochs));
Wstar = W;
bstar = b;
N=10000;
for i=1:n_epochs
    %generate the set of mini-batches
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end; %?
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);

        P = EvaluateClassifier(Xbatch, Wstar, bstar);
        [ngrad_W, ngrad_b] = ComputeGradients(Xbatch, Ybatch, P, Wstar, lambda);
        Wstar = Wstar - eta * ngrad_W;
        bstar = bstar - eta * ngrad_b;
    end
    [J,~] = ComputeCost(X, Y, Wstar, bstar, 0);
    train_loss(i) = J;
    [J2, ~] = ComputeCost(X_val, Y_val, Wstar, bstar, 0);
    valid_loss(i) = J2;
end
figure
plot(train_loss)
hold on
plot(valid_loss)
hold off
legend('training loss', 'validation loss');%,'validation loss',2);
acc = ComputeAccuracy(X_test, y_test, Wstar, bstar);
title(['lambda=', num2str(lambda), ' epochs=', num2str(n_epochs), ' batch=100',' eta=', num2str(eta), ' Accuarcy = ', num2str(acc)]);
end


%% 2.1(b) "early stopping"
function method_b(X, Y, X_val, Y_val, X_test, y_test, n_batch, eta, n_max, W, b, lambda)
%train_loss = zeros(size(n_epochs));
%valid_loss = zeros(size(n_epochs));
Wstar = W;
bstar = b;
N=10000;
for i=1:n_max
    %generate the set of mini-batches
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end; %?
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);

        P = EvaluateClassifier(Xbatch, Wstar, bstar);
        [ngrad_W, ngrad_b] = ComputeGradients(Xbatch, Ybatch, P, Wstar, lambda);
        Wstar = Wstar - eta * ngrad_W;
        bstar = bstar - eta * ngrad_b;
    end
    [J,~] = ComputeCost(X, Y, Wstar, bstar, 0);
    train_loss(i) = J;
    [J2, ~] = ComputeCost(X_val, Y_val, Wstar, bstar, 0);
    valid_loss(i) = J2;
    if (i>10) && (valid_loss(i)>=valid_loss(i-1)) && (valid_loss(i-1)>=valid_loss(i-2) && (valid_loss(i-2)>=valid_loss(i-3)))
        n_epochs = i;
        break
    end
end
figure
plot(train_loss)
hold on
plot(valid_loss)
hold off
legend('training loss', 'validation loss');%,'validation loss',2);
acc = ComputeAccuracy(X_test, y_test, Wstar, bstar);
xlabel('epoch')
ylabel('cross-entropy loss')
title(['lambda=', num2str(lambda), ' epochs=', num2str(n_epochs), ' batch=100',' eta=', num2str(eta), ' Accuarcy = ', num2str(acc)]);
end

%% 2.1(d) learning rate decay
function method_d(X, Y, X_val, Y_val, X_test, y_test, n_batch, eta, n_epochs, W, b, lambda)
train_loss = zeros(size(n_epochs));
valid_loss = zeros(size(n_epochs));
Wstar = W;
bstar = b;
N=10000;
for i=1:n_epochs
    %generate the set of mini-batches
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end; %?
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);

        P = EvaluateClassifier(Xbatch, Wstar, bstar);
        [ngrad_W, ngrad_b] = ComputeGradients(Xbatch, Ybatch, P, Wstar, lambda);
        Wstar = Wstar - eta * ngrad_W;
        bstar = bstar - eta * ngrad_b;
    end
    [J,~] = ComputeCost(X, Y, Wstar, bstar, 0);
    train_loss(i) = J;
    [J2, ~] = ComputeCost(X_val, Y_val, Wstar, bstar, 0);
    valid_loss(i) = J2;
    eta = eta * 0.9;
end
figure
plot(train_loss)
hold on
plot(valid_loss)
hold off
legend('training loss', 'validation loss');%,'validation loss',2);
acc = ComputeAccuracy(X_test, y_test, Wstar, bstar);
xlabel('epoch')
ylabel('cross-entropy loss')
title(['lambda=', num2str(lambda), ' epochs=', num2str(n_epochs), ' batch=100', ' Accuarcy = ', num2str(acc)]);
end

%% 2.1(g) shuffle training examples at the begining of every epoch
function method_g(X, Y, X_val, Y_val, X_test, y_test, n_batch, eta, n_epochs, W, b, lambda)
train_loss = zeros(size(n_epochs));
valid_loss = zeros(size(n_epochs));
Wstar = W;
bstar = b;
N=10000;
for i=1:n_epochs
    %shuffle trainX
    ord = randperm(size(X,2));
    X = X(:,ord);
    Y = Y(:,ord);
    %generate the set of mini-batches
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end; %?
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);

        P = EvaluateClassifier(Xbatch, Wstar, bstar);
        [ngrad_W, ngrad_b] = ComputeGradients(Xbatch, Ybatch, P, Wstar, lambda);
        Wstar = Wstar - eta * ngrad_W;
        bstar = bstar - eta * ngrad_b;
    end
    [J,~] = ComputeCost(X, Y, Wstar, bstar, 0);
    train_loss(i) = J;
    [J2, ~] = ComputeCost(X_val, Y_val, Wstar, bstar, 0);
    valid_loss(i) = J2;
end
figure
plot(train_loss)
hold on
plot(valid_loss)
hold off
legend('training loss', 'validation loss');%,'validation loss',2);
acc = ComputeAccuracy(X_test, y_test, Wstar, bstar);
xlabel('epoch')
ylabel('cross-entropy loss')
title(['lambda=', num2str(lambda), ' epochs=', num2str(n_epochs), ' batch=100',' eta=', num2str(eta), ' Accuarcy = ', num2str(acc)]);
end

%% compare
function method(X, Y, y, X_val, Y_val, y_val, X_test, y_test, n_batch, eta, n_epochs, W, b, lambda)
train_loss = zeros(size(n_epochs));
valid_loss = zeros(size(n_epochs));
train_acc = zeros(size(n_epochs));
valid_acc = zeros(size(n_epochs));
Wstar = W;
bstar = b;
N=10000;
for i=1:n_epochs
    %generate the set of mini-batches
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end; %?
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);

        P = EvaluateClassifier(Xbatch, Wstar, bstar);
        [ngrad_W, ngrad_b] = ComputeGradients(Xbatch, Ybatch, P, Wstar, lambda);
        Wstar = Wstar - eta * ngrad_W;
        bstar = bstar - eta * ngrad_b;
    end
    [J,~] = ComputeCost(X, Y, Wstar, bstar, 0);
    train_loss(i) = J;
    [J2, ~] = ComputeCost(X_val, Y_val, Wstar, bstar, 0);
    valid_loss(i) = J2;
    %train_acc(i) = ComputeAccuracy(X, y, Wstar, bstar);
    %valid_acc(i) = ComputeAccuracy(X, y_val, Wstar, bstar);
    eta = eta * 0.9;
end
figure
plot(train_loss)
hold on
plot(valid_loss)
hold off
legend('training loss', 'validation loss');%,'validation loss',2);
acc = ComputeAccuracy(X_test, y_test, Wstar, bstar);
xlabel('epoch')
ylabel('cross-entropy loss')
title(['lambda=', num2str(lambda), ' epochs=', num2str(n_epochs), ' batch=100', ' Accuarcy = ', num2str(acc)]);
end

%% 2.2 SVM multi-class loss 
function HL = ForwardPass(X, Y, W, b)
S = W*X + b * ones(1,size(X,2));%K*N
K = 10;
n = size(X,2);
z = zeros(size(S));
s = S - ones(K,K) * (Y.* S) + ones(K, n); %K*n
HL = max(z, s) - Y;%K*n
end

function J = ComputeSVMCost(X, Y, W, b, lambda)
HL = ForwardPass(X, Y, W, b);%K*N(n)
n = size(X,2);
% z = zeros(K, n);
% s = S.' - ((Y.* S)*ones(n,n)).' + ones(n,K); %N*K
% loss = max(z, s.') - Y.* ones(K, n);
J = sum(HL(:)) / n + lambda * sum(sum(W.^2));
end

function [grad_W, grad_b] = ComputeSVMGradients(X, Y, loss, W, b, lambda)
[~, n_b] = size(X);%3072*10000(100)
% grad_W = 1/n_b * G * X.'+ 2 * lambda * W; %K*d
% grad_b = 1/n_b * G * ones(n_b,1);% eye(n_b); %K*1
%grad_W = zeros(size(Y,1), size(X,1));%K*d
%grad_b = zeros(size(Y,1),1);%K*1
loss = loss ~= 0;% set all non-zero element to 1 K*n
nzs = sum(loss,1); %1*n vector containing the sum of the row
G = loss - nzs .* Y;%K*n
grad_W = 1/n_b * G * X.' + 2 * lambda * W;%K*d
grad_b = 1/n_b * G * ones(n_b,1);%K*1
end

function exercise2(X, Y, y, X_val, Y_val, y_val, X_test, y_test, n_batch, eta, n_epochs, W, b, lambda)
train_loss = zeros(size(n_epochs));
valid_loss = zeros(size(n_epochs));
train_acc = zeros(size(n_epochs));
valid_acc = zeros(size(n_epochs));
Wstar = W;
bstar = b;
Wstar_ = W;
bstar_ = b;
N=10000;
for i=1:n_epochs
    %generate the set of mini-batches
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end; %?
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);
        
        HL = ForwardPass(Xbatch, Ybatch, Wstar, bstar);
        [ngrad_W, ngrad_b] = ComputeSVMGradients(Xbatch, Ybatch, HL, Wstar, bstar, lambda);
        Wstar = Wstar - eta * ngrad_W;
        bstar = bstar - eta * ngrad_b;
    end
    J = ComputeSVMCost(X, Y, Wstar, bstar, 0);
    train_loss(i) = J;
    J2 = ComputeSVMCost(X_val, Y_val, Wstar, bstar, 0);
    valid_loss(i) = J2;
    eta = eta * 0.9;
    %train_acc(i) = ComputeAccuracy(X, y, Wstar, bstar);
    %valid_acc(i) = ComputeAccuracy(X, y_val, Wstar, bstar);
end
figure
plot(train_loss)
hold on
plot(valid_loss)
hold off
legend('training SVM loss', 'validation SVM loss');%,'validation loss',2);
acc = ComputeAccuracy(X_test, y_test, Wstar, bstar);
title(['lambda=', num2str(lambda), ' epochs=', num2str(n_epochs), ' batch=100', ' Accuracy = ',num2str(acc)]);%,' eta=', num2str(eta), ' Accuarcy = ', num2str(acc)]);
% figure
% plot(train_acc)
% hold on
% plot(valid_acc)
% hold off
% legend('training accuracy', 'validation accuracy');
% title(['Accuracy = ',num2str(acc)])
end
