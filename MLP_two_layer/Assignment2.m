addpath DirName/Datasets/cifar-10-batches-mat/
%rng(400)

%% check with e.g.trainX(1:20,1:2), trainY(:, 1:2) and W(:, 1:20)
% [trainX, trainY, trainy, validX, validY, validy, testX, testy, W, b] = GetData(1);
% [P, h] = ForwardPass(trainX(1:20,1:2), {W{1}(:, 1:20),W{2}}, b);
% [grad_W, grad_b] = ComputeGradients(trainX(1:20,1:2), trainY(:, 1:2), P, h, {W{1}(:, 1:20),W{2}}, 0);
% [grad_b_1, grad_W_1] = ComputeGradsNum(trainX(1:20,1:2), trainY(:, 1:2),{W{1}(:, 1:20),W{2}}, b, 0, 1e-5);
% [grad_b_2, grad_W_2] = ComputeGradsNumSlow(trainX(1:20,1:2), trainY(:, 1:2), {W{1}(:, 1:20),W{2}}, b, 0, 1e-5);
% maxerror_w_1 = max(max(ComputeDiff(grad_W{1}, grad_W_1{1})));
% maxerror_w_2 = max(max( ComputeDiff(grad_W{1}, grad_W_2{1})));
% maxerror_b_1 = max(ComputeDiff(grad_b{1}, grad_b_1{1}));
% maxerror_b_2 = max(ComputeDiff(grad_b{1}, grad_b_2{1}));
% maxerror_w_1a = max(max(ComputeDiff(grad_W{2}, grad_W_1{2})));
% maxerror_w_2a = max(max(ComputeDiff(grad_W{2}, grad_W_2{2})));
% maxerror_b_1a = max(ComputeDiff(grad_b{2}, grad_b_1{2}));
% maxerror_b_2a = max(ComputeDiff(grad_b{2}, grad_b_2{2}));

%[Wstar, bstar] = MiniBatchGDtry(trainX(:,1:100), trainY(:,1:100), trainy(1:100), validX(:,1:100), validY(:,1:100), validy(1:100), testX(:,1:100), testy(1:100), 100, 0.01, 200, W, b, 0);

%% train with cyclical learning rate
% Eta = CyclicalEta(1e-5, 1e-1, 500, 1);
% figure
% plot(Eta)
% [trainX, trainY, trainy, validX, validY, validy, testX, testy, W, b] = GetData(1);
% [train_loss, valid_loss, acc] = MiniBatchGD(trainX, trainY, trainy, validX, validY, validy, testX, testy, 100, 500, 1, W, b, 0.01, 1);

%% train your network for real
%[trainX, trainY, trainy, validX, validY, validy, testX, testy, W, b] = GetData(1);
%best_score = MiniBatchGD(trainX, trainY, trainy, validX, validY, validy, testX, testy, 100, 800, 3, W, b, 0.01, 1);
%[trainX, trainY, trainy, validX, validY, validy, testX, testy, W, b] = GetData(2);
% coarse search
%all_para = SearchLambda(-6,-2, 100, 2, trainX, trainY, trainy, validX, validY, validy, testX, testy, W, b);
%save('all_para');
% fine search
%alla_para = SearchLambda(-5,-3, 100, 3, trainX, trainY, trainy, validX, validY, validy, testX, testy, W, b);
%save('alla_para')
% train with best settings
%[trainX, trainY, trainy, validX, validY, validy, testX, testy, W, b] = GetData(3);
%[train_loss, valid_loss, acc] = MiniBatchGD(trainX, trainY, trainy, validX, validY, validy, testX, testy, 100, 900, 3, W, b, 9.87e-4, 1);

%% Exercise 1: Read the data & initialize parameters
function [X, Y, y] = LoadBatch(filename)
A = load(filename);
X = double(A.data');%d*N
X = X/255;
y = A.labels+1;%N*1
Y = bsxfun(@eq, y(:), 1:max(y)).';%K*N
% transform X to zero mean
mean_X = mean(X, 2);
std_X = std(X, 0, 2);
X = X - repmat(mean_X, [1, size(X, 2)]);
X = X ./ repmat(std_X, [1, size(X, 2)]);
end

function [W, b] = Initialize(K, d)
m = 50; % nr of nodes in the hidden layer
W1 = randn(m, d)/sqrt(d);%m*d
b1 = zeros(m, 1);
W2 = randn(K, m)/sqrt(m);%K*m
b2 = zeros(K, 1);
W = {W1, W2};
b = {b1, b2};% 1*2 cell array
end

%% compute the gradients for the network parameters
% forward pass
function [P, h] = ForwardPass(X, W, b)
[~, n_b] = size(X);
s1 = W{1} * X + b{1} * ones(1,n_b);%m*n
h = max(zeros(50,n_b), s1);%m*n
s = W{2} * h + b{2} * ones(1,n_b);%
P = softmax(s);%K*n(N)
end

% cost function
function [J, loss] = ComputeCost(X, Y, W, b, lambda)
[P, ~] = ForwardPass(X, W, b);%K*n 
[~,n] = size(Y);
% % sum the diagonal of the loss matrix
% loss = -log(Y.'*P);%n*n
% J = 1/n * sum(diag(loss)) + lambda * (sum(sum(W{1}.^2)) + sum(sum(W{2}.^2)));
%calculate it sample by sample
loss = 0;
for i=1:n
    loss = loss - log(Y(:,i).' * P(:,i));
end
J = loss / n + lambda * (sum(sum(W{1}.^2)) + sum(sum(W{2}.^2)));
end

% compute gradient for mini-batch algorithm
function [grad_W, grad_b] = ComputeGradients(X, Y, P, h, W, lambda)
G = - Y + P;%batch K*N
[~, n_b] = size(X);%d*n(100)
grad_W2 = 1/n_b * G * h.'+ 2 * lambda * W{2}; %K*m
grad_b2 = 1/n_b * G * ones(n_b,1); %K*1
GG = W{2}.' * G; %m*n
h = h ~= 0; %m*n
GG = GG .* h; %m*n
grad_W1 = 1/n_b * GG * X.' + 2 * lambda * W{1}; %m*d
grad_b1 = 1/n_b * GG * ones(n_b, 1);
grad_W = {grad_W1, grad_W2};
grad_b = {grad_b1, grad_b2};
end

% fast
function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

[c, ~] = ComputeCost(X, Y, W, b, lambda);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        [c2, ~] = ComputeCost(X, Y, W, b_try, lambda);
        grad_b{j}(i) = (c2-c) / h;
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})   
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        [c2, ~] = ComputeCost(X, Y, W_try, b, lambda);
        
        grad_W{j}(i) = (c2-c) / h;
    end
end
end

% slow
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end
end

% compute difference for gradient checking
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

%% Exercise3: train your network with cyclical learning rates
function Eta = CyclicalEta(eta_min, eta_max, n_s, n_c)
Eta = zeros(n_s * 2 * n_c);
for i=1:n_c
    for j=1:n_s
        k = 2 * n_s * (i-1) + j;
        Eta(k) = eta_min + j * (eta_max - eta_min)/n_s;
        Eta(k+n_s) = eta_max - j * (eta_max - eta_min)/n_s;
    end
end
end

% compute accuracy
function acc = ComputeAccuracy(X, y, W, b)
[~,n] = size(X);
[P,~] = ForwardPass(X, W, b);%K*n
[~, K] = max(P);%K:1*n
S = K.'- double(y);
acc = nnz(~S)/n;
end


function [Wstar, bstar] = MiniBatchGDtry(X, Y, y, X_val, Y_val, y_val, X_test, y_test, n_batch, eta, n_epochs, W, b, lambda)
train_loss = zeros(1,n_epochs);
valid_loss = zeros(1,n_epochs);
train_cost = zeros(1,n_epochs);
valid_cost = zeros(1,n_epochs);
train_acc = zeros(1,n_epochs);
valid_acc = zeros(1,n_epochs);
Wstar = W;
bstar = b;
N=size(X,2);
for i=1:n_epochs
    %generate the set of mini-batches
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end; 
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);

        [P,h] = ForwardPass(Xbatch, Wstar, bstar);
        [ngrad_W, ngrad_b] = ComputeGradients(Xbatch, Ybatch, P, h, Wstar, lambda);
        Wstar{1} = Wstar{1} - eta * ngrad_W{1};
        Wstar{2} = Wstar{2} - eta * ngrad_W{2};
        bstar{1} = bstar{1} - eta * ngrad_b{1};
        bstar{2} = bstar{2} - eta * ngrad_b{2};
    end
    [train_loss(i),~] = ComputeCost(X, Y, Wstar, bstar, 0);
    [valid_loss(i), ~] = ComputeCost(X_val, Y_val, Wstar, bstar, 0);
    [train_cost(i),~] = ComputeCost(X, Y, Wstar, bstar, lambda);
    [valid_cost(i), ~] = ComputeCost(X_val, Y_val, Wstar, bstar, lambda);
    train_acc(i) = ComputeAccuracy(X, y, Wstar, bstar);
    valid_acc(i) = ComputeAccuracy(X_val, y_val, Wstar, bstar);
end
figure
subplot(2,3,1)
plot(train_cost)
hold on
plot(valid_cost)
hold off
%ylim([0 inf])
legend('training', 'validation');
title('cost');
subplot(2,3,2)
plot(train_loss)
hold on
plot(valid_loss)
hold off
legend('training', 'validation');
%ylim([0 inf])
title('loss');
subplot(2,3,3)
acc = ComputeAccuracy(X_test, y_test, Wstar, bstar);
plot(train_acc)
hold on
plot(valid_acc)
hold off
%ylim([0 inf])
legend('training', 'validation');
title(['Test Accuracy = ', num2str(acc)]);
%suptitle(['lambda=', num2str(lambda), ' epochs=', num2str(n_epochs), ' batch=100'])
end


%% Exercise 4: Train your network for real
function [train_loss, valid_loss, acc] = MiniBatchGD(X, Y, y, X_val, Y_val, y_val, X_test, y_test, n_batch, n_s, n_c, W, b, lambda, show_flag)
eta = CyclicalEta(1e-5, 1e-1, n_s, n_c);
n_step = n_s * 2 * n_c;
N=size(X,2);
n_epochs = ceil(n_step/(N/n_batch));
wait = floor(n_s * 2 / 9);
len = 9 * n_c + 1; %28
Wstar = W;
bstar = b;
m=1;
if show_flag==1
    train_loss = zeros(1, len);
    valid_loss = zeros(1, len);
    train_cost = zeros(1, len);
    valid_cost = zeros(1, len);
    train_acc = zeros(1, len);
    % performance before training
    [train_loss(m),~] = ComputeCost(X, Y, Wstar, bstar, 0);
    [valid_loss(m), ~] = ComputeCost(X_val, Y_val, Wstar, bstar, 0);
    [train_cost(m),~] = ComputeCost(X, Y, Wstar, bstar, lambda);
    [valid_cost(m), ~] = ComputeCost(X_val, Y_val, Wstar, bstar, lambda);
    train_acc(m) = ComputeAccuracy(X, y, Wstar, bstar);
end
valid_acc = zeros(1, len);
valid_acc(m) = ComputeAccuracy(X_val, y_val, Wstar, bstar);
for i=1:n_epochs
    %generate the set of mini-batches
    for j=1:N/n_batch
        k = (i-1)*N/n_batch + j;%step
        if (k == (n_step+1))
            break
        else            
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            inds = j_start:j_end; 
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);

            [P, h] = ForwardPass(Xbatch, Wstar, bstar);
            [ngrad_W, ngrad_b] = ComputeGradients(Xbatch, Ybatch, P, h, Wstar, lambda);
            Wstar{1} = Wstar{1} - eta(k) * ngrad_W{1};
            Wstar{2} = Wstar{2} - eta(k) * ngrad_W{2};
            bstar{1} = bstar{1} - eta(k) * ngrad_b{1};
            bstar{2} = bstar{2} - eta(k) * ngrad_b{2};
            
            if(mod(k, wait) == 0 && show_flag==1)
                m = m + 1;
                [train_loss(m),~] = ComputeCost(X, Y, Wstar, bstar, 0);
                [valid_loss(m), ~] = ComputeCost(X_val, Y_val, Wstar, bstar, 0);
                [train_cost(m),~] = ComputeCost(X, Y, Wstar, bstar, lambda);
                [valid_cost(m), ~] = ComputeCost(X_val, Y_val, Wstar, bstar, lambda);
                train_acc(m) = ComputeAccuracy(X, y, Wstar, bstar);
                valid_acc(m) = ComputeAccuracy(X_val, y_val, Wstar, bstar);
            %elseif(mod(k, wait) == 0)
                %m = m + 1;
                %valid_acc(m) = ComputeAccuracy(X_val, y_val, Wstar, bstar);
            end
        end
    end
end
acc = ComputeAccuracy(X_test, y_test, Wstar, bstar);
if show_flag==1
    xx = linspace(0, n_step, len);
    figure
    subplot(2,3,1)
    plot(xx, train_cost)
    hold on
    plot(xx, valid_cost)
    hold off
    legend('training', 'validation')
    ylabel('cost');
    xlabel('steps')
    subplot(2,3,2)
    plot(xx, train_loss)
    hold on
    plot(xx, valid_loss)
    hold off
    legend('training', 'validation')
    ylabel('loss')
    xlabel('steps')
    subplot(2,3,3)
    acc = ComputeAccuracy(X_test, y_test, Wstar, bstar);
    plot(xx, train_acc)
    hold on
    plot(xx, valid_acc)
    hold off
    legend('training', 'validation')
    title(['Test Accuracy = ', num2str(acc)]);
    ylabel('accuracy')
    xlabel('steps')
end
best_score = max(valid_acc);
end

% get data and initalize parameters
function [trainX, trainY, trainy, validX, validY, validy, testX, testy, W, b] = GetData(flag)
if flag==1
    [trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');%d*N, K*N, N*1
    [validX, validY, validy] = LoadBatch('data_batch_2.mat');%d*N, K*N, N*1
    [testX, ~, testy] = LoadBatch('test_batch.mat');%d*N, K*N, N*1
else
    [trainX1, trainY1, trainy1] = LoadBatch('data_batch_1.mat');
    [trainX2, trainY2, trainy2] = LoadBatch('data_batch_2.mat');
    [trainX3, trainY3, trainy3] = LoadBatch('data_batch_3.mat');
    [trainX4, trainY4, trainy4] = LoadBatch('data_batch_4.mat');
    [trainX5, trainY5, trainy5] = LoadBatch('data_batch_5.mat');
    [testX, ~, testy] = LoadBatch('test_batch.mat');
    trainX = cat(2, trainX1, trainX2, trainX3, trainX4, trainX5);
    trainY = cat(2, trainY1, trainY2, trainY3, trainY4, trainY5);
    trainy = cat(1, trainy1, trainy2, trainy3, trainy4, trainy5);
    if (flag==2)
        validX = trainX(:, end-4999: end);%3072*5000
        validY = trainY(:, end-4999: end);
        validy = trainy(end-4999:end,:);
        trainX = trainX(:,1: end-5000);%3072%45000
        trainY = trainY(:,1: end-5000);
        trainy = trainy(1:end-5000,:);
    else
        validX = trainX(:, end-999: end);%3072*5000
        validY = trainY(:, end-999: end);
        validy = trainy(end-999:end,:);
        trainX = trainX(:,1: end-1000);%3072%45000
        trainY = trainY(:,1: end-1000);
        trainy = trainy(1:end-1000,:);
    end   
end
% initial W and b
[d, ~] = size(trainX);
K = 10;
[W, b] = Initialize(K,d);
end

%% Coarse-to-fine random search to set lambda
function all_para = SearchLambda(l_min, l_max, n_batch, n_c, trainX, trainY, trainy, validX, validY, validy, testX, testy, W, b)
%N = size(trainX, 2);
%n_s = 2 * floor(N / n_batch);
n_s = 900;
nn = 15;
lambda_list = zeros(1,nn);
valid_acc_list = zeros(1,nn);
for i=1:nn
    % generate lambda sample 
    l = l_min + (l_max - l_min)*rand(1, 1);
    lambda = 10^l
    lambda_list(i) = lambda;
    valid_acc_list(i) = MiniBatchGD(trainX, trainY, trainy, validX, validY, validy, testX, testy, n_batch, n_s, n_c, W, b, lambda, 0);
end
all_para = {lambda_list, valid_acc_list, n_batch, n_s, 2};
figure
plot(lambda_list, valid_acc_list, 'ro')
xlabel('lambda')
ylabel('best validation accuracy')
end
