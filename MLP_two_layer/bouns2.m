addpath DirName/Datasets/cifar-10-batches-mat/

%% Exercise 5.1
rng(400)

%% (b) explore whether having more hidden nodes improves the final classification rate
%[trainX, trainY, trainy, validX, validY, validy, testX, testy] = GetData(2);
% para_nn_1 = SearchNN(100, 3, trainX, trainY, trainy, validX, validY, validy, testX, testy, 9.8e-4);
% save('para_nn_1')
%% (c) ensemble of classifiers
%initial W and b
% [d, ~] = size(trainX);
% K = 10;
% [W, b] = Initialize(K,d,50);
% [valid_acc_list, acc_list] = MiniBatchGD(trainX, trainY, trainy, validX, validY, validy, testX, testy, 100, 900, 6, W, b, 9.8e-4, 1, 0);
% [d, ~] = size(trainX);
% K = 10;
% [W, b] = Initialize(K,d,1000);
% [valid_acc_list, acc_list] = MiniBatchGD_D(trainX, trainY, trainy, validX, validY, validy, testX, testy, 100, 900, 6, W, b, 9.8e-4, 1, 0.9, 0);

%% (d) dropout
% %initial W and b
% [d, ~] = size(trainX);
% K = 10;
% [W, b] = Initialize(K,d,1000);
% acc_list = zeros(1,5);
% valid_acc = zeros(1,5);
% for i=1:5
%     p = (10-i)/10
%     [valid_acc(i+1), acc_list(i+1)] = MiniBatchGD_D(trainX, trainY, trainy, validX, validY, validy, testX, testy, 100, 900, 3, W, b, 9.8e-4, 0, p, 0);
% end
% figure
% xx = [0, 0.1, 0.1, 0.3, 0.4, 0.5];
% plot(xx, acc_list, 'ro')
% ylabel('Accuracy')
% xlabel('dropout probability')

%% best test score
%[trainX, trainY, trainy, validX, validY, validy, testX, testy] = GetData(3);
%not ensemble 1000 nodes with 0.9 dropout
%initial W and b
% [d, ~] = size(trainX);
% K = 10;
% [W, b] = Initialize(K,d,1000);
%[valid_acc, acc] = MiniBatchGD_D(trainX, trainY, trainy, validX, validY, validy, testX, testy, 100, 980, 6, W, b, 9.8e-4, 0, 0.9, 1, 0);


%% Exercise 5.2 find good values for eta_min and eta_max 
%LRtest(0, 0.2, 980, 100, trainX, trainY, trainy, validX, validY, validy, testX, testy, W, b, 9.8e-4);
% compare with the best model
[valid_acc, acc] = MiniBatchGD_D(trainX, trainY, trainy, validX, validY, validy, testX, testy, 100, 980, 6, W, b, 9.8e-4, 0, 0.9, 1, 1);

%% dropout
function [best_score,acc] = MiniBatchGD_D(X, Y, y, X_val, Y_val, y_val, X_test, y_test, n_batch, n_s, n_c, W, b, lambda, ensemble_flag, dropout, show_flag, eta_flag)
n_step = n_s * 2 * n_c;
if eta_flag==1
    eta = CyclicalEta(0.01, 0.05, n_s, n_c);
else
    eta = CyclicalEta(1e-5, 1e-1, n_s, n_c);
end
count = 9;
N=size(X,2);
n_epochs = ceil(n_step/(N/n_batch));
wait = floor(n_s * 2 / count);
len = count * n_c + 1; 
Wstar = W;
bstar = b;
n_test = size(X_test, 2);
p_ens = zeros(n_c, 10, n_test);%K*n
c = 0;
m = 1;
if show_flag==1
    train_loss = zeros(1, len);
    train_cost = zeros(1, len);
    % performance before training
    [train_loss(m),~] = ComputeCost(X, Y, Wstar, bstar, 0);
    [train_cost(m),~] = ComputeCost(X, Y, Wstar, bstar, lambda);
end
train_acc = zeros(1, len);
[train_acc(m), ~] = ComputeAccuracy(X, y, Wstar, bstar);
valid_acc = zeros(1, len);
[valid_acc(m), ~] = ComputeAccuracy(X_val, y_val, Wstar, bstar);
for i=1:n_epochs
    %generate the set of mini-batches
    for j=1:N/n_batch
        k = (i-1)*N/n_batch + j;%step
        if (k == (n_step+1))
            %[P, ~] = ForwardPass(X_test, Wstar, bstar);
            %p_list{n_c} = P;
            break
        else            
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            inds = j_start:j_end; 
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);
                       
            [P, h] = ForwardPass(Xbatch, Wstar, bstar, dropout);
            [ngrad_W, ngrad_b] = ComputeGradients(Xbatch, Ybatch, P, h, Wstar, lambda);
            Wstar{1} = Wstar{1} - eta(k) * ngrad_W{1};
            Wstar{2} = Wstar{2} - eta(k) * ngrad_W{2};
            bstar{1} = bstar{1} - eta(k) * ngrad_b{1};
            bstar{2} = bstar{2} - eta(k) * ngrad_b{2};
            
            if (mod(k, n_s * 2) == 0 && (ensemble_flag==1))
                c = c + 1
                acc = ComputeAccuracy(X_test, y_test, Wstar, bstar)
                [P, ~] = ForwardPass(X_test, Wstar, bstar, 0);
                p_ens(c, :, :) = P;
                PP = zeros(10, n_test);
                PP(:,:) = mean(p_ens);
                [~, KK] = max(PP);%K:1*n
                S = KK.'- double(y_test);
                acc_ens = nnz(~S)/n_test
            elseif(mod(k, wait) == 0 && show_flag==1)
                m = m + 1;
                [train_loss(m),~] = ComputeCost(X, Y, Wstar, bstar, 0);
                [train_cost(m),~] = ComputeCost(X, Y, Wstar, bstar, lambda);
                %[train_acc(m), ~] = ComputeAccuracy(X, y, Wstar, bstar);
            elseif(mod(k, wait) == 0)
                m = m + 1;
                %[valid_acc(m), ~] = ComputeAccuracy(X_val, y_val, Wstar, bstar);
            end
        end
    end
end
acc = ComputeAccuracy(X_test, y_test, Wstar, bstar)
% if ensemble_flag == 1
%     P = zeros(10, n_test);
%     P(:,:) = mean(p_ens);
%     [~, K] = max(P);%K:1*n
%     S = K.'- double(y_test);
%     acc_ens = nnz(~S)/n_test
% end
if show_flag == 1
    xx = linspace(0, n_step, len);
    figure
    subplot(1,2,1)
    plot(xx, train_cost)
    ylabel('training cost');
    xlabel('steps')
    subplot(1,2,2)
    plot(xx, train_loss)
    ylabel('training loss')
    xlabel('steps')
    %title(['Test Accuracy = ', num2str(acc)]);
end
best_score = max(valid_acc);
end


function LRtest(min_lr, max_lr, stepsize, n_batch, trainX, trainY, trainy, validX, validY, validy, testX, testy, W, b, lambda)
[valid_acc, test_acc] = MiniBatchGD(trainX, trainY, trainy, validX, validY, validy, testX, testy, n_batch, stepsize, 1, W, b, lambda, 0, 0, 1);
end


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

function [W, b] = Initialize(K, d, m)
%m = 50; % nr of nodes in the hidden layer
W1 = randn(m, d)/sqrt(d);%m*d
b1 = zeros(m, 1);
W2 = randn(K, m)/sqrt(m);%K*m
b2 = zeros(K, 1);
W = {W1, W2};
b = {b1, b2};% 1*2 cell array
end

%% compute the gradients for the network parameters
% forward pass
function [P, h] = ForwardPass(X, W, b, drop)
[~, n_b] = size(X);
m = size(W{1},1);
s1 = W{1} * X + b{1} * ones(1,n_b);%m*n
h = max(zeros(m, n_b), s1);%m*n
if drop ~= 0
    h = h .* binornd(1, drop, size(h)) / drop;
end
s = W{2} * h + b{2} * ones(1,n_b);%
P = softmax(s);%K*n(N)
end

% cost function
function [J, loss] = ComputeCost(X, Y, W, b, lambda)
[P, ~] = ForwardPass(X, W, b, 0);%K*n 
[~,n] = size(Y);
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

%% initialize cyclical learning rates
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

%% compute accuracy
function [acc, P] = ComputeAccuracy(X, y, W, b)
[~,n] = size(X);
[P,~] = ForwardPass(X, W, b, 0);%K*n
[~, K] = max(P);%K:1*n
S = K.'- double(y);
acc = nnz(~S)/n;
end


%% Train your network for real
function [best_score,acc] = MiniBatchGD(X, Y, y, X_val, Y_val, y_val, X_test, y_test, n_batch, n_s, n_c, W, b, lambda, show_flag, ensemble_flag, eta_flag)
if eta_flag == 1
    eta = linspace(0, 0.2, n_s);
    count = 80;
    n_step = n_s;
    len = count/2 + 1;
else
    %eta = CyclicalEta(1e-5, 1e-1, n_s, n_c);
    eta = CyclicalEta(0.01, 0.05, n_s, n_c);
    n_step = n_s * 2 * n_c;
    count = 9;
    len = count * n_c + 1; %28
end
N=size(X,2);
n_epochs = ceil(n_step/(N/n_batch));
wait = floor(n_s * 2 / count);
Wstar = W;
bstar = b;
n_test = size(X_test, 2);
p_ens = zeros(n_c, 10, n_test);%K*n
c = 0;
m = 1;
if show_flag==1
    train_loss = zeros(1, len);
    %valid_loss = zeros(1, len);
    train_cost = zeros(1, len);
    %valid_cost = zeros(1, len);
    % performance before training
    [train_loss(m),~] = ComputeCost(X, Y, Wstar, bstar, 0);
    %[valid_loss(m), ~] = ComputeCost(X_val, Y_val, Wstar, bstar, 0);
    [train_cost(m),~] = ComputeCost(X, Y, Wstar, bstar, lambda);
    %[valid_cost(m), ~] = ComputeCost(X_val, Y_val, Wstar, bstar, lambda);
end
train_acc = zeros(1, len);
[train_acc(m), ~] = ComputeAccuracy(X, y, Wstar, bstar);
valid_acc = zeros(1, len);
[valid_acc(m), ~] = ComputeAccuracy(X_val, y_val, Wstar, bstar);
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

            [P, h] = ForwardPass(Xbatch, Wstar, bstar, 0.9);
            [ngrad_W, ngrad_b] = ComputeGradients(Xbatch, Ybatch, P, h, Wstar, lambda);
            Wstar{1} = Wstar{1} - eta(k) * ngrad_W{1};
            Wstar{2} = Wstar{2} - eta(k) * ngrad_W{2};
            bstar{1} = bstar{1} - eta(k) * ngrad_b{1};
            bstar{2} = bstar{2} - eta(k) * ngrad_b{2};
            
            if (mod(k, n_s * 2) == 0 && (ensemble_flag==1))
                c = c + 1;
                [P, ~] = ForwardPass(X_test, Wstar, bstar, 0);
                p_ens(c, :, :) = P;
            end
            if(mod(k, wait) == 0 && show_flag==1)
                m = m + 1;
                [train_loss(m),~] = ComputeCost(X, Y, Wstar, bstar, 0);
                %[valid_loss(m), ~] = ComputeCost(X_val, Y_val, Wstar, bstar, 0);
                [train_cost(m),~] = ComputeCost(X, Y, Wstar, bstar, lambda);
                %[valid_cost(m), ~] = ComputeCost(X_val, Y_val, Wstar, bstar, lambda);
                [train_acc(m), ~] = ComputeAccuracy(X, y, Wstar, bstar);
                %[valid_acc(m), ~] = ComputeAccuracy(X_val, y_val, Wstar, bstar);
            elseif(mod(k, wait) == 0 && eta_flag==1)
                m = m + 1;
                [train_acc(m), ~] = ComputeAccuracy(X, y, Wstar, bstar);
            elseif(mod(k, wait) == 0)
                m = m + 1;
                [valid_acc(m), ~] = ComputeAccuracy(X_val, y_val, Wstar, bstar);
            end
        end
    end
end
acc = ComputeAccuracy(X_test, y_test, Wstar, bstar)
if ensemble_flag == 1
    P = zeros(10, n_test);
    P(:,:) = mean(p_ens);
    [~, K] = max(P);%K:1*n
    S = K.'- double(y_test);
    acc_ens = nnz(~S)/n_test
end
if eta_flag == 1
    figure
    subplot(2,1,1)
    xx = linspace(0, 0.2, len);
    plot(xx, train_acc)
    xlabel('Learning rate')
    ylabel('Accuracy')
end
if show_flag == 1
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

%% get data and initalize parameters
function [trainX, trainY, trainy, validX, validY, validy, testX, testy] = GetData(flag)
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
        validX = trainX(:, end-999: end);%3072*1000
        validY = trainY(:, end-999: end);
        validy = trainy(end-999:end,:);
        trainX = trainX(:,1: end-1000);%3072%49000
        trainY = trainY(:,1: end-1000);
        trainy = trainy(1:end-1000,:);
    end   
end
end

%% test on increasing number of hidden nodes
function all_para = SearchNN(n_batch, n_c, trainX, trainY, trainy, validX, validY, validy, testX, testy, lambda)
N = size(trainX, 2);
n_s = 2 * floor(N / n_batch);
nr_list = [40 50 100 200 300 500 700 1000];
valid_acc_list = zeros(size(nr_list));
acc_list = zeros(size(nr_list));
for i=1:size(nr_list, 2)
    [d, ~] = size(trainX);
    K = 10;
    [W, b] = Initialize(K, d, nr_list(i));
    %lambda = lambda * (nr_list(i)/50)
    %lambda = lambda * exp((nr_list(i)/50))
    [valid_acc_list(i), acc_list(i)] = MiniBatchGD(trainX, trainY, trainy, validX, validY, validy, testX, testy, n_batch, n_s, n_c, W, b, lambda, 0, 0, 0);
end
all_para = {nr_list, acc_list, valid_acc_list, n_batch, lambda, n_c, n_s};
figure
% plot(nr_list, valid_acc_list, 'ro')
% %xlabel('n\_s')
% ylabel('best validation accuracy')
plot(nr_list, acc_list)
hold on
plot(nr_list, acc_list, 'ro')
hold off
xlabel('number of hidden nodes')
ylabel('test accuracy')
end
