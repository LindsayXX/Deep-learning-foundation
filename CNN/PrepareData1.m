load('assignment3_names.mat')
% all_names: 1*20050 cell
% ys: 20050*1 double
%% get a vector containing the unique characters
C = unique(cell2mat(all_names));
d = numel(C);%28 dim of one-hot vector, existed alphabet
n_len = max(strlength(all_names));%19 maximum length of name
CC = unique(ys);
k = numel(CC);%18 classes

%char_to_ind = containers.Map('KeyType','char','ValueType','int32');
Key = num2cell(C);
Val = int32(1:d);
char_to_ind = containers.Map(Key, Val);
y = bsxfun(@eq, ys(:), 1:max(ys)).';%k*N
[~, N] = size(y);

%% encode your input name
% d * n_len input matrix X
% has at most 1 non-zeros element per column
% N: #names in the datasets
% d: dim of one-hot vector (Background 1)
%X = cell(N, 1);%(d * n_len, N);
Xx = zeros(d*n_len, N);
X = zeros(d, n_len, N);
for i=1:N
    name = all_names{i};
    name = name(:);
    x = zeros(1, n_len);
    for j=1:length(name)
        x(1, j) = char_to_ind(name(j));   
    end
    onehot = bsxfun(@eq, x(:), 1:d).';%28*19
    %inds = (i-1)*d+1 : i*d;
    Xx(:, i) = onehot(:);
    X(:, :, i) = onehot;
end


%% split into train and valid set
valid = fopen('Validation_Inds.txt', 'r');
V = fscanf(valid, '%u');%252*1
validX = X(:,:,V);%252*1cell, 28*19
validY = y(:,V);%19*252
validYs = ys(V);
validXx = Xx(:, V);

T = 1:N;
T(V) = 0;
T = nonzeros(T.');
trainX = X(:,:,T);%19798*1cell, 28*19
trainXx = Xx(:, T);
trainY = y(:,T);%19 * 19798
trainYs = ys(T);

save('names_data.mat', 'validXx','trainXx','trainY', 'trainYs', 'validYs', 'validY');
%save('names.mat', 'trainY', 'validY');
save('c2i.mat','char_to_ind')

Class{1} = [1 1986];
Class{2} = [1987 2240];
Class{3} = [2241 2745];
Class{4} = [2746 3028];
Class{5} = [3029 6682];
Class{6} = [6683 6945];
Class{7} = [6946 7655];
Class{8} = [7656 7844];
Class{9} = [7845 8062];
Class{10} = [8063 8757];
Class{11} = [8758 9734];
Class{12} = [9735 9814];
Class{13} = [9815 9939];
Class{14} = [9940 9999];
Class{15} = [10000 19369];
Class{16} = [19370 19455];
Class{17} = [19456 19739];
Class{18} = [19740 19798];

country{1} = 'Arabic';
country{2} = 'Chinese';
country{3} = 'Czech';
country{4} = 'Dutch';
country{5} = 'English';
country{6} = 'French';
country{7} = 'German';
country{8} = 'Greek';
country{9} = 'Irish';
country{10} = 'Italian';
country{11} = 'Japanese';
country{12} = 'Korean';
country{13} = 'Polish';
country{14} = 'Portuguese';
country{15} = 'Russian';
country{16} = 'Scottish';
country{17} = 'Spanish';
country{18} = 'Vietnamese';

save('classind.mat', 'Class','country')