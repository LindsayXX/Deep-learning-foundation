%% check
%load('DebugInfo.mat')

load('names_data.mat', 'validX');%trainX: 28*19 * 19798 , trainY 18 * 19798
load('names.mat', 'validY');%validX: 28*19 * 252, validY 18*252
% x_input: 28*nlen * 1 vector
% X_input: 28*nlen matrix
% F: 28*5*4 3D array corresponding to 4 convolutional filters
% vecF: 28*5*4 * 1 vector 
% S: 4*nlen-5+1 matrix
% vecS: 4*(nlen-4) * 1 vector

n = 2;
n_len = 19;
X_input = validX(:,:,1);
%X_input = reshape(X_input, d, )
x_input = reshape(X_input, [28*19, 1]);


%nlen = size(X_input, 2);

F = F(:,:,1:2);
[d, k, nf] = size(F);

%[d, k, nf] = size(ConvNet.F{1});
MF = MakeMFMatrix(F, n_len);%nlen
%difff = VF(:) - F(:);
MX = MakeMXMatrix(x_input, d, k, nf);
%s1 = MX * ConvNet.F{1}(:);
s1 = MX * (F(:));
s2 = MF * x_input;
diff1 = s1 - s2;%0
vecs = [s2.' s1.'].';
diff = vecS - vecs;%0

load('names_data.mat', 'trainX', 'validX');%trainX: 28*19 * 19798 , trainY 18 * 19798
load('names.mat', 'trainY', 'validY');%validX: 28*19 * 252, validY 18*252


%% M_{F, nlen}
% nlen: length of row in X
% F: dd(nlen) * k
% MF size: (nlen-k+1)*nf X nlen*dd
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

%% M_{X, k, nf}
% k: width of filter
% nf: number of filter
% X: d*nlen
% x_input: vectorized
% X_input has size d x nlen
% can get nlen by dividing the length of x_input by d
%[d, k, nf] = size(ConvNet.F{1})
function MX = MakeMXMatrix(x_input, d, k, nf)
nlen = size(x_input,1) / d;
X = reshape(x_input, [d, nlen]);
MX = zeros((nlen-k+1)*nf, k*nf*d);
for i=1:(nlen-k+1)
    x = X(:, i:i+k-1);%d*k
    vx = x(:).';%d*k*1
    %vx = x_input((i-1)*d+1:(i-1)*d+d*k);
    vx = reshape(vx, 1, []);
    %MX((i-1) + j, (j-1)*d*k+1:j*d*k) = vx;
    for j=1:nf
        MX((i-1)*nf + j, (j-1)*d*k+1:j*d*k) = vx;
    end
end
MX = sparse(MX);
end

