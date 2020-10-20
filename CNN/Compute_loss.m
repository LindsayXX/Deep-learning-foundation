function loss = Compute_Loss(X_batch, Ys_batch, ConvNet)
% forward pass 
[~, ~, P_batch] = ForwardPass(X_batch, ConvNet);
% loss
[~, n] = size(Ys_batch);
loss = 0;
for i=1:n
    loss = loss - log(Ys_batch(:,i).' * P_batch(:,i));
end
loss = loss/n;
end

function [X_batch_1, X_batch_2, P_batch] = ForwardPass(X_batch, ConvNet)
% forward pass 
%n_len1 = n_len - k1 + 1;
n_len = 19;
n_len1 = 15;
MFs{1} = MakeMFMatrix(ConvNet.F{1}, n_len);
MFs{2} = MakeMFMatrix(ConvNet.F{2}, n_len1);
[~, N] = size(X_batch);
for i=1:N
X_batch_1(:,i) = max(MFs{1} * X_batch(:, i), 0);
X_batch_2(:,i) = max(MFs{2} * X_batch_1(:, i), 0);
S_batch = ConvNet.W * X_batch_2(:, i);
P_batch(:, i) = softmax(S_batch);
end
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