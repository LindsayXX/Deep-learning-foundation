validation_loss = 0;
load('valid_loss.mat','valid_loss');
validation_loss = valid_loss;
load('model2.mat', 'valid_loss');
validation_loss = [validation_loss valid_loss];
load('model3.mat','valid_loss');
validation_loss = [validation_loss valid_loss];
ind = 1:10:size(validation_loss,2);
validation_loss = validation_loss(ind);
n_update = 500;
x = linspace(0, size(validation_loss,2)-1,size(validation_loss,2)) * n_update;
%[valid_acc, M] = ConfusionMatrix(X_val, Ys_val, ConvNet.W, MFs);
figure
subplot(1,2,1)
plot(x, validation_loss);
ylabel('loss')
xlabel('steps')
title('Validation loss')
% subplot(1,2,2)
% imagesc(M)
% title(['Acc=',num2str(valid_acc)])
