# Deep-learning-foundation
A collection of assignments from two foundamental deep learning courses at KTH, namely Artificial Neural Networks and Deep Architectures (DD2437) and Deep Learning in Data Science (DD2424)，where I studied the classic&foundamental deep learning algorithms (MLP, CNN, RNN, AE, Hopfield Network, competitive learning and SOM) and implemented them from scratch in MATLAB / Python with Numpy library. Note that some codes need to be further cleaned.
## Content of each folder
>MLP_one_layer

Implement, train, and test a one layer network with multiple outputs to classify images from the CIFAR-10 dataset. 

Training the network using mini-batch gradient descent applied to a cost function that computes the cross-entropy loss of the classifier applied to the labelled training data and an L2 regularization term on the weight matrix. Exploring tricks/avenues that help bumping up performance such as finding good hyperparemeters, learning rate decay, Xavier initialization, ensemble, etc.

>MLP_two_layer

Implement, train, and test a two layer network with multiple outputs to classify images from the CIFAR-10 dataset. 

Train the network with mini-batch gradient descent +  cross-entropy loss function + L2 regularization term on the weight matrix.
Hyperparameter search (Coarse-to-fine random search), cyclical learning rate, data augementation, dropout and so on are tried to improve the performance.

>CNN

Implement and train a ConvNet to predict the language of a surname from its spelling. 

>RNN

Implement and train an RNN to synthesize English text character.
Train a vanilla RNN with outputs, using the text from the book The Goblet of Fire by J.K. Rowling, with AdaGrad as the SGD optimizor.

>ANN_lab1

Learning and generalisation in feed-forward networks - from perceptron learning to backprop

>ANN_lab2

Radial basis functions, competitive learning and self-organisation

>ANN_lab3

Hopfield networks

>ANN_lab4

Deep neural network architectures with autoencoders

## Requirements


## References
[DD2424 course materials](https://www.kth.se/student/kurser/kurs/DD2424)

[DD2437 course materials](https://www.kth.se/student/kurser/kurs/DD2437)

[Cyclical Learning Rates](https://arxiv.org/abs/1506.01186)
