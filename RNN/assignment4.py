import numpy as np
import matplotlib.pyplot as plt
import tqdm
#utf-8


np.random.seed(10)

"""
1. Preparing Data
"""
def readData(file):
    f = open(file, encoding='utf-8')
    book_data = f.read()#string
    chars = list(set(book_data))#80
    f.close()
    K = len(chars)
    inds = list(range(0, K))
    char2ind = dict(zip(chars, inds))
    ind2char = dict(zip(inds, chars))

    return book_data, K, char2ind, ind2char


def computeDiff(ga, gn, eps=1e-5):
    error = (np.abs(ga - gn) / np.maximum(eps, np.abs(ga) + np.abs(gn)))

    return error


class RNN():
    def __init__(self, K, char2ind, ind2char, m=100, eta=0.1, seq_length=25, sig=0.01):
        self.m = m
        self.eta = eta
        self.seq_len = seq_length
        self.K = K
        self.U = np.random.randn(m, K) * sig
        self.W = np.random.randn(m, m) * sig
        self.V = np.random.randn(K, m) * sig
        self.b = np.zeros((m, 1))#(m)
        self.c = np.zeros((K, 1))#(K)
        self.char2ind = char2ind
        self.ind2char = ind2char
        self.hprev = np.zeros(self.m)
        self.m_U = np.zeros(self.U.shape)
        self.m_V = np.zeros(self.V.shape)
        self.m_W = np.zeros(self.W.shape)
        self.m_b = np.zeros(self.b.shape)
        self.m_c = np.zeros(self.c.shape)


    def onehot(self, x):
        # convert character to one-hot vector
        n = len(x)
        X = np.zeros((K, n))
        if isinstance(x[0], int):
            for i in range(n):
                ind = x[i]
                X[ind, i] = 1
        else:
            for i in range(n):
                ind = self.char2ind[x[i]]
                X[ind, i] = 1

        return X


    def decode(self, Y):
        # Y: Kxn matrix
        n = Y.shape[1]
        output = []
        for i in range(n):
            ind = np.argmax(Y[:, i])
            output.append(self.ind2char[ind])

        return ''.join(output)


    def forward(self, X):
        tao = X.shape[1]
        P = np.zeros((self.K, tao))
        for t in range(tao):
            self.a[:, t] = (np.dot(self.W, self.h[:, t].reshape(-1, 1)) + np.dot(self.U, X[:, t].reshape(-1, 1)) + self.b).reshape(-1,)
            self.h[:, t+1] = np.tanh(self.a[:, t])
            o = np.dot(self.V, self.h[:, t+1].reshape(-1, 1)) + self.c
            P[:, t] = (np.exp(o) / np.sum(np.exp(o))).reshape(-1,)

        return P


    def loss(self, X, Y):
        cost = 0
        P = self.forward(X)
        for t in range(X.shape[1]):
            cost -= np.log(np.dot(Y[:, t], P[:, t]))

        return cost


    """
    2. Back-propagation
    """
    def backward(self, X, Y):
        grad = {}
        P = self.forward(X)
        tao = X.shape[1]

        # Gradient of V
        grad_o = - (Y - P)#(K, seq_length)
        grad_V = np.zeros(self.V.shape)
        for t in range(tao):
            grad_V += np.dot(grad_o[:, t].reshape(-1, 1), self.h[:, t+1].reshape(1, -1))
        grad['V'] = grad_V
        #self.grad_V = grad_V
        # Gradient of c (K*1)
        grad['c'] = np.sum(grad_o, axis=1).reshape(-1, 1)
        #self.grad_c = grad['c']

        # Gradient of W (m*m)
        grad_h = np.zeros((self.m, tao))
        grad_a = np.zeros((self.m, tao))
        grad_h[:, tao-1] = np.dot(grad_o[:, tao - 1], self.V)
        grad_a[:, tao-1] = np.dot(grad_h[:, tao-1], np.diag(1 - np.tanh(self.a[:, tao-1]) ** 2))
        for i in reversed(range(0, tao-1)):
            grad_h[:, i] = np.dot(grad_o[:, i], self.V) + np.dot(grad_a[:, i+1], self.W)
            grad_a[:, i] = np.dot(grad_h[:, i], np.diag(1 - np.tanh(self.a[:, i]) ** 2))

        grad_W = np.zeros(self.W.shape)
        for i in range(tao):
            grad_W += np.dot(grad_a[:, i].reshape(-1, 1), self.h[:, i].reshape(1, -1))
        grad['W'] = grad_W
        #self.grad_W = grad_W
        # Gradient of b(m*1)
        grad['b'] = np.sum(grad_a, axis=1).reshape(-1, 1)
        #self.grad_b = grad['b']

        # Gradient of U(m*K)
        grad_U = np.zeros(self.U.shape)
        for j in range(tao):
            grad_U += np.dot(grad_a[:, j].reshape(-1, 1), X[:, j].reshape(1, -1))
        grad['U'] = grad_U
        #self.grad_U = grad_U

        return self.clipgrad(grad)


    def clipgrad(self, grads):
        keys = grads.keys()
        new_grads = {}
        for key in keys:
            new_grad = np.zeros(grads[key].shape)
            new_grad = np.maximum(np.minimum(grads[key], 5), -5)
            new_grads[key] = new_grad

        return new_grads


    """
    3. AdaGrad
    """
    def gradupdate(self, grads, eps=1e-7):
        self.m_W += grads['W'] ** 2
        self.W -= self.eta / (np.sqrt(self.m_W + eps)) * grads['W']
        self.m_V += grads['V'] ** 2
        self.V -= self.eta / (np.sqrt(self.m_V + eps)) * grads['V']
        self.m_U += grads['U'] ** 2
        self.U -= self.eta / (np.sqrt(self.m_U + eps)) * grads['U']
        self.m_b += grads['b'] ** 2
        self.b -= self.eta / (np.sqrt(self.m_b + eps)) * grads['b']
        self.m_c += grads['c'] ** 2
        self.c -= self.eta / (np.sqrt(self.m_c + eps)) * grads['c']

    def train(self, book_data, iter, e=0, update=100, syn=500):
        smooth_losses = []
        smooth_loss = 0
        epoch = 0
        for i in range(iter):
            X = self.onehot(book_data[e: e + self.seq_len])
            Y = self.onehot(book_data[e+1: e + self.seq_len+1])
            self.h = np.zeros((self.m, self.seq_len + 1))
            self.h[:, 0] = self.hprev
            self.a = np.zeros((self.m, self.seq_len))

            grads = self.backward(X, Y)
            self.gradupdate(grads)# AdaGrad algorithm
            loss = self.loss(X, Y)
            if i==0:
                smooth_loss = loss
            else:
                smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                smooth_losses.append(smooth_loss)
            if i % update == 0:
                print(f'iter = {i}, smooth_loss = {smooth_loss}')
            if i % syn == 0:
                self.synthesize(X[:, 0])
            e += self.seq_len
            if (e > len(book_data) - self.seq_len):
                print(f'one epoch finished in {i} steps')
                e = 0
                self.hprev = np.zeros(self.m)
                epoch += 1
                if epoch >= 2:
                    plt.figure()
                    plt.plot(smooth_losses)
                    plt.xlabel('steps')
                    plt.ylabel('smooth loss')
                    plt.show()
            else:
                e += self.seq_len
                self.hprev = self.h[:, -1]
        print(f'iter = {iter}, smooth_loss = {smooth_loss}')
        #X = self.onehot(book_data[e - self.seq_len: e])
        self.synthesize(X[:, 0], n=1000)


    """
    4. Synthesizing text
    """
    def synthesize(self, x0, n=200):
        Y = np.zeros((self.K, n + 1))
        Y[:, 0] = x0
        xnext = x0.reshape(-1, 1)
        hprev = self.hprev.reshape(-1, 1)
        for t in range(n):
            #P = np.zeros((self.K, 1))
            a = (np.dot(self.W, hprev) + np.dot(self.U, xnext) + self.b).reshape(-1, 1)
            hprev = np.tanh(a)
            o = np.dot(self.V, hprev) + self.c
            P = (np.exp(o) / np.sum(np.exp(o))).reshape(-1,)
            '''
            cp = np.cumsum(P)
            aa = np.random.rand(1)
            ixs = np.where((cp - aa) > 0)[0]
            ii = int(ixs[0])
            '''
            ii = np.random.choice(a=list(range(self.K)), size=1, p=P)[0]
            xnext = self.onehot([int(ii)])
            Y[:, t + 1] = xnext.reshape(-1, )
        output = self.decode(Y)
        print(output)
        #return output

    def gradcheck(self, input, output, hh):
        X = self.onehot(input)
        Y = self.onehot(output)
        self.h = np.zeros((self.m, len(input) + 1))
        self.h[:, 0] = self.h0.reshape(-1, )
        self.a = np.zeros((self.m, len(input)))

        Grad = self.backward(X, Y)

        GradNum = {}

        RNNG = {"W": self.W, "U": self.U, "V": self.V, "b": self.b, "c": self.c}
        keys = RNNG.keys()
        for key in keys:
            print('Computing numerical gradient for ', key)
            GradNum[key] = self.ComputeGradNumSlow(X, Y, key, hh)

        diff = {}
        diff['W'] = np.max(computeDiff(Grad['W'], GradNum['W']))
        diff['U'] = np.max(computeDiff(Grad['U'], GradNum['U']))
        diff['V'] = np.max(computeDiff(Grad['V'], GradNum['V']))
        diff['b'] = np.max(computeDiff(Grad['b'], GradNum['b']))
        diff['c'] = np.max(computeDiff(Grad['c'], GradNum['c']))

        return diff

    def ComputeGradNumSlow(self, X, Y, f, h):
        para = {"W": self.W, "U": self.U, "V": self.V, "b": self.b, "c": self.c}
        # n = np.size(RNN[f])
        n1 = para[f].shape[0]
        n2 = para[f].shape[1]
        grad = np.zeros(para[f].shape)
        for i in range(n1):
            for j in range(n2):
                para[f][i, j] -= h
                l1 = self.loss(X, Y)
                para[f][i, j] += 2 * h
                l2 = self.loss(X, Y)
                grad[i, j] = (l2 - l1) / (2 * h)
                para[f][i, j] -= h

        return grad





if __name__ == '__main__':
    book_data, K, char2ind, ind2char = readData('goblet_book.txt')
    '''
    net = RNN(K, char2ind, ind2char)
    h0 = np.zeros((100, 1))
    x0 = np.zeros(K)
    x0[20] = 1
    n = 100
    random_stuff = net.synthesize(x0, n)
    print(random_stuff)
    '''
    # gradient check
    '''
    seq_length = 25
    X_chars = book_data[0:seq_length]
    Y_chars = book_data[1:seq_length+1]
    net = RNN(K, char2ind, ind2char, m=5)
    diff = net.gradcheck(X_chars, Y_chars, 1e-4)
    print(diff)
    '''
    net = RNN(K, char2ind, ind2char)
    #net.train(book_data, update=500, iter=100000, syn=10000)
    #net.train(book_data, update=200, iter=10000, syn=1000)
    net.train(book_data, update=5000, iter=400000, syn=50000)






