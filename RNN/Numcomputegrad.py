import numpy as np
import copy

def ComputeGradNumSlow(X, Y, f, RNN, m, h):
    #n = np.size(RNN[f])
    n1 = RNN[f].shape[0]
    n2 = RNN[f].shape[1]
    grad = np.zeros(RNN[f].shape)
    hprev = np.zeros((m, 1))
    RNN_try = copy.deepcopy(RNN)
    for i in range(n1):
        for j in range(n2):
            RNN_try[f][i, j] = RNN[f][i, j] - h
            l1 = ComputeLoss(X, Y, hprev, RNN_try)
            RNN_try[f][i, j] = RNN[f][i, j] + h
            l2 = ComputeLoss(X, Y, hprev, RNN_try)
            grad[i, j] = (l2 - l1) / (2*h)

    return grad


def ComputeLoss(X, Y, h0, RNN):
    cost = 0
    # forward
    tao = X.shape[1]
    h_pre = h0
    P = np.zeros(X.shape)
    for t in range(tao):
        a = np.dot(RNN['W'], h_pre) + np.dot(RNN['U'], X[:, t].reshape(-1, 1)) + RNN['b']
        h_pre = np.tanh(a)
        o = np.dot(RNN['V'], h_pre) + RNN['c']
        P[:, t] = (np.exp(o) / np.sum(np.exp(o))).reshape(-1,)

    for j in range(Y.shape[1]):
        cost -= np.log(np.dot(Y[:, j], P[:, j]))

    return cost

#if __name__ == '__main__':
