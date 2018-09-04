# author: viaeou
import random

import numpy as np
import sys
sys.path.append('../')
import utils.lr_utils as lr_utils


def select_j_rand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


def kernel_trans(X, A, kTup):
    m, n = np.shape(X)
    K = np.zeros((m, 1))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            delta_row = X[j, :] - A
            K[j] = delta_row * delta_row.T
        K = np.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('The kernel is not recognized')

    return K


def normalize_data(train_set_x, train_set_y, test_set_x, test_set_y):
    train_x = train_set_x.reshape(train_set_x.shape[0], -1) / 255.
    test_x = test_set_x.reshape(test_set_x.shape[0], -1) / 255.
    train_y = train_set_y.transpose()
    test_y = test_set_y.transpose()
    return train_x, train_y, test_x, test_y


class OptStruct:
    def __init__(self, X, y, C, toler, kind_tuple):
        """

        :param X:
        :param y:
        :param C:
        :param toler:
        :param kind_tuple:
        """
        self.X = X
        self.y = y
        self.C = C
        self.toler = toler
        self.m = np.shape(X)[0]
        self.alphas = np.zeros((self.m, 1))
        self.b = 0
        self.e_cache = np.zeros((self.m, 2))
        self.K = np.zeros((self.m, self.m))
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.X, self.X[i, :], kind_tuple)


def cal_ek(opt_struc, i):
    g_x_k = np.multiply(opt_struc.alphas, opt_struc.y).T*opt_struc.K[:, i] + opt_struc.b
    ek = g_x_k - opt_struc.y
    return ek


def select_j(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]  #返回矩阵中的非零位置的行数
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = cal_ek(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE): #返回步长最大的aj
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = select_j_rand(i, oS.m)
        Ej = cal_ek(oS, j)
    return j, Ej


def inner_l(i, opt_struc):
    Ei = cal_ek(opt_struc, i)
    if ((opt_struc.y[i] * Ei <  - opt_struc.toler) and (opt_struc.alphas[i] < opt_struc.C))\
        or ((opt_struc.y[i] > Ei) and (opt_struc.alphas[i] > 0)):
        j, Ej = select_j(i, opt_struc, Ei)


def smo(X, y, C, toler, max_iters, kind_tuple=('lin', 0)):
    opt_struc = OptStruct(X, y, C, toler, kind_tuple)
    iter = 0
    entire_set = True
    alpha_pair_changed = 0
    if iter < max_iters and (alpha_pair_changed > 0 or entire_set):
        alpha_pair_changed = 0
        if entire_set:
            for i in range(opt_struc.m):
                alpha_pair_changed += inner_l(i, opt_struc)
    return 0


def svm_with_rbf(train_x, train_y, test_x, test_y):
    b, alphas = smo(train_x, train_y, 200, 0.0001, 10000, ('rbf', 1.3))
    sv_index = np.nonzero(alphas)[0]
    sv_features = train_x[sv_index]
    sv_labels = train_y[sv_index]
    print('there are %d support vectors'%(np.shape(sv_features)[0]))
    m, n = np.shape(train_x)
    err_cnt = 0
    for i in range(m):
        kernel_eval = kernel_trans(sv_features, train_x[i, :], ('rbf', 1.3))
        predict = kernel_eval.T * np.multiply(sv_labels, alphas[sv_index]) + b
        if np.sign(predict) != np.sign(train_y[i]):
            err_cnt += 1
    print('the error rate is: %f'%(np.float(err_cnt/m)))


def main():
    train_set_x, train_set_y, test_set_x, test_set_y, classes = lr_utils.load_dataset()
    train_x, train_y, test_x, test_y = normalize_data(train_set_x, train_set_y, test_set_x, test_set_y)
    # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    svm_with_rbf(train_x, train_y, test_x, test_y)


if __name__ == '__main__':
    main()