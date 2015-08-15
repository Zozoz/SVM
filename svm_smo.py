#!/usr/bin/env python
# encoding: utf-8


import time
import numpy as np
import matplotlib.pyplot as plt


def GaussianKernel(train_x, sample_x, kernel_option):
    '''
        Gaussian Kernel function.
    '''
    kernel_type = kernel_option[0]
    m = train_x.shape[0]
    K = np.mat(np.zeros((m, 1)))
    if kernel_type == 'linear':
        K = train_x * sample_x.T
    elif kernel_type == 'rbf':
        sigma = kernel_option[1]
        for i in xrange(m):
            minus = train_x[i, :] - sample_x
            K[i] = np.exp(minus * minus.T / (-2.0 * sigma**2))
    else:
        raise NameError('Don not support kernel type! You can use linear or rbf!')
    return K


class SVM(object):

    def __init__(self, data_x, data_y, C, toler, kernel_option):
        self.train_x = data_x
        self.train_y = data_y
        self.C = C
        self.toler = toler
        self.kernel_option = kernel_option
        self.m, self.n = np.shape(self.train_x)
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = GaussianKernel(self.train_x, self.train_x[i, :], kernel_option)
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0

        self.error = np.mat(np.zeros((self.m, 2)))
        self.support_vector = None
        self.support_vector_alphas = None
        self.support_vector_y = None
        self.support_vector_index = None

    # calculate the error for x_k, k is the index
    def get_gx(self, k):
        gx = (self.alphas.T * np.multiply(self.train_y, self.K[:, k]) + self.b)
        #gx = (np.multiply(self.alphas, self.train_y).T * self.K[:, k] + self.b)
        return gx - float(self.train_y[k])

    def update_error(self, k):
        self.error[k] = [1, self.get_gx(k)]

    def select_alpha_j(self, alpha_i, error_i):
        self.error[alpha_i] = [1, error_i]
        candidate_alpha = np.nonzero(self.error[:, 0].A)[0]
        max_step = 0
        alpha_j = 0
        error_j = 0

        if len(candidate_alpha) > 1:
            for alpha_k in candidate_alpha:
                if alpha_k == alpha_i:
                    continue
                error_k = self.get_gx(alpha_k)
                if abs(error_k - error_i) > max_step:
                    max_step = error_k - error_i
                    alpha_j = alpha_k
                    error_j = error_k
        else:
            alpha_j = alpha_i
            while alpha_j == alpha_i:
                alpha_j = int(np.random.uniform(0, self.m))
            error_j = self.get_gx(alpha_j)

        return alpha_j, error_j

    def is_satisfied_kkt(self, y, g, alpha):
        if self.alphas.T * self.train_y != 0:
            return False
        if not (0 <= alpha <= self.C):
            return False
        if y * g >= 1 and alpha == 0:
            return True
        if y * g == 1 and 0 < alpha < self.C:
            return True
        if y * g <= 1 and alpha == self.C:
            return True
        return False

    def inner_loop(self, alpha_i):
        error_i = self.get_gx(alpha_i)

        #if self.train_y[alpha_i] * error_i < self.toler and self.alphas[alpha_i] < self.C or\
        #        self.train_y[alpha_i] * error_i > self.toler and self.alphas[alpha_i] > 0:
        if not self.is_satisfied_kkt(self.train_y[alpha_i], error_i, self.alphas[alpha_i]):
            # step 1: select alpha j
            alpha_j, error_j = self.select_alpha_j(alpha_i, error_i)
            alpha_i_old = self.alphas[alpha_i].copy()
            alpha_j_old = self.alphas[alpha_j].copy()
            # step 2: calculate the boundary L and H for alpha j
            if self.train_y[alpha_i] != self.train_y[alpha_j]:
                L = max(0, alpha_j_old - alpha_i_old)
                H = min(self.C, self.C + alpha_j_old - alpha_i_old)
            else:
                L = max(0, alpha_j_old + alpha_i_old - self.C)
                H = min(self.C, alpha_j_old + alpha_i_old)
            if L == H:
                return 0
            # step 3: calculate eta
            eta = 2.0 * self.K[alpha_i, alpha_j] - self.K[alpha_i, alpha_i] - self.K[alpha_j, alpha_j]
            if eta >= 0:
                return 0
            eta = -eta
            # step 4: update alpha j
            self.alphas[alpha_j] += self.train_y[alpha_j] * (error_i - error_j) / eta
            # step 5: clip alpha j
            if self.alphas[alpha_j] > H:
                self.alphas[alpha_j] = H
            if self.alphas[alpha_j] < L:
                self.alphas[alpha_j] = L
            # step 6: if alpha j don't move, just return
            if abs(alpha_j_old - self.alphas[alpha_j]) < self.toler:
                self.update_error(alpha_j)
                return 0
            # step 7: update alpha i after optimizing alpha j
            self.alphas[alpha_i] += self.train_y[alpha_i] * self.train_y[alpha_j] * (alpha_j_old - self.alphas[alpha_j])
            # step 8: update threshold b
            b1 = self.b - error_i - self.train_y[alpha_i] * self.K[alpha_i, alpha_i]\
                    * (self.alphas[alpha_i] - alpha_i_old) - self.train_y[alpha_j]\
                    * self.K[alpha_j, alpha_i] * (self.alphas[alpha_j] - alpha_j_old)
            b2 = self.b - error_j - self.train_y[alpha_i] * self.K[alpha_i, alpha_j]\
                    * (self.alphas[alpha_i] - alpha_i_old) - self.train_y[alpha_j]\
                    * self.K[alpha_j, alpha_j] * (self.alphas[alpha_j] - alpha_j_old)
            if 0 < self.alphas[alpha_i] < self.C:
                self.b = b1
            elif 0 < self.alphas[alpha_j] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            # step 9: update error
            self.update_error(alpha_i)
            self.update_error(alpha_j)
            return 1
        else:
            return 0


    def train_SVM(self, max_iter):
        start_time = time.time()

        entire_set = True
        iter_count = 0
        alpha_pairs_changed = 0
        while iter_count < max_iter and alpha_pairs_changed > 0 or entire_set:
            alpha_pairs_changed = 0
            if entire_set:
                for i in xrange(self.m):
                    alpha_pairs_changed += self.inner_loop(i)
                print '---iter %d entire set, alpha pairs changed %d.' % (iter_count, alpha_pairs_changed)
                iter_count += 1
            else:
                non_bound_alphas = np.nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                for i in non_bound_alphas:
                    alpha_pairs_changed += self.inner_loop(i)
                print '---iter %d non boundary, alpha pairs changed %d.' % (iter_count, alpha_pairs_changed)
                iter_count += 1

            if entire_set:
                entire_set = False
            elif alpha_pairs_changed == 0:
                entire_set = True

        print 'SVM training had Completed! Took %fs!' % (time.time() - start_time)
        self.support_vector_index = np.nonzero(self.alphas.A > 0)[0]
        self.support_vector = self.train_x[self.support_vector_index]
        self.support_vector_y = self.train_y[self.support_vector_index]
        self.support_vector_alphas = self.alphas[self.support_vector_index]

    def select_i(self):
        error = 0
        index = -1
        for i in xrange(self.m):
            if not self.is_satisfied_kkt(self.train_y[i], self.get_gx(i), self.alphas[i]):
                self.error[i] = [1, self.get_gx(i)]
                print self.error[i]
                if error < abs(self.error[i, 1]):
                    error = abs(self.error[i, 1])
                    index = i
        print 'index: %d' % index
        return index

    def train_SVM_0(self, max_iter):
        start_time = time.time()
        iter_count = 0
        while True:
            cnt = 0
            for i in xrange(self.m):
                if not self.is_satisfied_kkt(self.train_y[i], self.get_gx(i), self.alphas[i]):
                    t = self.inner_loop(i)
                    if t == 1:
                        cnt += 1
            #t = self.inner_loop(self.select_i())
            #if t == 0:
            #    break
            iter_count += 1
            print '---iter %d, changed %d ---' % (iter_count, cnt)
            if cnt == 0:
                break
            if iter_count > max_iter:
                break
        print 'SVM training had Completed! Took %fs!' % (time.time() - start_time)
        self.support_vector_index = np.nonzero(self.alphas.A > 0)[0]
        self.support_vector = self.train_x[self.support_vector_index]
        self.support_vector_y = self.train_y[self.support_vector_index]
        self.support_vector_alphas = self.alphas[self.support_vector_index]

    def test_SVM(self, test_x, test_y):
        test_x = np.mat(test_x)
        test_y = np.mat(test_y)
        m = test_x.shape[0]
        match_count = 0
        for i in xrange(m):
            k = GaussianKernel(self.support_vector, test_x[i, :], self.kernel_option)
            predict = k.T * np.multiply(self.support_vector_y, self.support_vector_alphas) + self.b
            if np.sign(predict) == np.sign(test_y[i]):
                match_count += 1
        return float(match_count) / m

    def show_SVM(self):
        if self.train_x.shape[1] != 2:
            print 'Can not plot. Because the dimension is not 2!'
            return
        for i in xrange(self.m):
            if self.train_y[i] == -1:
                plt.plot(self.train_x[i, 0], self.train_x[i, 1], 'or')
            else:
                plt.plot(self.train_x[i, 0], self.train_x[i, 1], 'ob')
        for i in self.support_vector_index:
            plt.plot(self.train_x[i, 0], self.train_x[i, 1], 'oy')
        w = np.zeros((2, 1))
        for i in self.support_vector_index:
            w += np.multiply(self.alphas[i] * self.train_y[i], self.train_x[i, :].T)
        min_x = min(self.train_x[:, 0])[0, 0]
        max_x = max(self.train_x[:, 0])[0, 0]
        y_min_x = float(-self.b - w[0] * min_x) / w[1]
        y_max_x = float(-self.b - w[0] * max_x) / w[1]
        d = 1.0 / np.sqrt((w.T * w)[0, 0])
        plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
        plt.plot([min_x, max_x], [y_min_x+d, y_max_x+d], '-g')
        plt.plot([min_x, max_x], [y_min_x-d, y_max_x-d], '-g')
        plt.show()


def test_1():
    data = []
    label = []
    with open('testSet.txt') as fp:
        for line in fp:
            line = line.strip().split('\t')
            data.append([float(item) for item in line[0:-1]])
            label.append(float(line[-1]))
    data = np.mat(data)
    label = np.mat(label).T
    train_x = data[0:80, :]
    train_y = label[0:80, :]
    test_x = data[80:100, :]
    test_y = label[80:100, :]
    C = 6
    toler = 0.000001
    max_iter = 50
    svm = SVM(train_x, train_y, C, toler, kernel_option=('linear', 1.0))
    svm.train_SVM_0(max_iter)
    accuracy = svm.test_SVM(test_x, test_y)
    print 'Accuracy: %.3f%%' % (accuracy * 100)
    svm.show_SVM()


def test_2():
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    with open('train.data') as fp:
        for line in fp:
            line = line.strip().split(' ')
            train_x.append([float(item) for item in line])
    with open('train.label') as fp:
        for line in fp:
            train_y.append(float(line))
    with open('test.data') as fp:
        for line in fp:
            line = line.strip().split(' ')
            test_x.append([float(item) for item in line])
    with open('test.label') as fp:
        for line in fp:
            test_y.append(float(line))
    train_x = np.mat(train_x)
    train_y = np.mat(train_y).T
    test_x = np.mat(test_x)
    test_y = np.mat(test_y).T

    C = 6
    toler = 0.00001
    max_iter = 50
    svm = SVM(train_x, train_y, C, toler, kernel_option=('linear', 1.0))
    svm.train_SVM_0(max_iter)
    accuracy = svm.test_SVM(test_x, test_y)
    print 'Accuracy: %.3f%%.' % (accuracy * 100)
    svm.show_SVM()

if __name__ == '__main__':
    test_2()









