cimport cython
cimport numpy as np
import numpy as np
import pandas as pd

import time

@cython.boundscheck(False)
@cython.wraparound(False)

cdef class FM:
    cdef public dict indices
    cdef public double learning_rate, regularizer
    cdef public str solver
    cdef public int k, nz
    cdef public int iterations
    cdef public np.ndarray v
    cdef public np.ndarray w1
    cdef public double w0
    cdef public bint verbose
    def __init__(self,
                 k = 40,
                 learning_rate = 10,
                 solver='stochastic',
                 iterations = 100,
                 regularizer = 0.00035,
                 verbose = True):
        self.learning_rate = learning_rate
        self.solver=solver
        self.k = k
        self.iterations = iterations
        self.indices = {}
        self.regularizer = regularizer
        self.verbose = verbose
    def set_indices(self, X):
        i = 0
        cols = {}
        for col in X.columns:
            cols[col] = i
            i += 1

        indices = {}
        nz = 0
        for col in X.columns:
            if X[col].dtype == 'object':
                indices[cols[col]] = {}
                colset = set(X[col])
                for a in colset:
                    indices[cols[col]][a] = nz
                    nz += 1
            else:
                indices[cols[col]] = nz
                nz += 1
        self.indices = indices
        self.nz = nz

    def getIndexVal(self, col, val):
        if isinstance(self.indices[col], int) or isinstance(self.indices[col], float):
            return self.indices[col], val
        else:
            try:
                return self.indices[col][val], 1.
            except:
                #print(col, val)
                return 0., 0.

    def getIndexValArray(self, arr):
        out = []
        for i in range(len(arr)):
            out.append(self.getIndexVal(i, arr[i]))
        return out


    cdef initialize_params(self, X, y):
        self.set_indices(X)
        self.w0 = np.mean(y)
        self.w1 = np.zeros(self.nz)
        self.v = np.random.normal(scale=0.1,size=(self.nz, self.k))


    cdef sgd(self, np.ndarray X, np.ndarray y, verbose = True):

        cdef double learning_rate = self.learning_rate
        cdef double regularizer = self.regularizer
        cdef double w0 = self.w0
        cdef np.ndarray[np.double_t, ndim = 1] w1 = self.w1
        cdef np.ndarray[np.double_t, ndim = 2] v = self.v
        cdef int nz = self.nz

        cdef np.ndarray[np.double_t, ndim = 1] sum3
        cdef list _ivals, preds
        cdef int index, epoch
        cdef double sum1, sum2, s1, s2, val, res, b
        cdef np.ndarray _xvals
        cdef dict ind

        cdef int m = X.shape[0]
        cdef int n = X.shape[1]


        for epoch in range(self.iterations):
            start_time = time.time()

            preds = []
            for i in range(m):

                _x = X[i, :]
                _ivals = self.getIndexValArray(_x)
                ind = {}
                sum1 = 0
                sum2 = 0
                sum3 = np.zeros(self.k)

                for col in range(n):
                    index, val = _ivals[col]
                    ind[index] = val
                    sum1 += w1[index] * val

                for f in range(self.k):
                    s1 = 0.0
                    s2 = 0.0
                    for col in range(n):
                        index, val = _ivals[col]
                        temp = v[index, f] * val
                        s1 += temp
                        s2 += temp*temp
                    sum3[f] = s1
                    sum2 += s1*s1 - s2

                y_hat = w0 + sum1 + 0.5*sum2
                y_hat = max(1., y_hat)
                y_hat = min(5., y_hat)
                res = (y_hat - y[i])
                if self.verbose:
                    preds.append(abs(res)**2)

                b = learning_rate*regularizer
                # update rule for w0
                w0 = w0 - learning_rate * res - learning_rate * w0 * regularizer

                for col in range(n):
                    #index = col
                    #if col in ind:
                    #    val = ind[col]
                    index, val = _ivals[col]
                    temp = learning_rate * val * res
                    w1[index] -= (temp + b*w1[index])
                    for f in range(self.k):
                        v[index, f] -= (temp * (sum3[f] - v[index, f] * val) + b*v[index, f])
                    #else:
                    #    w1[index] -= b*w1[index]
                    #    for f in range(self.k):
                    #        v[index, f] -= b*v[index, f]

            print("epoch {} time {} mse {}".format(epoch, time.time()-start_time, np.mean(preds)))
        self.w0 = w0
        self.w1 = w1
        self.v = v

    def fit(self, X, y):
        """

        Args:
            matrix:
            verbose:

        Returns:

        """

        self.initialize_params(X, y)
        X = np.array(X)
        y = np.array(y)
        self.sgd(X, y)
        return

    def predict(self, X, verbose=True):

        X = np.array(X)

        cdef double w0 = self.w0
        cdef np.ndarray[np.double_t, ndim = 1] w1 = self.w1
        cdef np.ndarray[np.double_t, ndim = 2] v = self.v
        cdef int nz = self.nz

        cdef np.ndarray[np.double_t, ndim = 1] sum3
        cdef list _ivals, preds
        cdef int index, col, f, i
        cdef double sum1, sum2, s1, s2, val, y_hat, temp
        cdef np.ndarray _xvals

        cdef int m = X.shape[0]
        cdef int n = X.shape[1]

        preds = []
        for i in range(m):

            _x = X[i, :]
            _ivals = self.getIndexValArray(_x)
            ind = {}
            sum1 = 0
            sum2 = 0
            sum3 = np.zeros(self.k)

            for col in range(n):
                index, val = _ivals[col]
                ind[index] = val
                sum1 += w1[index] * val


            for f in range(self.k):
                s1 = 0
                s2 = 0
                for col in range(n):
                    index, val = _ivals[col]
                    temp = v[index, f] * val
                    s1 += temp
                    s2 += temp*temp
                sum3[f] = s1
                s1 = s1*s1
                sum2 += s1 - s2

            y_hat = w0 + sum1 + 0.5*sum2
            y_hat = max(1., y_hat)
            y_hat = min(5., y_hat)
            preds.append(y_hat)

        return preds
