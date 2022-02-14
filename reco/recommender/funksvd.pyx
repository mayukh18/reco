cimport cython
cimport numpy as np
import numpy as np
from utils import create_utility_matrix

@cython.boundscheck(False)
@cython.wraparound(False)

cdef class FunkSVD:
    cdef public int k
    cdef public int iterations
    cdef public str method
    cdef public double learning_rate
    cdef public double regularizer
    cdef public bint bias

    cdef public double global_mean
    cdef public list users, items
    cdef public dict userdict, itemdict
    cdef public np.ndarray userfeatures
    cdef public np.ndarray itemfeatures
    cdef public np.ndarray user_bias
    cdef public np.ndarray item_bias
    cdef public np.ndarray mask

    def __init__(self,
                 k=32,
                 iterations = 100,
                 learning_rate = 0.00001,
                 regularizer = 0.000001,
                 method = 'batch',
                 bias=True):

        self.k = k
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.regularizer = regularizer
        self.method = method
        self.bias = bias

    def fit(self, X, formatizer={'user':0, 'item':1, 'value':2}, verbose = True):

        if self.method == 'batch':
            try:
                self.batchGD(X, formatizer, verbose)
            except:
                raise

        elif self.method == 'stochastic':
            self.stochasticGD(X, formatizer, verbose)

        else:
            raise AttributeError('Invalid method. Valid methods are batch and stochastic')

    def batchGD(self, X, formatizer, verbose):

        cdef np.ndarray[np.double_t, ndim=2] userfeatures, itemfeatures, res, user_bias, item_bias, y_hat
        cdef double global_mean
        cdef double regularizer = self.regularizer
        cdef double learning_rate = self.learning_rate
        cdef int epoch
        cdef dict userindexes, itemindexes

        utilmatrix, self.users, self.items = create_utility_matrix(data=X,formatizer=formatizer)
        m, n = utilmatrix.shape

        self.mask = np.isnan(utilmatrix)
        masked_arr = np.ma.array(utilmatrix, mask=self.mask)
        utilmatrix = masked_arr.filled(0)

        N = np.sum(~self.mask)
        global_mean = np.ma.mean(masked_arr)

        userfeatures = np.random.random((len(self.users), self.k))/self.k
        itemfeatures = np.random.random((len(self.items), self.k))/self.k
        user_bias = np.reshape(np.ma.mean(masked_arr, axis=1) - global_mean, (-1,1))
        item_bias = np.reshape(np.ma.mean(masked_arr, axis=0) - global_mean, (1, -1))

        import time
        for epoch in range(self.iterations):
            #start_time = time.clock()
            y_hat = np.dot(userfeatures, itemfeatures.T) + global_mean + user_bias + item_bias
            y_hat_mask = np.ma.array(y_hat, mask=self.mask)
            y_hat = y_hat_mask.filled(0)

            res = utilmatrix - y_hat

            if verbose:
                error = np.ma.sum(np.ma.abs(res)) / N
                print("Epoch "+str(epoch)+": Error: "+str(error))

            userfeatures += learning_rate * (np.dot(res, itemfeatures) - regularizer * userfeatures)
            itemfeatures += learning_rate * (np.dot(res.T, userfeatures) - regularizer * itemfeatures)

            incr_user_bias = np.reshape(np.ma.mean(np.ma.array(res - regularizer * user_bias, mask=self.mask), axis=1), \
                                        (m,1))
            incr_item_bias = np.reshape(np.ma.mean(np.ma.array(res - regularizer * item_bias, mask=self.mask), axis=0), \
                                        (1,n))

            user_bias += learning_rate * incr_user_bias
            item_bias += learning_rate * incr_item_bias
            #print("Epoch {} time {}".format(epoch, time.clock()-start_time))

        userindexes = {self.users[i]:i for i in range(len(self.users))}
        itemindexes = {self.items[i]:i for i in range(len(self.items))}

        self.userfeatures = userfeatures
        self.itemfeatures = itemfeatures
        self.global_mean = global_mean
        self.user_bias = user_bias
        self.item_bias = item_bias
        self.userdict = userindexes
        self.itemdict = itemindexes

    def stochasticGD(self, X, formatizer, verbose):

        itemField = formatizer['item']
        userField = formatizer['user']
        valueField = formatizer['value']

        X = X[[userField, itemField, valueField]]

        cdef list users, items

        users = list(set(X.loc[:, userField]))
        items = list(set(X.loc[:, itemField]))

        cdef double global_mean = np.mean(X.loc[:, valueField])
        cdef double learning_rate = self.learning_rate
        cdef double regularizer = self.regularizer
        cdef np.ndarray[np.double_t, ndim=2] userfeatures = np.random.random((len(users), self.k))/self.k
        cdef np.ndarray[np.double_t, ndim=2] itemfeatures = np.random.random((len(items), self.k))/self.k
        cdef np.ndarray[np.double_t, ndim=2] user_bias = np.zeros((len(users), 1))
        cdef np.ndarray[np.double_t, ndim=2] item_bias = np.zeros((1, len(items)))

        cdef int i, f, userindex, itemindex
        cdef double res, ui_dot, y_hat, r, error
        cdef dict userindexes, itemindexes
        cdef list ratings

        userindexes = {users[i]:i for i in range(len(users))}
        itemindexes = {items[i]:i for i in range(len(items))}

        ratings = [(userindexes[x[0]], itemindexes[x[1]], x[2]) for x in X.values]

        N = len(ratings)

        for epoch in range(self.iterations):
            error = 0
            for userindex,itemindex,r in ratings:
                res = 0

                if self.bias == True:
                    ui_dot = 0
                    for f in range(self.k):
                        ui_dot += userfeatures[userindex, f]*itemfeatures[itemindex, f]
                    y_hat = ui_dot + global_mean + user_bias[userindex, 0] + item_bias[0, itemindex]
                    res = r - y_hat
                    user_bias[userindex, 0] += learning_rate * (res - regularizer * user_bias[userindex, 0])
                    item_bias[0, itemindex] += learning_rate * (res - regularizer * item_bias[0, itemindex])

                else:
                    y_hat = 0
                    for f in range(self.k):
                        y_hat += userfeatures[userindex, f]*itemfeatures[itemindex, f]
                    res = r - y_hat - global_mean

                error += abs(res)

                for f in range(self.k):
                    userfeatures[userindex, f] += learning_rate * (res * itemfeatures[itemindex, f] - regularizer * userfeatures[userindex, f])
                    itemfeatures[itemindex, f] += learning_rate * (res * userfeatures[userindex, f] - regularizer * itemfeatures[itemindex, f])

            if verbose:
                error = error / N
                print("Epoch " + str(epoch) + ": Error: " + str(error))

        self.users = users
        self.items = items
        self.userfeatures = userfeatures
        self.itemfeatures = itemfeatures
        self.global_mean = global_mean
        self.user_bias = user_bias
        self.item_bias = item_bias
        self.userdict = userindexes
        self.itemdict = itemindexes

    def predict(self, X, formatizer = {'user': 0, 'item': 1}, verbose=False):
        """

        :param X: the test set. 2D, array-like consisting of two eleents in each row
                  corresponding to the userId and itemId

               formatizer: to change the default format

        :return: 1D, a list giving the value/rating corresponding to each user-item
                 pair in each row of X.
        """
        cdef list testusers, testitems, users, items
        cdef dict userdict, itemdict
        cdef np.ndarray[np.double_t, ndim=2] userfeatures = self.userfeatures
        cdef np.ndarray[np.double_t, ndim=2] itemfeatures = self.itemfeatures
        cdef np.ndarray[np.double_t, ndim=2] user_bias = self.user_bias
        cdef np.ndarray[np.double_t, ndim=2] item_bias = self.item_bias
        cdef float global_mean = self.global_mean

        testusers = X[formatizer['user']].tolist()
        testitems = X[formatizer['item']].tolist()

        users = self.users
        items = self.items
        userdict = self.userdict
        itemdict = self.itemdict

        # user and item in the test set may not always occur in the train set. In these cases
        # we can not find those values from the utility matrix.
        # That is why a check is necessary.
        # 1. both user and item in train
        # 2. only user in train
        # 3. only item in train
        # 4. none in train

        cdef list predictions
        cdef int userindex, itemindex
        predictions = []

        import time
        start_time = time.clock()

        if self.bias:

            for i in range(len(testusers)):
                user = testusers[i]
                item = testitems[i]
                if user in userdict and item in itemdict:
                    userindex = userdict[user]
                    itemindex = itemdict[item]
                    ssum = 0
                    for f in range(self.k):
                        ssum += userfeatures[userindex, f]*itemfeatures[itemindex, f]
                    pred = ssum + global_mean + user_bias[userindex, 0] + item_bias[0, itemindex]
                    predictions.append(ssum + global_mean + user_bias[userindex, 0] + item_bias[0, itemindex])
                elif user in userdict:
                    predictions.append(global_mean + user_bias[userdict[user], 0])
                elif item in itemdict:
                    predictions.append(global_mean + item_bias[0, itemdict[item]])
                else:
                    predictions.append(global_mean)

                #predictions.append(pred)

        else:

            for i in range(len(testusers)):
                user = testusers[i]
                item = testitems[i]

                if user in users and item in items:
                    userindex = userdict[user]
                    itemindex = itemdict[item]
                    pred = global_mean + np.sum(userfeatures[userindex] * itemfeatures[itemindex])
                elif user in users:
                    userindex = userdict[user]
                    pred = global_mean + np.sum(userfeatures[userindex] * np.mean(itemfeatures, axis=0))
                elif item in items:
                    itemindex = itemdict[item]
                    pred = global_mean + np.sum(itemfeatures[itemindex] * np.mean(userfeatures, axis=0))
                else:
                    pred = global_mean

                predictions.append(pred)

        print("time taken {} secs".format(time.clock() - start_time))

        return predictions
