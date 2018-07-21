cimport cython
cimport numpy as np
import numpy as np
from utils import create_utility_matrix

@cython.boundscheck(False)
@cython.wraparound(False)

class FunkSVD:
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
        import time

        itemField = formatizer['item']
        userField = formatizer['user']
        valueField = formatizer['value']

        X = X[[userField, itemField, valueField]]
        self.users = list(set(X.ix[:, userField]))
        self.items = list(set(X.ix[:, itemField]))

        cdef double global_mean = np.mean(X.ix[:, valueField])
        cdef list users = self.users
        cdef list items = self.items
        cdef double learning_rate = self.learning_rate
        cdef double regularizer = self.regularizer
        cdef np.ndarray[np.double_t, ndim=2] userfeatures = np.zeros((len(users), self.k))/self.k
        cdef np.ndarray[np.double_t, ndim=2] itemfeatures = np.zeros((len(items), self.k))/self.k
        cdef np.ndarray[np.double_t, ndim=2] user_bias = np.zeros((len(users), 1))
        cdef np.ndarray[np.double_t, ndim=2] item_bias = np.zeros((1, len(items)))

        cdef int i, f, userindex, itemindex
        cdef double res, ui_dot, y_hat, r, error
        cdef dict userindexes, itemindexes
        cdef list ratings

        if self.bias == True:
            for i in range(len(self.users)):
                # check if this works
                user = self.users[i]
                user_bias[i,0] = np.mean(X[X[userField] == user][valueField]) - global_mean
            for i in range(len(self.items)):
                # check if this works
                item = self.items[i]
                item_bias[0,i] = np.mean(X[X[itemField] == item][valueField]) - global_mean

        userindexes = {self.users[i]:i for i in range(len(self.users))}
        itemindexes = {self.items[i]:i for i in range(len(self.items))}

        ratings = [(userindexes[x[0]], itemindexes[x[1]], x[2]) for x in X.values]

        N = len(ratings)
        print("N {}".format(N))
        exec_time = []

        for epoch in range(self.iterations):
            error = 0
            #start_time = time.clock()
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
                #print(error)

                for f in range(self.k):
                    userfeatures[userindex, f] += learning_rate * (res * itemfeatures[itemindex, f] - regularizer * userfeatures[userindex, f])
                    itemfeatures[itemindex, f] += learning_rate * (res * userfeatures[userindex, f] - regularizer * itemfeatures[itemindex, f])

            if verbose:
                error = error / N
                print("Epoch " + str(epoch) + ": Error: " + str(error))
                #print("Epoch " + str(epoch))

        self.userfeatures = userfeatures
        self.itemfeatures = itemfeatures
        self.global_mean = global_mean
        self.user_bias = user_bias
        self.item_bias = item_bias
        self.userdict = userindexes
        self.itemdict = itemindexes

    def predict(self, X, formatizer = {'user': 0, 'item': 1}):
        """

        :param X: the test set. 2D, array-like consisting of two eleents in each row
                  corresponding to the userId and itemId

               formatizer: to change the default format

        :return: 1D, a list giving the value/rating corresponding to each user-item
                 pair in each row of X.
        """

        users = X[formatizer['user']].tolist()
        items = X[formatizer['item']].tolist()


        # user and item in the test set may not always occur in the train set. In these cases
        # we can not find those values from the utility matrix.
        # That is why a check is necessary.
        # 1. both user and item in train
        # 2. only user in train
        # 3. only item in train
        # 4. none in train

        predictions = []

        import time
        start_time = time.clock()

        if self.bias:

            for i in range(len(users)):
                user = users[i]
                item = items[i]

                if user in self.users and item in self.items:
                    userindex = self.userdict[user]
                    itemindex = self.itemdict[item]
                    pred = np.sum(self.userfeatures[userindex] * self.itemfeatures[itemindex]) + self.global_mean + \
                        self.user_bias[userindex, 0] + self.item_bias[0, itemindex]
                elif user in self.users:
                    userindex = self.users.index(user)
                    pred = self.global_mean + self.user_bias[userindex, 0]
                elif item in self.items:
                    itemindex = self.items.index(item)
                    pred = self.global_mean + self.item_bias[0, itemindex]
                else:
                    pred = self.global_mean

                predictions.append(pred)

        else:

            for i in range(len(users)):
                user = users[i]
                item = items[i]

                if user in self.users and item in self.items:
                    userindex = self.userdict[user]
                    itemindex = self.itemdict[item]
                    pred = self.global_mean + np.sum(self.userfeatures[userindex] * self.itemfeatures[itemindex])
                elif user in self.users:
                    userindex = self.users.index(user)
                    pred = self.global_mean + np.sum(self.userfeatures[userindex] * np.mean(self.itemfeatures, axis=0))
                elif item in self.items:
                    itemindex = self.items.index(item)
                    pred = self.global_mean + np.sum(self.itemfeatures[itemindex] * np.mean(self.userfeatures, axis=0))
                else:
                    pred = self.global_mean

                predictions.append(pred)

        print("time {}".format(time.clock() - start_time))

        return predictions
