import numpy as np
import pandas as pd
from .utils import super_str, convert_to_array, special_sort, dissimilarity
from math import sqrt











class SVDRecommender:

    """
    Singular Value Decomposition is an important technique used in recommendation systems.
    Using SVD, the complete utility matrix is decomposed into user and item features.
    Thus the dimensionality of the matrix is reduced and we get the most important features
    neglecting the weaker ones.

    The utility matrix is initially sparse having a lot of missing values. The missing values
    are filled in using the mean for that item.

    no_of_features: the number of the biggest features that are to be taken for each user and
                    item. default value is 15.

    method: 1. default: The mean for the item is deducted from the user-item pair value in
                        the utility matrix. SVDRecommender is computed. With the computed values, the
                        mean for the item is added back to get the final result.

    formatizer: a dict having the keys 'user', 'item' and 'value' each having an integer value
                that denotes the column numbers of the corresponding things in the array
                provided in the fit and predict method. The 'value' will be used only in the
                fit method.


    Attributes:
                instantiation outside init:
                no_of_users=int()
                no_of_items=int()
                user_index=list()
                item_index=list()
                Usk=None
                skV=None
    """


    def __init__(self,
                 no_of_features=15,
                 method='default',
                 ):
        self.parameters={"no_of_features", "method"}
        self.method = method
        self.no_of_features = no_of_features


    def get_params(self, deep=False):
        out=dict()
        for param in self.parameters:
            out[param]=getattr(self, param)

        return out


    def set_params(self, **params):

        for a in params:
            if a in self.parameters:
                setattr(self,a,params[a])
            else:
                raise AttributeError("No such attribute exists to be set")






    def create_utility_matrix(self, data, formatizer = {'user':0, 'item': 1, 'value': 2}):

        """

        :param dataset_array:   Array-like, 2D, nx3
        :param indices:         pass the formatizer
        :return:                the utility matrix. 2D, n x m, n=users, m=items
        """
        itemField = formatizer['item']
        userField = formatizer['user']
        valueField = formatizer['value']

        userList = data.loc[:,userField].tolist()
        itemList = data.loc[:,itemField].tolist()
        valueList = data.loc[:,valueField].tolist()

        users = list(set(data.loc[:,userField]))
        items = list(set(data.loc[:,itemField]))

        users_index = {users[i]: i for i in range(len(users))}



        pd_dict = {item: [np.nan for i in range(len(users))] for item in items}

        for i in range(0,len(data)):
            item = itemList[i]
            user = userList[i]
            value = valueList[i]

            pd_dict[item][users_index[user]] = value
            #print i

        X = pd.DataFrame(pd_dict)
        X.index = users

        users = list(X.index)
        items = list(X.columns)

        return np.array(X), users, items


    def fit(self, user_item_matrix, userList, itemList):



        """
        :param X: nx3 array-like. Each row has three elements in the order given by the
                  formatizer. The userId, itemId and the value/rating.

               formatizer: to change the default format

        :return: Does not return anything. Just fits the data and forms U, s, V by SVDRecommender
        """

        self.users = list(userList)
        self.items = list(itemList)

        self.user_index = {self.users[i]: i for i in range(len(self.users))}
        self.item_index = {self.items[i]: i for i in range(len(self.items))}


        mask=np.isnan(user_item_matrix)
        masked_arr=np.ma.masked_array(user_item_matrix, mask)

        self.predMask = ~mask

        self.item_means=np.mean(masked_arr, axis=0)
        self.user_means=np.mean(masked_arr, axis=1)
        self.item_means_tiled = np.tile(self.item_means, (user_item_matrix.shape[0],1))

        # utility matrix or ratings matrix that can be fed to svd
        self.utilMat = masked_arr.filled(self.item_means)

        # for the default method
        if self.method=='default':
            self.utilMat = self.utilMat - self.item_means_tiled


        # Singular Value Decomposition starts
        # k denotes the number of features of each user and item
        # the top matrices are cropped to take the greatest k rows or
        # columns. U, V, s are already sorted descending.

        k = self.no_of_features
        U, s, V = np.linalg.svd(self.utilMat, full_matrices=False)

        U = U[:,0:k]
        V = V[0:k,:]
        s_root = np.diag([sqrt(s[i]) for i in range(0,k)])

        self.Usk=np.dot(U,s_root)
        self.skV=np.dot(s_root,V)
        self.UsV = np.dot(self.Usk, self.skV)

        self.UsV = self.UsV + self.item_means_tiled




    def predict(self, X, formatizer = {'user': 0, 'item': 1}):
        """

        :param X: the test set. 2D, array-like consisting of two eleents in each row
                  corresponding to the userId and itemId

               formatizer: to change the default format

        :return: 1D, a list giving the value/rating corresponding to each user-item
                 pair in each row of X.
        """

        users = X.loc[:,formatizer['user']].tolist()
        items = X.loc[:,formatizer['item']].tolist()

        if self.method == 'default':

            values = []
            for i in range(len(users)):
                user = users[i]
                item = items[i]

                # user and item in the test set may not always occur in the train set. In these cases
                # we can not find those values from the utility matrix.
                # That is why a check is necessary.
                # 1. both user and item in train
                # 2. only user in train
                # 3. only item in train
                # 4. none in train

                if user in self.user_index:
                    if item in self.item_index:
                        values.append( self.UsV[self.user_index[user], self.item_index[item]] )
                    else:
                        values.append( self.user_means[ self.user_index[user] ] )

                elif item in self.item_index and user not in self.user_index:
                    values.append( self.item_means[self.item_index[item] ])

                else:
                    values.append(np.mean(self.item_means)*0.6 + np.mean(self.user_means)*0.4)

        return values


    def topN_similar(self, x, column='item', N=10, weight=True):

        """
        Gives out the most similar contents compared to the input content given. For an user input gives out similar
        users. For an item input, gives out the most similar items.

        :param x: the identifier string for the user or item.
        :param column: either 'user' or 'item'
        :param N: The number of best matching similar content to output
        :param weight: True or False. True means the feature differences are weighted. Puts more penalty on the differences
        between bigger features.

        :return: A list of tuples.
        """
        out=list()

        if column=='user':
            if x not in self.user_index:
                raise Exception("Invalid user")
            else:
                for user in self.user_index:
                    if user != x:
                        temp = dissimilarity(self.Usk[self.user_index[user],:], self.Usk[self.user_index[x],:], weighted=weight)
                        out.append((user, temp))
        if column=='item':
            if x not in self.item_index:
                raise Exception("Invalid item")
            else:
                for item in self.item_index:
                    if item != x:
                        temp = dissimilarity(self.skV[:, self.item_index[item]], self.skV[:, self.item_index[x]], weighted=weight)
                        out.append((item, temp))

        out = special_sort(out, order='ascending')
        out = out[:N]
        return out



    def recommend(self, users_list, N=10, values = False):

        # utilMat element not zero means that element has already been
        # discovered by the user and can not be recommended
        predMat = np.ma.masked_where(self.predMask, self.UsV).filled(fill_value=-999)
        out = []

        if values == True:
            for user in users_list:
                try:
                    j = self.user_index[user]
                except:
                    raise Exception("Invalid user:", user)
                max_indices = predMat[j,:].argsort()[-N:][::-1]
                out.append( [(self.items[index],predMat[j,index]) for index in max_indices ] )

        else:
            for user in users_list:
                try:
                    j = self.user_index[user]
                except:
                    raise Exception("Invalid user:", user)
                max_indices = predMat[j,:].argsort()[-N:][::-1]
                out.append( [self.items[index] for index in max_indices ] )


        return out


