import numpy as np
from utils import super_str, convert_to_array, special_sort, dissimilarity
from scipy.linalg import sqrtm











class SVDRecommender(object):

    """
    Singular Value Decomposition is an important technique used in recommendation systems.
    Using SVD, the complete utility matrix is decomposed into user and item features.
    Thus the dimensionality of the matrix is reduced and we get the most important features
    neglecting the weaker ones.

    The utility matrix is initially sparse having a lot of missing values. The missing values
    are filled in using the mean for that item.

    no_of_features: the number of the biggest features that are to be taken for each user and
                    item. default value is 15.

    method: 1. default: The mean for the item and the SVDRecommender prediction for the user-item pair
                        from SVDRecommender are added with multiplication factor coeff.
                        Refer to coeff.
            2. 'zero': The mean for the item is deducted from the user-item pair value in
                        the utility matrix. SVDRecommender is computed. With the computed values, the
                        mean for the item is added back to get the final result.

    formatizer: a dict having the keys 'user', 'item' and 'value' each having an integer value
                that denotes the column numbers of the corresponding things in the array
                provided in the fit and predict method. The 'value' will be used only in the
                fit method.

    coeff:  used only in the default method. This is the proportion of the value predicted
            by the SVDRecommender method that is added to (1-coeff)*(mean for the item) to get the
            final prediction.

            final_prediction = coeff*svd_prediction + (1-coeff)*(mean for the item)

            default value is 0.25

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
                 coeff=0.25,
                 formatizer={'user':0,'item':1,'value':2}
                 ):
        self.parameters={"no_of_features", "method",
                     "coeff"}
        self.method = method
        self.coeff = coeff
        self.no_of_features = no_of_features
        self.formatizer = formatizer
        self.item_means=list()
        self.user_means=list()


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






    def create_utility_matrix(self,dataset_array,indices):

        """

        :param dataset_array:   Array-like, 2D, nx3
        :param indices:         pass the formatizer
        :return:                the utility matrix. 2D, n x m, n=users, m=items
        """
        itemField=indices['item']
        userField=indices['user']
        valueField=indices['value']

        values=dict()
        past_user=list()
        past_item=list()

        count=0
        for i in range(0,len(dataset_array)):
            item=str(dataset_array[i,itemField])
            user=str(dataset_array[i,userField])

            if item not in past_item:
                past_item.append(item)

            if user not in past_user:
                values[user]={}
                past_user.append(user)
            count=count+1

            values[user][item]=float(dataset_array[i,valueField])

        # assess size of the utility matrix
        self.no_of_users=len(past_user)
        self.no_of_items=len(past_item)
        self.dictX=values

        utilMat=np.empty((self.no_of_users,self.no_of_items))
        utilMat[:]=np.nan


        for user in values:
            for item in values[user]:
                utilMat[past_user.index(user),past_item.index(item)]=values[user][item]

        self.user_index=past_user
        self.item_index=past_item

        return utilMat


    def fit(self, X, formatizer = None):

        """
        :param X: nx3 array-like. Each row has three elements in the order given by the
                  formatizer. The userId, itemId and the value/rating.

               formatizer: to change the default format

        :return: Does not return anything. Just fits the data and forms U, s, V by SVDRecommender
        """
        if formatizer != None:
            self.formatizer = formatizer

        X=convert_to_array(X)
        X=self.create_utility_matrix(X,self.formatizer)

        mask=np.isnan(X)
        masked_arr=np.ma.masked_array(X, mask)

        self.item_means=np.mean(masked_arr, axis=0)
        self.user_means=np.mean(masked_arr, axis=1)

        # the all important utility matrix or ratings matrix

        matrix = masked_arr.filled(self.item_means)

        # for the ZERO method
        if self.method=='zero':
            for i in range(matrix.shape[1]):
                matrix[:,i] = matrix[:,i] - self.item_means[i]


        # Singular Value Decomposition starts
        # k denotes the number of features of each user and item
        # the top matrices are cropped to take the greatest k rows or
        # columns. U, V, s are already sorted descending.

        k=self.no_of_features
        U, s, V=np.linalg.svd(matrix, full_matrices=False)
        s=np.diag(s)
        s=s[0:k,0:k]
        U=U[:,0:k]
        V=V[0:k,:]

        s_root=sqrtm(s)

        self.Usk=np.dot(U,s_root)
        self.skV=np.dot(s_root,V)




    def predict(self, X, formatizer = None):
        """

        :param X: the test set. 2D, array-like consisting of two eleents in each row
                  corresponding to the userId and itemId

               formatizer: to change the default format

        :return: 1D, a list giving the value/rating corresponding to each user-item
                 pair in each row of X.
        """
        if formatizer != None:
            self.formatizer = formatizer

        X=convert_to_array(X)
        values=list()

        if self.method=='zero':
            for i in range(len(X)):
                user=super_str(X[i,self.formatizer['user']])
                item=super_str(X[i,self.formatizer['item']])

                # user and item in the test set may not always occur in the train set. In these cases
                # we can not find those values from the utility matrix.
                # That is why a check is necessary.
                # 1. both user and item in train
                # 2. only user in train
                # 3. only item in train
                # 4. none in train

                if user in self.user_index:
                    if item in self.item_index:
                        temp=np.dot(self.Usk[self.user_index.index(user),:], self.skV[:,self.item_index.index(item)])
                        temp = temp + self.user_means[self.user_index.index(user)]
                        values.append(temp)
                    else:
                        values.append(self.user_means[self.user_index.index(user)])
                elif item in self.item_index and user not in self.user_index:
                    values.append(self.item_means[self.item_index.index(item)])
                else:
                    values.append(np.mean(self.item_means)*0.6 + np.mean(self.user_means)*0.4)

        elif self.method=='default':
            for i in range(len(X)):
                user=super_str(X[i,self.formatizer['user']])
                item=super_str(X[i,self.formatizer['item']])

                # user and item in the test set may not always occur in the train set. In these cases
                # we can not find those values from the utility matrix.
                # That is why a check is necessary.
                # 1. both user and item in train
                # 2. only user in train
                # 3. only item in train
                # 4. none in train

                if user in self.user_index:
                    if item in self.item_index:
                        temp=np.dot(self.Usk[self.user_index.index(user),:], self.skV[:,self.item_index.index(item)])
                        temp = temp * self.coeff + self.user_means[self.user_index.index(user)] * (1-self.coeff)
                        values.append(temp)
                    else:
                        values.append(self.user_means[self.user_index.index(user)])
                elif item in self.item_index and user not in self.user_index:
                    values.append(self.item_means[self.item_index.index(item)])
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
                raise KeyError("Invalid user")
            else:
                for user in self.user_index:
                    if user!=x:
                        temp=dissimilarity(self.Usk[self.user_index.index(user),:], self.Usk[self.user_index.index(x),:], weighted=weight)
                        out.append((user, temp))
        if column=='item':
            if x not in self.item_index:
                raise KeyError("Invalid item")
            else:
                for item in self.item_index:
                    if item!=x:
                        temp=dissimilarity(self.skV[:, self.item_index.index(item)], self.skV[:, self.item_index.index(x)], weighted=weight)
                        out.append((item, temp))

        out=special_sort(out, order='ascending')
        out=out[:N]
        return out



    def topN_predict(self, user, N=10):
        out=list()
        if user not in self.user_index:
            raise KeyError("Invalid user")
        else:
            for item in self.item_index:
                    if item not in self.dictX[user]:
                        temp=np.dot(self.Usk[self.user_index.index(user),:], self.skV[:,self.item_index.index(item)])
                        out.append((item, temp))

        out=special_sort(out,order='descending')
        out=out[:N]
        return out


