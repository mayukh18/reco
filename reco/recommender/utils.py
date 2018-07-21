import numpy as np
import pandas as pd
from copy import deepcopy

def super_str(x):

    if isinstance(x,np.int64):
        x=float(x)

    if isinstance(x,int):
        x=float(x)

    ans=str(x)

    return ans

def convert_to_array(x):

    if isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)

def special_sort(a, order='ascending'):
    n=len(a)

    if order=='ascending':
        for i in range(1,n):
            j=deepcopy(i)

            while j>0 and a[j][1]<a[j-1][1]:
                temp=a[j-1]
                a[j-1]=a[j]
                a[j]=temp

                j=j-1

    elif order=='descending':
        for i in range(1,n):
            j=deepcopy(i)

            while j>0 and a[j][1]>a[j-1][1]:
                temp=a[j-1]
                a[j-1]=a[j]
                a[j]=temp

                j=j-1
    return a

def dissimilarity(arr1, arr2, weighted):
    n=arr1.shape[0]
    s=0
    if weighted==True:
        for i in range(0,n):
            diff=abs(arr1[i]-arr2[i])
            s = s + (diff*(n-i)/n)
    else:
        for i in range(0,n):
            diff=abs(arr1[i]-arr2[i])
            s = s + (diff)
    return s


def create_utility_matrix(data, formatizer = {'user':0, 'item': 1, 'value': 2}):

    """

    :param data:            pandas dataframe, 2D, nx3
    :param formatizer:      dict having the column name or ids for users, items and ratings/values
    :return:                1. the utility matrix. (2D, n x m, n=users, m=items)
                            2. list of users (in order with the utility matrix rows)
                            3. list of items (in order with the utility matrix columns)
    """
    itemField = formatizer['item']
    userField = formatizer['user']
    valueField = formatizer['value']

    userList = data.ix[:,userField].tolist()
    itemList = data.ix[:,itemField].tolist()
    valueList = data.ix[:,valueField].tolist()

    users = list(set(data.ix[:,userField]))
    items = list(set(data.ix[:,itemField]))

    users_index = {users[i]: i for i in range(len(users))}

    pd_dict = {item: [np.nan for i in range(len(users))] for item in items}

    for i in range(0,len(data)):
        item = itemList[i]
        user = userList[i]
        value = valueList[i]

        pd_dict[item][users_index[user]] = value

    X = pd.DataFrame(pd_dict)
    X.index = users

    users = list(X.index)
    items = list(X.columns)

    return np.array(X), users, items