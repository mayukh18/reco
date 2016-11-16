import pandas as pd

def load(dataset):

    if dataset == 'movielens100k':
        out = pd.read_csv('movielens100k.csv')
        return out

    elif dataset == 'movielens1m':
        out = pd.read_csv('movielens1m.csv')
        return out