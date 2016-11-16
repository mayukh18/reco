import pandas as pd
from os.path import join
from os.path import dirname

def load_movielens():

    module_path = dirname(__file__)
    full_filename = join(module_path, 'movielens100k.csv')

    out = pd.read_csv(full_filename)
    return out