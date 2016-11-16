import sys
import numpy as np
sys.path.append("../recommender/")
import pandas as pd
from svd import SVDRecommender

data = pd.read_csv("../datasets/movielens100k.csv")
test_users = [1,34,412,645]

users = data['userId'].tolist()
items = data['movieId'].tolist()

mak = [(users[i], items[i]) for i in range(len(users))]
#print mak[:100]

svd = SVDRecommender()
a = svd.create_utility_matrix(data)
mask = np.isnan(a)
masked_arr = np.ma.masked_array(a, mask)
dummy = np.zeros([a.shape[0], a.shape[1]])
#print dummy

xx1 = masked_arr[1,:12]
#print xx1
xx2 = np.array([i for i in range(12)])

result = np.ma.masked_where(~mask, dummy).filled(fill_value=-999)
print result

#print "fit done"
#out = svd.recommend(test_users,N=5, values=True)

#print out
