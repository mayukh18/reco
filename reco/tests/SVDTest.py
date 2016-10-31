import sys
import numpy as np
sys.path.append("../recommender/")
import pandas as pd
from svd import SVDRecommender

data = pd.read_csv("../datasets/movielens100k.csv")
#print
test_users = [1,34,412,645]

users = data['userId'].tolist()
items = data['movieId'].tolist()

mak = [(users[i], items[i]) for i in range(len(users))]

print (645,2284) in mak

# [[4439, 5607], [60806, 3655], [95856, 95170], [2284, 82739]]
"""
a = pd.DataFrame()
ax = ['a','b','c']
b = [0 for i in range(3)]
for i in range(len(ax)):

a.index = [1,2,3]
print a

"""

svd = SVDRecommender()
a = svd.create_utility_matrix(data)
svd.fit(a)
#print "fit done"
out = svd.recommend(test_users,N=5, values=True)

print out
