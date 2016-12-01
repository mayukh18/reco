from reco.recommender import SVDRecommender
from reco.datasets import load_movielens
import numpy as np

data = load_movielens()
#print(data[30:])
svd = SVDRecommender(no_of_features=4)
user_item_matrix, users, items = svd.create_utility_matrix(data, formatizer={'user':'userId', 'item':'movieId', 'value':'rating'})

#x = np.array(user_item_matrix)

#print(user_item_matrix)

svd.fit(user_item_matrix, users, items)

test_users = [1, 65, 444, 321]

results = svd.recommend(test_users, N=4)
#similars = svd.topN_similar(x=65, N=5, column='user')
print(results)
#print(similars)