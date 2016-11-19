from reco.recommender import SVDRecommender
from reco.datasets import load_movielens


data = load_movielens()
svd = SVDRecommender(no_of_features=4)
user_item_matrix = svd.create_utility_matrix(data, formatizer={'user':'userId', 'item':'movieId', 'value':'rating'})
svd.fit(user_item_matrix)

test_users = [1, 65, 444, 321]

print(svd.recommend(test_users, N=4))

