from reco.recommender import SVDRecommender
from reco.datasets import load_movielens


data = load_movielens()
svd = SVDRecommender(no_of_features=12)
data = svd.create_utility_matrix(data)
svd.fit(data)
users = [1, 65, 444, 321]
print(svd.recommend(users, N=4))

