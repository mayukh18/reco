from reco.recommender import SVDRecommender
from reco.datasets import load
from reco.metrics import rmse


data = load(dataset='movielens100k')

print(data)
