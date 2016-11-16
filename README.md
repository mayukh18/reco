# reco
## a simple yet versatile recommendation systems library in python

Currently it has:
  1. similarity based collaborative filtering module
  2. svd module

Still a lot of improvements, modifications and additions are pending.

It is available on pypi, use pip to install

`$ pip install reco`

Full documentation available [here](http://reco.readthedocs.io/en/master/)

### Demo / Example
#### SVDRecommender

```
from reco.recommender import SVDRecommender
from reco.datasets import load_movielens


data = load_movielens()
svd = SVDRecommender(no_of_features=12)
user_item_matrix = svd.create_utility_matrix(data, formatizer={'user':'userId', 'item':'movieId', 'value':'rating'})
svd.fit(user_item_matrix)

test_users = [1, 65, 444, 321]

results = svd.recommend(test_users, N=4)
print(results)

```