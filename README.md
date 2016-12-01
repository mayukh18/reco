# reco
## a simple yet versatile recommendation systems library in python

Currently it has:
  1. svd module
  2. similarity based collaborative filtering module
  3. metrics
  4. datasets

It is available on pypi, use pip to install

`$ pip install reco`

### Demo / Example
#### SVDRecommender

```
from reco.recommender import SVDRecommender
from reco.datasets import load_movielens

data = load_movielens()

svd = SVDRecommender(no_of_features=4)

# Creates the user-item matrix, the userIds on the rows and the itemIds on the columns.
user_item_matrix, users, items = svd.create_utility_matrix(data, formatizer={'user':'userId', 'item':'movieId', 'value':'rating'})

# fits the svd model to the matrix data.
svd.fit(user_item_matrix, users, items)

##### TESTING #####

test_users = [1, 65, 444, 321]

# recommends 4 undiscovered items per each user
results = svd.recommend(test_users, N=4)

# outputs 5 most similar users to user with userId 65
similars = svd.topN_similar(x=65, N=5, column='user')

print(results)
print(similars)

```

Full documentation available [here](http://reco.readthedocs.io/en/master/)
