# reco
## a simple yet versatile recommendation systems library in python

Currently it has:
  1. user-user / item-item collaborative filtering
  2. SVD ( based on scipy's implementation )
  3. Simon Funk's SVD
  4. Factorization Machine

Install the latest version by running

`pip install git+https://github.com/mayukh18/reco.git`

Or you can install (old version) from PyPI by running

`pip install reco`


References:

1. Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl. Item-Based Collaborative Filtering Recommendation Algorithms, 2001.[[pdf]](http://files.grouplens.org/papers/www10_sarwar.pdf)

2. Simon Funk's Blog. Netflix Update: Try This at Home. [[link]](https://sifter.org/simon/journal/20061211.html)

3. Arkadiusz Paterek. Improving regularized singular value decomposition for collaborative filtering, 2007. [[pdf]](https://www.cs.uic.edu/~liub/KDD-cup-2007/proceedings/Regular-Paterek.pdf)

4. Steffen Rendle. Factorization Machines, 2010. [[pdf]](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)


#### Two small examples to get you started

**FunkSVD**

```
from reco.recommender import FunkSVD
from reco.metrics import rmse
from reco.datasets import loadMovieLens100k

train, test, _, _ = loadMovieLens100k(train_test_split=True)

f = FunkSVD(k=64, learning_rate=0.002, regularizer = 0.05, iterations = 150, method = 'stochastic', bias=True)

# fits the model to the data
f.fit(X=train, formatizer={'user':'userId', 'item':'itemId', 'value':'rating'},verbose=True)

# predicts the ratings from the test set
preds = f.predict(X=test, formatizer={'user':'userId', 'item':'itemId'})
print(rmse(preds, list(test['rating']))
```

**SVDRecommender**

```
from reco.recommender import SVDRecommender
from reco.datasets import loadMovieLens100k
from reco.metrics import rmse

train, test, _, _ = loadMovieLens100k(train_test_split=True)

svd = SVDRecommender(no_of_features=8)

# Creates the user-item matrix, the userIds on the rows and the itemIds on the columns.
user_item_matrix, users, items = svd.create_utility_matrix(train, formatizer={'user':'userId', 'item':'itemId', 'value':'rating'})

# fits the svd model to the matrix data.
svd.fit(user_item_matrix, users, items)

# predict the ratings from test set
preds = svd.predict(test, formatizer = {'user':'userId', 'item': 'itemId'})
print(rmse(preds, list(test['rating'])))

test_users = [1, 65, 444, 321]
# recommends 4 undiscovered items per each user
results = svd.recommend(test_users, N=4)

# outputs 5 most similar users to user with userId 65
similars = svd.topN_similar(x=65, N=5, column='user')
```



More examples are available [here](https://github.com/mayukh18/reco/tree/master/examples)

Partial documentation available [here](https://reco.readthedocs.io/en/master/)