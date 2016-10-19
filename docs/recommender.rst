.. something to write



**************
Recommenders
**************

There are 2 recommenders available

* SVDRecommender
* CFRecommender

CFRecommender
==============

This is the collaborative filtering module. It has 3 similarity coefficients available to choose from, namely pearson, jaccard and cosine. ::

    reco.recommender.CFRecommender(formatizer = {'user':0,'item':1,'value':2}, sim_engine = 'pearson')

Parameters
-----------

**formatizer** :
^^^^^^^^^^^^^^^^

The format of the datasets that the recommender will be working upon. The values corresponding to every key is the column number for that key. For example the default format is that user_ids are in first column, then item_ids and then the corresponding ratings or values.

**sim_engine** :
^^^^^^^^^^^^^^^^

The similarity coefficient for measuring the similarity between entities (users or items). The available options are 'pearson','jaccard','cosine'.


Methods
--------

**fit**:
^^^^^^^^
Fits the data to the recommender. ::

    fit(data, formatizer)

*Parameters*

* data: The dataset.
* formatizer: Overrides the format during initialization for this fit method.

**predict**:
^^^^^^^^^^^^

Predicts the rating or value for each user-item pair in the rows of the X. ::

    predict(X, formatizer=None)

*Parameters*

* X: The dataset for prediction.
* formatizer: Overrides the format during initialization for this predict method.

**getRecommendations**:
^^^^^^^^^^^^^^^^^^^^^^^

Recommends top N items for the user given (reducing N does not improve speed)::

    getRecommendations(user, score=True, N=10)

*Parameters*

* user: The user_id for which to recommend.
* score: Whether or not to give out the predicted rating for each item. If score is True, output is a list of tuples.
* N: Number of items to recommend.


**topMatches**:
^^^^^^^^^^^^^^^^^^^^^^^

Gives the top N similar users for the user given (reducing N does not improve speed)::

    topMatches(self, user, score = True, N=10)

*Parameters*

* user: The user_id for which to process.
* score: Whether or not to give out the similarity coefficient for each user. If score is True, output is a list of tuples.
* N: Number of users to find.




SVDRecommender
==============

This is the singular vector decomposition module. It breaks the dataset as feature vectors of each user and each item.

    reco.recommender.SVDRecommender(no_of_features = 15, method = 'default', formatizer = {'user':0,'item':1,'value':2})

Parameters
-----------

**formatizer** :
^^^^^^^^^^^^^^^^

The format of the datasets that the recommender will be working upon. The values corresponding to every key is the column number for that key. For example the default format is that user_ids are in first column, then item_ids and then the corresponding ratings or values.

**method** :
^^^^^^^^^^^^^^^^

'default' is the only option at this moment.

**no_of_features** :
^^^^^^^^^^^^^^^^

The number of features in the feature vectors of each user and each item that the dataset is decomposed into.

Methods
--------

**fit**:
^^^^^^^^
Fits the data to the recommender. ::

    fit(X, formatizer = None)

*Parameters*

* X: The dataset.
* formatizer: Overrides the format during initialization for this fit method.

**predict**:
^^^^^^^^
Predicts the rating or value for each user-item pair in the rows of the X as a list. ::

    fit(X, formatizer = None)

*Parameters*

* X: The dataset.
* formatizer: Overrides the format during initialization for this predict method.


**topN_similar**:
^^^^^^^^
Predicts the rating or value for each user-item pair in the rows of the X as a list. ::

    topN_similar(x, column='item', N=10, weight=True)

*Parameters*

* x: The id for the user or item.
* column: 'item' if x is an item or 'user' if x is an user.
* N: Number of similar entities to find.
* weight: Give the associateds weights of similarity.

**topN_predict**:
^^^^^^^^
Predicts the rating or value for each user-item pair in the rows of the X as a list. ::

    topN_predict(user, N=10)

*Parameters*

* user: The user_id for which to recommend.
* N: Number of items to recommend.





