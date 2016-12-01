

********
Metrics
********

reco has some metrics that you can use to assess the performance of your recommenders.

**rmse**:
^^^^^^^^^^^^^^
Root Mean Square Error. It is useful in assessing the performance of the predict methods, i.e. in assessing the accuracy of the predicted ratings. ::

    reco.metrics.rmse(true, predicted)

* true: True value list.
*predicted: predicted value list.

**kendalltau**:
^^^^^^^^^^^^^^^

Measures the Kendal-Tau correlation between 2 ranked lists. Useful in assessing the performance of the recommend methods, i.e. the accuracy of the orders in which the items are recommended. For example, better or more relevant items should be at the start. ::

    reco.metrics.kendalltau(rankA, rankB)

rankA and rankB must be of the same length. Kendal Tau measures the similarity in the order of elements in the 2 ranked lists.