

*************************
SVD Recommender Tutorial
*************************

Below is a tutorial on using the SVD recommender module.::

    from reco.recommender import SVDRecommender
    from reco.datasets import load_movielens
    import numpy as np

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

    ##### The End #####
