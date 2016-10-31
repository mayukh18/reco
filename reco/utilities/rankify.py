



def rankify(X1, X2=None, merge=False, formatizer = {'user':0,'item':1,'value':2}):

    """


    Args:
        X1: Dataset 1
        X2: Dataset 2, used only when merge = True
        merge: If True, merges datasets X1 and X2
        formatizer:

    Returns: a dict having, for every user, a ranklist of items that has a valid value for the corresponding user in the dataset.

    """

