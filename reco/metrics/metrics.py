from math import sqrt
from scipy.stats import kendalltau





def rmse(true, predicted):

    true=list(true)
    predicted=list(predicted)
    error=list()

    error=[true[i]-predicted[i] for i in range(len(true))]
    mse=sum([error[i]*error[i] for i in range(len(error))])/len(error)
    value=sqrt(mse)

    return value

def kendalltau(rankA, rankB):

    if len(rankA) != len(rankB):
        raise TypeError("The two rank lists must be of the same length.")

    N = len(rankA)

    if isinstance(rankA[0], tuple):
        rankA = [rankA[i][0] for i in range(N)]

    if isinstance(rankB[0], tuple):
        rankB = [rankB[i][0] for i in range(N)]

    listA = [i for i in range(N)]
    listB = [rankB.index(rankA[i]) for i in range(N)]

    return kendalltau(listA, listB)[0]

