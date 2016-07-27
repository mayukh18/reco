from math import sqrt





def rmse(true, predicted):

    true=list(true)
    predicted=list(predicted)
    error=list()

    error=[true[i]-predicted[i] for i in range(len(true))]
    mse=sum([error[i]*error[i] for i in range(len(error))])/len(error)
    value=sqrt(mse)

    return value