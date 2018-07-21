import random
from metrics import  rmse
import numpy as np
from svd import SVDRecommender





def chunk(xs, n):
    ys = list(xs)
    random.shuffle(ys)
    ylen = len(ys)
    size, leftover = divmod(ylen, n)
    chunks = [ys[size*i : size*(i+1)] for i in xrange(n)]
    edge = size*n
    for i in xrange(leftover):
        chunks[i%n].append(ys[edge+i])
    return chunks


def cross_val_score(model=None, data=None, cv=10, scorer=rmse):

    data=np.array(data)
    print(data.shape)
    chunks=chunk(data, cv)
    #print chunks
    score=list()

    for i in range(10):

        iter_data=list()
        for j in range(len(chunks)):
            if j!=i:
                iter_data.extend(chunks[j])

        pred_data=np.array(chunks[i])
        iter_data=np.array(iter_data)



        model.fit(iter_data)
        pred=model.predict(pred_data)
        score.append(rmse(pred_data[ : , model.formatizer['value']], pred))
        print(score[i])

    return np.mean(score)





