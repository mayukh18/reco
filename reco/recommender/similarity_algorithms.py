# calculates the similarity coefficient (pearson coefficient)################



def sim_pearson(prefs,user1,user2):
    # get the list of mutually rated items
    list={}

    if user1 not in prefs or user2 not in prefs:
        return 0.5

    for item in prefs[user1]:
        if item in prefs[user2]:
            ###
            #print item,prefs[user1][item],prefs[user2][item]
            list[item]=1

    # number of mutually rated elements
    n=len(list)

    # no element in common
    if n==0: return 0

    # sum the mutual ratings
    sum1=sum([prefs[user1][it] for it in list])
    sum2=sum([prefs[user2][it] for it in list])

    # sum up the squares
    sum1sq=sum([pow(prefs[user1][it],2) for it in list])
    sum2sq=sum([pow(prefs[user2][it],2) for it in list])

    # sum of products
    psum=sum([prefs[user1][it]*prefs[user2][it] for it in list])

    # calculate pearson score
    num=psum-(sum1*sum2/n)
    den=pow(((sum1sq-pow(sum1,2)/n)*(sum2sq-pow(sum2,2)/n)),0.5)


    if den==0: return 0

    r=num/den
    r=min(r,1.0)
    return r


def sim_jaccard(prefs,user1,user2):

    count=0
    for item in prefs[user1]:
        if item in prefs[user2]:
            count+=1

    n1=len(prefs[user1])
    n2=len(prefs[user2])
    count=float(count)

    r=float(count/(n1+n2-count))

    return r

def sim_cosine(prefs,user1,user2):
    list={}
    num=0
    den=0
    for item in prefs[user1]:
        if item in prefs[user2]:
            list[item]=1
            num+=prefs[user1][item]*prefs[user2][item]

    if num==0:
        return 0

    den= pow( sum(pow(prefs[user1][item],2) for item in list)*sum(pow(prefs[user2][item],2) for item in list) , 0.5 )
    num=float(num)

    return num/den

