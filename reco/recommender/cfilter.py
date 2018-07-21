import pandas as pd
from .similarity_algorithms import sim_pearson,sim_jaccard,sim_cosine



class CFRecommender:

    def __init__(self,
                 formatizer = {'user':0,'item':1,'value':2},
                 sim_engine = 'pearson'):
        self.formatizer = formatizer
        self.ratings = {}


        if sim_engine == 'cosine':
            self.engine = sim_cosine
        elif sim_engine == 'jaccard':
            self.engine = sim_jaccard
        elif sim_engine == 'pearson':
            self.engine = sim_pearson
        else:
            raise AttributeError("Invalid similarity engine. Choose between 'pearson', 'jaccard' and 'cosine'")



    def fit(self, data, formatizer = None):

        if formatizer != None:
            self.formatizer = formatizer

        ratings=dict()

        count=0
        past_user=-999
        for i in range(0,len(data)):
            item=str(data.ix[i][self.formatizer['item']])
            user=str(data.ix[i][self.formatizer['user']])
            if user!=past_user:
                ratings[user]={}
            count=count+1

            ratings[user][item]=float(data.ix[i][self.formatizer['value']])
            past_user=user

        self.ratings = ratings


    def predict_single(self, user, item):

        totals=0
        simSums=0

        for other in self.ratings:
            if other == user: continue
            sim= self.engine(self.ratings, user, other)

            if sim<=0: continue

            if item in self.ratings[other]:
                totals += self.ratings[other][item]*sim
                simSums += sim

        if simSums == 0:
            return 2.5
        return totals/simSums



    def predict(self, X, formatizer = None):

        if formatizer != None:
            self.formatizer = formatizer

        out = []

        for i in range(len(X)):
            ans = self.predict_single(X.ix[i][self.formatizer['user']], X.ix[i][self.formatizer['item']])
            out.append(ans)

        return out


    def getRecommendations(self, user, score=True, N=10):

        prefs = self.ratings

        totals={}
        simSums={}

        for other in prefs:
            if other == user: continue
            sim = self.engine(prefs,user,other)

            if sim<=0: continue

            for item in prefs[other]:
                # only items haven't yet rated by user
                if item not in prefs[user]:
                    totals.setdefault(item,0)
                    totals[item] += prefs[other][item]*sim
                    simSums.setdefault(item,0)
                    simSums[item] += sim

        # normalized list
        rankings=[(totals[item]/simSums[item], item) for item in totals]

        rankings.sort()
        rankings.reverse()
        if score==True:
            return rankings[:N]
        else:
            rank_pure={}
            for i in range(len(rankings)):
                rank_pure[i]=rankings[i][1]
            return rank_pure[:N]




    def topMatches(self, user, score = True, N=10):

        scores_=[(self.engine(self.ratings,user,other),other) for other in self.ratings if other!=user]

        scores_.sort()
        scores_.reverse()

        if score==True:
            return scores_[:N]
        else:
            rank_pure={}
            for i in range(len(scores_)):
                rank_pure[i]=scores_[i][1]
            return rank_pure[:N]
