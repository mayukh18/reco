import pandas as pd
a = pd.read_csv("movieLens.csv")

from CollaborativeFiltering import CollaborativeFiltering

b= CollaborativeFiltering(formatizer={'user': 'userId', 'item': 'movieName', 'value': 'rating'})

from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

x,y = train_test_split(a, test_size= 0.1)

x.reset_index(inplace=True)
y.reset_index(inplace=True)

b.fit(data=x)
m= y['rating'].tolist()
n= b.predict(X=y)
print mean_squared_error(m, n)
out = []
true= []
for i in range(len(m)):
    print m[i], n[i]
    if n[i]!=2.5:
        out.append(n[i])
        true.append(m[i])
print mean_squared_error(true, out)

"""

bb=transformPrefs(ratings)
print topMatches(bb, 'Remember Me (2010)',score=True,sim_engine='pearson')
print topMatches(bb, 'Remember Me (2010)',score=True,sim_engine='jaccard')
print sim_cosine(bb, 'Remember Me (2010)','What Would Jesus Buy? (2007)')
"""
"""
rank={}
for i in range(len(pp)):
    rank[i]=pp[i][1]
    print rank[i]
"""

#print ratings['199']
#k=whatsHot(ratings,time_value=5000000,personalized=1,user='344')

#print k[0:10]
#print ratings['216']

