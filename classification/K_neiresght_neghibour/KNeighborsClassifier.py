
X = [[0], [1], [2], [3]]
Y = [0, 0, 1, 1]
###"""""""""""""""""""""""""
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, Y)
print(neigh.predict([[1.1]]))
print(neigh.predict_proba([[0.9]]))


####""""""""""""""""""""""""""""""""""""""
samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
from sklearn.neighbors import  NearestNeighbors
neigh2 = NearestNeighbors(n_neighbors=1)

#train our model****************
neigh2.fit(samples)
print(neigh2.kneighbors([[1., 1., 1.]]))

#********************************************#
X = [[0], [3], [1]]
neigh3 = NearestNeighbors(n_neighbors=1)
neigh3.fit(X)
A = neigh.kneighbors_graph(X)
print(A.toarray())
