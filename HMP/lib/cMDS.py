#!/usr/bin/python

# Classical multidimensional scaling

import numpy as np
import sys

def cMDS(D,alreadyCentered=False):
    """
    D: distance / dissimilarity matrix, square and symetric, diagonal 0
    """
    (p,p2)=np.shape(D)
    if p != p2:
        sys.exit("D must be symetric...")
        
    # Double centering
    if not alreadyCentered:
        J=np.eye(p)-np.ones((p,p))/p
        Dcc=-0.5*np.dot(J,np.dot(D**2,J))
    else: 
        # allows robust centering with mediane and mad
        # outside this routine
        Dcc=D
    
    # Eigenvectors
    evals, evecs = np.linalg.eigh(Dcc)
    
    # Sort by eigenvalue in decreasing order
    idx = np.argsort(abs(evals))[::-1]
    #idx = np.argsort(evals)[::-1]
    evecst = evecs[:,idx]
    evalst= evals[idx]    
    
    # Undelying coordinates 
    idxPos,=np.where(evalst>0)
    Xe=np.dot(evecst[:,idxPos],np.diag(evalst[idxPos]**0.5))
    
    return evalst[idxPos], evecst[:,idxPos], Xe
    
    
"""
# check:
D=squareform(np.array([522,522,358,9219,5,879,9650,876,9649,9080.]))
#D=np.array([[0,1,2.],[1,0,4.5],[2,4.5,0.]])
Xe=cMDS(D)
print np.allclose(D,pdist(Xe))


import numpy as np
from sklearn import manifold
from sklearn.metrics import euclidean_distances
n_samples = 20
seed = np.random.RandomState(seed=3)
X_true = seed.randint(0, 20, 2 * n_samples).astype(np.float)
X_true = X_true.reshape((n_samples, 2))
# Center the data
# inutile : X_true -= X_true.mean()

D = euclidean_distances(X_true)


mds = manifold.MDS(n_components=20, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(D).embedding_
"""


