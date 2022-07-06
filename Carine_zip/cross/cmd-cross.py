#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
nSimplices on cross example
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import random as alea
from scipy.linalg import solve,pinv,pinv2
from scipy.spatial.distance import pdist, squareform

os.chdir(".")

# Librairie nSimplices courante
exec(compile(open("2021-04-01_nSimplices-lib.py", encoding="utf8").read(), "2021-04-01_nSimplices-lib.py", 'exec'))
#execfile("./2021-04-01_nSimplices-lib.py")

# set matplotlib default savefig directory
plt.rcParams["savefig.directory"] = os.getcwd() # To save figures to directory
                                                #   defined above
# test data
axe1True = [i for i in range(13)]+[6 for i in range(12)]
axe2True = [6 for i in range(13)]+[i for i in range(6)]+[i for i in range(7,13)]

# axe1True = [1,5,3,7,100,2,2,3,5,2,15,3,1]+[6 for i in range(12)]
# axe2True = [6,1,3,2,1,17,5,9,8,8,1,1,14]+[i for i in range(6)]+[i for i in range(7,13)]

#plt.plot(axe1True,axe2True,'.');plt.show()
""" euclidean distances """
D=pdist(np.array([axe1True,axe2True]).T)

""" A few outliers along new axis """
seed= 112 ; alea.seed(seed)
sD=squareform(D)
N=np.shape(sD)[0]
pc=0.05 #proportion of outliers
k=int(np.ceil(pc*N))
# Tirage aléatoire de quelques points hors plan
indices=np.sort(alea.sample(range(N),k))
DSO=1.*sD
for n in indices:
    horsplan=50*alea.random()
    print ("n,horsplan:"+str(n)+","+str(horsplan))
    for m in [x for x in range(N) if x !=n]:
        DSO[n,m]=DSO[m,n]=np.sqrt(DSO[n,m]**2+horsplan**2)

""" n = 3 , DSO """
n=3
Vn=[]
seed=245124512 ; alea.seed(seed)
for i in range(1000):
    indices=alea.sample(range(N),n+1)
    Vn.append(nSimplexVolume(indices,DSO))

#plt.hist(Vn) ; plt.show() # majorité nuls ou presque, un très petit nb ressort
#                          #   (comme ci-dessus, surement dû au bruit).


""" Parameters and formatting of input data """
cutoff=0.5
trim=0.9

# En entrée : DSO, qui contient quelques outliers hors-plan
lDSO=squareform(DSO) # shape DSO as other matrices, i.e. as a N*(N-1)/2-sized flat matrix.
data=squareform(lDSO)    #squareform( DNd ) #squareform(DNne)
#(D+1.*np.array(Noise)) ou 1e-6,1e-5,... 1., 10. *Noise
#D, Dd, DO, DOd, DSO, DNd. (Ddne)


""" Applications of nSimplices :
    - dimension detection
    - outlier detection
    - outliers are projected into relevant dimension
    - result : distance matrix to be used in classical MDS, for instance.
"""
print("\n Application of nSimplex \n ")
resu=nSimplwhichdim(data,cutoff,trim,ngmetric="rkurtosis")
var = np.array(resu[0][3][0])**2 / (2*np.mean(data,0))
print (np.std(resu[0][3][0]) , np.std(resu[0][3][0] / np.sqrt(2*np.mean(data,0))), np.std( var ), 1.4826*np.median(abs(var-np.median(var))))


""" cMDS and plot """

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,3))
#fig.suptitle('Horizontally stacked subplots')
#
va, ve, Xe = cMDS(squareform(D))
ax1.plot(Xe[:,0],Xe[:,1],'.')
ax1.set_title("TRUE")
#
va2, ve2, Xe2 = cMDS(DSO)
ax2.plot(Xe2[:,0],Xe2[:,1],'.', color='orange')
ax2.set_title("Contaminated")
#
va3, ve3, Xe3 = cMDS(resu[3][3])
ax3.plot(Xe3[:,0],Xe3[:,1],'.', color='green')
ax3.set_title("Restituted")
plt.show()


