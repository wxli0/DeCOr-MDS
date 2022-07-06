#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
nSimplices on hmp
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import random as alea
from scipy.linalg import solve,pinv,pinv2
from scipy.spatial.distance import pdist, squareform
#from statsmodels.robust.scale import mad

exec(compile(open(r"lib/cMDS.py", encoding="utf8").read(), "cMDS.py", 'exec'))
exec(compile(open(r"lib/liblnSimplices-rejeu.py", encoding="utf8").read(), "liblnSimplices-rejeu.py", 'exec'))

# execfile("./lib/cMDS.py")
# execfile("./lib/liblnSimplices-rejeu.py")
lieudata="./data/"
QE=np.loadtxt(lieudata+"v13lqphylotypeQuantE.csv",delimiter=",")
# -> 2255 samples on 425 phylogenies
Qcolor = list(np.loadtxt(lieudata+"couleurshtmlQIHP", dtype='str'))
#
#couleursQIHP[THROAT==1] <- "deeppink" #"hotpink2"
#couleursQIHP[EARS==1] <- "black" 
#couleursQIHP[STOOL==1] <- "cornflowerblue"
#couleursQIHP[NOSE==1] <- "darkgreen"
#couleursQIHP[ELBOWS==1] <- "red"
#couleursQIHP[MOUTH==1] <- "gray" 
#couleursQIHP[VAGINA==1] <- "orange"
#
DE=pdist(QE) #2255*2255, euclidean distances
#va, ve, Xe = cMDS(squareform(DNBe))
#plt.scatter(Xe[:,0],Xe[:,1],s=10,facecolors='none',edgecolors=Qcolor);plt.title("cMDS(DNBe)") ; plt.show()
#va, ve, Xe = cMDS(squareform(DEe))
#plt.scatter(Xe[:,0],Xe[:,1],s=10,facecolors='none',edgecolors=Qcolor);plt.title("cMDS(DE)") ; plt.show()
#va, ve, Xe = cMDS(squareform(DEo))
#plt.scatter(Xe[:,0],Xe[:,1],s=10,facecolors='none',edgecolors=Qcoloro);plt.title("cMDS(DEo)") ; plt.show()

#DNBecorr,listmini = inequality_correction(DNBe)

data=squareform(DE)
np.shape(data) #(2255,2255)
#data=squareform(DR) : see analysis commands in (...) nSimplices_hmp_DE.py

alea.seed(42)

""" Données d'entrée et paramètres """
cutoff=0.5
trim=0.9

dim=20 # 5,10,15,20
resu=voldim_corrabb(data,cutoff,trim,ngmetric="rkurtosis",nmin=dim,nmax=dim)
np.savez("./resu/hhsJhn_"+str(dim),resu[0])
np.savez("./resu/hsignif_"+str(dim),resu[1])
np.savez("./resu/outliers_"+str(dim),resu[2])
np.savez("./resu/cdata_"+str(dim),resu[3])

#var = np.array(resu[0][3][0])**2 / (2*np.mean(data,0))
#print np.std(resu[0][3][0]) , np.std(resu[0][3][0] / np.sqrt(2*np.mean(data,0))), np.std( var ), 1.4826*np.median(abs(var-np.median(var)))

va, ve, Xe = cMDS(resu[3][3])
plt.plot(Xe[:,0],Xe[:,1],'.');plt.legend(["QuantE+nSimplices"])
plt.show()

