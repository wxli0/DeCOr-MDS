#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from scipy.spatial.distance import pdist, squareform
import os
import numpy as np
exec(open("./nsimplices.py").read())
exec(open("./Robust-MDS-methods/wMDS.py").read())

data_path = os.path.join("./data/hmp_v13lqphylotypecounts_rs.csv")
df_hmp = np.loadtxt(data_path, delimiter=",")
D = squareform(pdist(df_hmp))
    
# wMDS
# print "wcCounts mds"
D=wcCounts_similDist(df_hmp, dist=True, simpl=True)
vawMDS, vewMDS, XewMDS =  cMDS(D)

np.savetxt("./outputs/hmp_wMDS_coord.txt", XewMDS, fmt='%f')
   
    
    




