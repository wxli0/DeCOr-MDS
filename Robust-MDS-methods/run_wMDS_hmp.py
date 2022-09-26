#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from scipy.spatial.distance import pdist, squareform
import os
import numpy as np
exec(open("./nsimplices.py").read())
exec(open("./Robust-MDS-methods/wMDS.py").read())

data_path = os.path.join("./data/hmp_v13lqphylotypecounts_rs_c.csv")
df_hmp = np.loadtxt(data_path, delimiter=",")
    
# wMDS
# print "wcCounts mds"
similarity_matrix =wcCounts_similDist(np.transpose(df_hmp), dist=True, simpl=True)
print("output data shape is:", similarity_matrix.shape)
vawMDS, vewMDS, XewMDS =  cMDS(similarity_matrix)

np.savetxt("./outputs/hmp_wMDS_coord.txt", XewMDS, fmt='%f')
   
    
    




