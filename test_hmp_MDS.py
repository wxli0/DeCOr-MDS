import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
import random as alea
from scipy.spatial.distance import pdist, squareform

# read colors
colors = np.loadtxt("./data/hmp_colors.txt", dtype="str")
exec(open("./nsimplices.py").read())
alea.seed(42)

dir="./data/"

""" NB normalization + cMDS """ 

data_path = os.path.join(dir, "hmp_v13lqphylotypeQuantNB_rs_c.csv")
df_hmp = np.loadtxt(data_path, delimiter=",")
hmp_dis_sq=squareform(pdist(df_hmp))

""" (0) Plot cMDS embedding using the pairs of axis from the four most significant axes """
va, ve, Xe = cMDS(hmp_dis_sq)
np.savetxt("./outputs/hmp_NB_cMDS_coord"+".txt", Xe, fmt='%f')

""" QE normalization + cMDS """ 
data_path = os.path.join(dir, "hmp_v13lqphylotypeQuantE_rs_c.csv")
df_hmp = np.loadtxt(data_path, delimiter=",")
hmp_dis_sq = squareform(pdist(df_hmp))

""" (0) Plot cMDS embedding using the pairs of axis from the four most significant axes """
va, ve, Xe = cMDS(hmp_dis_sq)
np.savetxt("./outputs/hmp_QE_cMDS_coord.txt", Xe, fmt='%f')

""" MDS using Mahattan distance (MDSm) """
data_path = os.path.join(dir, "hmp_v13lqphylotypecounts_rs_c.csv")
df_hmp = np.loadtxt(data_path, delimiter=",")
dist = manhattan_distances(df_hmp)
mds = MDS(dissimilarity='precomputed', random_state=0) # TODO: check with Carine, how to determine the number of components
dist_trans = mds.fit_transform(dist)
np.savetxt("./outputs/hmp_MDSm_coord.txt", dist_trans, fmt='%f')

""" MDS using Euclidean distance (MDSe) """
data_path = os.path.join(dir, "hmp_v13lqphylotypecounts_rs_c.csv")
df_hmp = np.loadtxt(data_path, delimiter=",")
dist = manhattan_distances(df_hmp)
mds = MDS(dissimilarity='precomputed', random_state=0) # TODO: check with Carine, how to determine the number of components
dist_trans = mds.fit_transform(dist)
np.savetxt("./outputs/hmp_MDSe_coord.txt", dist_trans, fmt='%f')

""" Robust MDS (RMDS) """

""" Robust MDS for strucuture outliers (RSMDS) """

""" Weighted MDS (wMDS) """

