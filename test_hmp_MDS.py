import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
import random as alea
from scipy.spatial.distance import pdist, squareform

# read colors
colors = np.loadtxt("./data/hmp_colors.txt", dtype="str")
exec(open("./nsimplices.py").read())
alea.seed(42)

""" NB normalization + MDS """ 
dir="./data/"
data_path = dir+"hmp_v13lqphylotypeQuantNB_rs.csv"
df_hmp = np.loadtxt(data_path, delimiter=",")
hmp_dis_sq=squareform(pdist(df_hmp))

""" (0) Plot MDS embedding using the pairs of axis from the four most significant axes """
va, ve, Xe = MDS(hmp_dis_sq)
np.savetxt("./outputs/hmp_NB_MDS_Xe"+".txt", Xe, fmt='%f')

""" QE normalization + MDS """ 
dir="./data/"
data_path = dir+"hmp_v13lqphylotypeQuantE_rs.csv"
df_hmp = np.loadtxt(data_path, delimiter=",")
hmp_dis_sq=squareform(pdist(df_hmp))

""" (0) Plot MDS embedding using the pairs of axis from the four most significant axes """
va, ve, Xe = MDS(hmp_dis_sq)
np.savetxt("./outputs/hmp_QE_MDS_Xe"+".txt", Xe, fmt='%f')