import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
import random as alea
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS



""" Run nSimplices on HMP dataset """

colors = np.loadtxt("./data/hmp_colors.txt", dtype="str")
exec(open("./nsimplices.py").read())
alea.seed(42)

# """ QE normalization + MDS + cMDS """ 
# dir="./data/"
# data_path = dir+"hmp_v13lqphylotypeQuantE_rs_c.csv"

# df_hmp = np.loadtxt(data_path, delimiter=",")
# hmp_dis_sq=squareform(pdist(df_hmp))


# # Plot cMDS embedding using the pairs of axis from the four most significant axes 
# enforce_dim = 11 # enforcing dimension 11 to be consistent with QE+nsimplices+cMDS
# embedding = MDS(n_components=enforce_dim) 
# corr_coord = embedding.fit_transform(hmp_dis_sq)
# corr_dis_sq=squareform(pdist(corr_coord))
# _, _, Xe = cMDS(corr_dis_sq)

# np.savetxt("./outputs/hmp_QE_MDS_cMDS_axes.txt", Xe, fmt='%f')

# num_axes = 4

# for first_dim in range(num_axes):
#     for second_dim in range(first_dim+1, num_axes):
#         plt.figure()
#         for i in range(Xe.shape[0]):
#             plt.scatter(Xe[i, second_dim], Xe[i, first_dim], s=5, c=colors[i])
#         plt.legend(["QuantE"])
#         plt.savefig("./outputs/hmp_QE_MDS_cMDS_"+str(first_dim)+"_"+str(second_dim)+".png")


""" NB normalization + MDS + cMDS """ 
dir="./data/"
data_path = dir+"hmp_v13lqphylotypeQuantNB_rs_c.csv"

df_hmp = np.loadtxt(data_path, delimiter=",")
hmp_dis_sq=squareform(pdist(df_hmp))

# run MDS
enforce_dim = 11 # enforcing dimension 11 to be consistent with QE+nsimplices+cMDS
embedding = MDS(n_components=enforce_dim) 
corr_coord = embedding.fit_transform(hmp_dis_sq)
corr_dis_sq=squareform(pdist(corr_coord))

# run cMDS to get the corrected coordinates in importance decreasing order
_, _, Xe = cMDS(corr_dis_sq)
np.savetxt("./outputs/hmp_NB_MDS_cMDS_axes.txt", Xe, fmt='%f')

# Plot cMDS embedding using the pairs of axis from the four most significant axes 
num_axes = 4
for first_dim in range(num_axes):
    for second_dim in range(first_dim+1, num_axes):
        plt.figure()
        for i in range(Xe.shape[0]):
            plt.scatter(Xe[i, second_dim], Xe[i, first_dim], s=5, c=colors[i])
        plt.legend(["QuantE"])
        plt.savefig("./outputs/hmp_NB_MDS_cMDS_"+str(first_dim)+"_"+str(second_dim)+".png")