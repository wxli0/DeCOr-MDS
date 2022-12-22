import numpy as np
import matplotlib.pyplot as plt
import os
import random as alea
from scipy.linalg import solve,pinv,pinv2
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from sklearn.decomposition import PCA
import time

# import nSimplices 
exec(compile(open(r"nsimplices.py", encoding="utf8").read(), "nsimplices.py", 'exec'))

# set matplotlib default savefig directory
plt.rcParams["savefig.directory"] = os.getcwd() # To save figures to directory
                                                #   defined above


# Read in dataset of main dimension 40
df_dim40 = pd.read_csv(r'data/synthetic_rdim40.csv',sep=';',header=None)
df_dim40.head()

# simulate outliers and calculate subspace dimensions
outlier_indices_gap = 0.2
res_outlier_indices_list = np.arange(0, 1, outlier_indices_gap)
prop_incre = 0.02 # simulate 2% more outliers per iteration
dim_pred_diff = []
dim_raw_diff= []
true_dim = 40
num_components = 50
df_prev = df_dim40
for i in range(len(res_outlier_indices_list)):
    res_outlier_indices = \
        range(int(res_outlier_indices_list[i] * df_prev.shape[0]), \
            int((res_outlier_indices_list[i]+outlier_indices_gap) * df_prev.shape[0]))
    df_prev = sim_outliers(df_prev, prop_incre, 40, 45, \
        res_outlier_indices = res_outlier_indices)
    out_dis=pdist(df_prev) # pairwise distance in tab (with outliers added)
    out_dis_sq=squareform(out_dis) # squared matrix form of D
    subspace_dim, outlier_indices = find_subspace_dim(out_dis_sq, 30, df_prev.shape[1])
    dim_pred_diff.append(subspace_dim - true_dim)
    dim_raw_diff.append(subspace_dim + int((subspace_dim * len(outlier_indices)/df_prev.shape[0]) - true_dim))
    print("subspace_dim is:", subspace_dim)


plt.figure(figsize=(7, 5.2))
props = np.arange(prop_incre, prop_incre+len(dim_pred_diff)*prop_incre, prop_incre)
plt.scatter(props, dim_pred_diff, color = "red", s=10)
plt.plot(props, dim_pred_diff, c="red", label = "after correction")
plt.scatter(props, dim_raw_diff, color = "black", s=10)
plt.plot(props, dim_raw_diff, c="black", label = "before correction")
plt.axhline(y=0, color='black', linestyle='dotted', label="baseline")
plt.xticks(props)
plt.xlabel(r'fraction of outliers $p$', fontsize=15)
plt.ylabel(r'$\bar{n}-d^{*}$', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=16)
plt.savefig("./outputs/synthetic_dim40.png")