import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import random as alea
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS


""" Prepare colors (only once) """
if not os.path.exists("./data/colors.txt"):
    print("======== loading colors ========")
    color_df = pd.read_csv("./data/v13lqphylotypePheno_QIHP.csv", header=0)
    colors = []

    print(color_df)
    for index, row in color_df.iterrows():
        if row["THROAT"]:
            colors.append("deeppink")
        elif row['EARS']:
            colors.append("black")
        elif row["STOOL"]:
            colors.append("cornflowerblue")
        elif row["NOSE"]:
            colors.append("darkgreen")
        elif row["ELBOWS"]:
            colors.append("red")
        elif row["MOUTH"]:
            colors.append("gray")
        elif row["VAGINA"]:
            colors.append("orange")
    print("colors len is:", len(colors))

    colors = np.array(colors)

    np.savetxt("./data/colors.txt", colors, fmt="%s")


""" Run nSimplices on HMP dataset """

colors = np.loadtxt("./data/hmp_colors.txt", dtype="str")
exec(open("./nsimplices.py").read())
alea.seed(42)

""" NB normalization + nSimplices + cMDS """ 
axes_output_path = "./outputs/hmp_NB_nSimplices_cMDS_axes.txt"
if not os.path.exists(axes_output_path):
    print("======== NB normalization + nSimplices + cMDS ========")
    dir="./data/"
    data_path = dir+"hmp_v13lqphylotypeQuantNB_rs_c.csv"
    df_hmp = np.loadtxt(data_path, delimiter=",")
    hmp_dis_sq=squareform(pdist(df_hmp))

    feature_num = 11
    dim_start = 11
    dim_end = feature_num

    print("hmp_dis_sq shape is:", hmp_dis_sq.shape)
    outlier_indices, subspace_dim , corr_pairwise_dis, corr_coord = nsimplices(hmp_dis_sq, feature_num, dim_start, dim_end)
    print("subspace dimension is:", subspace_dim)

    # run cMDS to get the corrected coordinates in importance decreasing order
    _, _, Xe = cMDS(corr_pairwise_dis)
    np.savetxt(axes_output_path, Xe, fmt='%f')


""" QE normalization + nSimplices + cMDS """ 
print("======== NB normalization + nSimplices + cMDS ========")
dir="./data/"
data_path = dir+"hmp_v13lqphylotypeQuantE_rs_c.csv"

df_hmp = np.loadtxt(data_path, delimiter=",")
hmp_dis_sq=squareform(pdist(df_hmp))

feature_num = 11
dim_start = 11
dim_end = feature_num

print("hmp_dis_sq shape is:", hmp_dis_sq.shape)
outlier_indices, subspace_dim , corr_pairwise_dis, corr_coord = nsimplices(hmp_dis_sq, feature_num, dim_start, dim_end)
print("subspace dimension is:", subspace_dim)

""" (1) Plot cMDS embedding using the pairs of axis from the four most significant axes """
va, ve, Xe = cMDS(corr_pairwise_dis)
axes_output_path = "./outputs/hmp_QE_nSimplices_cMDS_axes.txt"

np.savetxt(axes_output_path, Xe, fmt='%f')


figure_output_path = "./outputs/hmp_QE_nSimplices_cMDS_0_1.png"
if not os.path.exists(figure_output_path):
    print("======== plot pairwise 2D plot ========")
    num_axes = 3 # show pairwise 2D plot to decompose the 3D plot
    for first_dim in range(num_axes):
        for second_dim in range(first_dim+1, num_axes):
            plt.figure()
            # only plot stool (blue), ears (black), throad (pink) points 
            for i in range(Xe.shape[0]):
                plt.scatter(Xe[i, second_dim], Xe[i, first_dim], s=5, c=colors[i])
            plt.legend(["QuantE+nSimplices"])
            plt.savefig("./outputs/hmp_QE_nSimplices_cMDS_"+str(first_dim)+"_"+str(second_dim)+".png")


figure_output_path = "./outputs/hmp_QE_nSimplices_cMDS_subset_0_1.png"
if not os.path.exists(figure_output_path):
    print("======== plot pairwise 2D plot (subset) ========")
    num_axes = 3 # show pairwise 2D plot to decompose the 3D plot
    for first_dim in range(num_axes):
        for second_dim in range(first_dim+1, num_axes):
            plt.figure()
            # only plot stool (blue), ears (black), throad (pink) points 
            for i in range(Xe.shape[0]):
                if colors[i] not in ["cornflowerblue", "black", "orange"]:
                    continue
                plt.scatter(Xe[i, second_dim], Xe[i, first_dim], s=5, c=colors[i])
            plt.legend(["QuantE+nSimplices"])
            plt.savefig("./outputs/hmp_QE_nSimplices_cMDS_subset_247_"+str(first_dim)+"_"+str(second_dim)+".png")