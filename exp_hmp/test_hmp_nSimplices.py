import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
import random as alea
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS


""" Prepare colors (only once) """

# color_df = pd.read_csv("./data/v13lqphylotypePheno_QIHP.csv", header=0)
# colors = []

# print(color_df)
# for index, row in color_df.iterrows():
#     if row["THROAT"]:
#         colors.append("deeppink")
#     elif row['EARS']:
#         colors.append("black")
#     elif row["STOOL"]:
#         colors.append("cornflowerblue")
#     elif row["NOSE"]:
#         colors.append("darkgreen")
#     elif row["ELBOWS"]:
#         colors.append("red")
#     elif row["MOUTH"]:
#         colors.append("gray")
#     elif row["VAGINA"]:
#         colors.append("orange")
# print("colors len is:", len(colors))

# colors = np.array(colors)

# np.savetxt("./data/colors.txt", colors, fmt="%s")

""" Run nSimplices on HMP dataset """

colors = np.loadtxt("./data/hmp_colors.txt", dtype="str")
exec(open("./nsimplices.py").read())
alea.seed(42)

# """ NB normalization + nSimplices + cMDS """ 
# dir="./data/"
# data_path = dir+"hmp_v13lqphylotypeQuantNB_rs_c.csv"
# df_hmp = np.loadtxt(data_path, delimiter=",")
# hmp_dis_sq=squareform(pdist(df_hmp))

# feature_num = 11
# dim_start = 11
# dim_end = feature_num

# print("hmp_dis_sq shape is:", hmp_dis_sq.shape)
# outlier_indices, subspace_dim , corr_pairwise_dis, corr_coord = nsimplices(hmp_dis_sq, feature_num, dim_start, dim_end)
# print("subspace dimension is:", subspace_dim)

# """ (0) Plot cMDS embedding using the pairs of axis from the four most significant axes """
# va, ve, Xe = cMDS(corr_pairwise_dis)
# np.savetxt("./outputs/hmp_NB_nSimplices_cMDS_Xe.txt", Xe, fmt='%f')


""" QE normalization + nSimplices + cMDS """ 
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
Xe = corr_coord


# """ (0) Plot height vs. dimension and ratio vs dimension """

# ### Importance of dimension correction in higher dimension - Fig.4(A) height distribution 

# hcolls = []
# num_point = hmp_dis_sq.shape[0]
# for dim in range(dim_start, dim_end+1):
#     heights = nsimplices_all_heights(num_point, hmp_dis_sq, dim, seed=dim+1)
#     hcolls.append(heights)


# ### Importance of dimension correction in higher dimension - Fig.4(B) dimensionality inference

# # calculate median heights for tested dimension from start_dim to end_dim
# h_meds = []
# for hcoll in hcolls:
#     h_meds.append(np.median(hcoll))

# # calculate the ratio, where h_med_ratios[i] corresponds to h_meds[i-1]/h_meds[i]
# # which is the (median height of dim (i-1+start_dim))/(median height of dim (i+start_dim))
# h_med_ratios = []
# for i in range(1, len(hcolls)):
#     # print("dim", start_dim+i-1, "ratio is:", h_meds[i-1]/h_meds[i], h_meds[i-1], h_meds[i])
#     h_med_ratios.append(h_meds[i-1]/h_meds[i])

# # plot the height scatterplot and the ratios

# fig, ax1 = plt.subplots()
# color = 'tab:red'
# ax1.set_xlabel(r'dimension tested $n$')
# ax1.set_ylabel(r'median of heights', color = color)
# ax1.scatter(list(range(dim_start, dim_end+1)), h_meds, color = color)
# ax1.tick_params(axis ='y', labelcolor = color)
 
# # Adding Twin Axes to plot using dataset_2
# ax2 = ax1.twinx()
 
# color = 'tab:green'
# ax2.set_ylabel(r'heights median ratio: $h_{n-1}/h_n$', color = color)
# ax2.plot(list(range(dim_start+1, dim_end+1)), h_med_ratios, color = color)
# ax2.tick_params(axis ='y', labelcolor = color)
 
# # Show plot
# plt.savefig("./outputs/hmp_QE_nSimplices_cMDS_ratio.png")
# plt.close()

""" (1) Plot cMDS embedding using the pairs of axis from the four most significant axes """
# va, ve, Xe = cMDS(corr_pairwise_dis)
# np.savetxt("./outputs/hmp_QE_nSimplices_cMDS_Xe.txt", Xe, fmt='%f')
# embedding = MDS(n_components=11)
# Xe = embedding.fit_transform(hmp_dis_sq)
np.savetxt("./outputs/hmp_QE_nSimplices_MDS_Xe.txt", Xe, fmt='%f')


num_axes = 4

for first_dim in range(num_axes):
    for second_dim in range(first_dim+1, num_axes):
        plt.figure()
        for i in range(Xe.shape[0]):
            plt.scatter(Xe[i, second_dim], Xe[i, first_dim], s=5, c=colors[i])
        plt.legend(["QuantE+nSimplices"])
        plt.savefig("./outputs/hmp_QE_nSimplices_MDS_"+str(first_dim)+"_"+str(second_dim)+".png")