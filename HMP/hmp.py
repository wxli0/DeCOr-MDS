import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
import random as alea
from scipy.spatial.distance import pdist, squareform

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

"""Run nSimplices on HMP dataset"""

colors = np.loadtxt("./data/colors.txt", dtype="str")
exec(open("../nsimplices.py").read())
alea.seed(42)


dir="./data/"
data_path = dir+"v13lqphylotypeQuantE_rs.csv"
if len(sys.argv) != 1:
    data_path = sys.argv[1] # TODO: remove argv[1] in the end

df_hmp = np.loadtxt(data_path, delimiter=",")
ori_dis_sq=squareform(pdist(df_hmp))

feature_num = len(df_hmp[0])
dim_start = 1
dim_end = 30

print("ori_dis_sq shape is:", ori_dis_sq.shape)
outlier_indices, subspace_dim , corr_pairwise_dis, corr_coord = nsimplices(ori_dis_sq, feature_num, dim_start, dim_end)
print("subspace dimension is:", subspace_dim)

va, ve, Xe = MDS(corr_pairwise_dis)
np.savetxt("./outputs/Xe_dim"+str(subspace_dim)+".txt", Xe, fmt='%f')

num_eigen = 4

for first_dim in range(num_eigen):
    for second_dim in range(first_dim+1, num_eigen):
        plt.figure()
        for i in range(Xe.shape[0]):
            plt.scatter(Xe[i, second_dim], Xe[i, first_dim], s=5, c=colors[i])
        plt.legend(["QuantE+nSimplices"])
        plt.savefig("./outputs/"+"dim"+str(subspace_dim)+"_"+str(first_dim)+"_"+str(second_dim)+".png")