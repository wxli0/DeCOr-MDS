import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
import random as alea
from scipy.spatial.distance import pdist, squareform

# prepare colors

# color_df = pd.read_csv("./data/v13lqphylotypePheno_rs.csv", header=0)
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

colors = np.loadtxt("./data/colors.txt", dtype="str")




exec(open("../../nsimplices.py").read())
lieudata="./data/"
data_path = lieudata+"v13lqphylotypeQuantE.csv"
if len(sys.argv) != 1:
    data_path = sys.argv[1]

QE=np.loadtxt(data_path, delimiter=",")

DE=pdist(QE) #2255*2255, euclidean distances
#va, ve, Xe = cMDS(squareform(DNBe))
#plt.scatter(Xe[:,0],Xe[:,1],s=10,facecolors='none',edgecolors=Qcolor);plt.title("cMDS(DNBe)") ; plt.show()
#va, ve, Xe = cMDS(squareform(DEe))
#plt.scatter(Xe[:,0],Xe[:,1],s=10,facecolors='none',edgecolors=Qcolor);plt.title("cMDS(DE)") ; plt.show()
#va, ve, Xe = cMDS(squareform(DEo))
#plt.scatter(Xe[:,0],Xe[:,1],s=10,facecolors='none',edgecolors=Qcoloro);plt.title("cMDS(DEo)") ; plt.show()

#DNBecorr,listmini = inequality_correction(DNBe)

ori_dis_sq=squareform(DE)
# np.shape(data) #(2255,2255)
#data=squareform(DR) : see analysis commands in (...) nSimplices_hmp_DE.py

alea.seed(42)

""" Données d'entrée et paramètres """


dim=11 # 5,10,15,20

feature_num = 834
dim_start = 10
dim_end = 11

outlier_indices, subspace_dim , corr_pairwise_dis, corr_coord = nsimplices(ori_dis_sq, feature_num, dim_start, dim_end)
# np.savez("./resu/hhsJhn_"+str(dim),resu[0])
# np.savez("./resu/hsignif_"+str(dim),resu[1])
# np.savez("./resu/outliers_"+str(dim),resu[2])
# np.savez("./resu/cdata_"+str(dim),resu[3])
print("subspace dimension is:", subspace_dim)

va, ve, Xe = MDS(corr_pairwise_dis)

print("Xe shape is:", Xe.shape)

for i in range(Xe.shape[0]):
    # print(Xe[i, 0])
    # print(Xe[i, 1])
    plt.scatter(Xe[i, 0], Xe[i, 1], c=colors[i])
# plt.plot(Xe[:,0],Xe[:,1],'.', color = colors[:100])
plt.legend(["QuantE+nSimplices"])

output_file = "dim" + str(dim)
if len(sys.argv) != 0:
    output_file += "_subset"
plt.savefig("./resu/"+output_file+".png")

