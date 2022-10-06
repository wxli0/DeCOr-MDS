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
    color_df = pd.read_csv("./data/v13lqphylotypePheno_rs_c.csv", header=0)
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
QE_nSimplices_cMDS_axes_output_path = "./outputs/hmp_QE_nSimplices_cMDS_axes.txt"
if not os.path.exists(QE_nSimplices_cMDS_axes_output_path):
    print("======== QE normalization + nSimplices + cMDS ========")
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

    np.savetxt(QE_nSimplices_cMDS_axes_output_path, Xe, fmt='%f')


figure_output_path = "./outputs/hmp_QE_nSimplices_cMDS_0_1.png"
if not os.path.exists(figure_output_path):
    print("======== plot pairwise 2D plot ========")
    Xe = np.loadtxt(QE_nSimplices_cMDS_axes_output_path)
    num_axes = 3 # show pairwise 2D plot to decompose the 3D plot
    for first_dim in range(num_axes):
        for second_dim in range(first_dim+1, num_axes):
            plt.figure()
            # only plot stool (blue), ears (black), throad (pink) points 
            for i in range(Xe.shape[0]):
                plt.scatter(Xe[i, second_dim], Xe[i, first_dim], s=5, c=colors[i])
            plt.legend(["QuantE+nSimplices"])
            plt.savefig("./outputs/hmp_QE_nSimplices_cMDS_"+str(first_dim)+"_"+str(second_dim)+".png")


figure_output_path = "./outputs/hmp_QE_nSimplices_cMDS_subset_247_0_1.png"
if not os.path.exists(figure_output_path):
    print("======== plot pairwise 2D plot (subset) ========")
    Xe = np.loadtxt(QE_nSimplices_cMDS_axes_output_path)
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


def color_to_site(color):
    """
    Returns the color to site mapping
    """
    if color == "deeppink":
        return "THROAT"
    if color == "black":
        return "EARS"
    if color == "cornflowerblue":
        return "STOOL"
    if color == "darkgreen":
        return "NOSE"
    if color == "red":
        return "ELBOWS"
    if color == "gray":
        return ["MOUTH"]
    if color == "orange":
        return "VAGINA"


""" QE normalization + MDS + cMDS """ 
QE_MDS_cMDS_axes_output_path = "./outputs/hmp_QE_MDS_cMDS_axes.txt"
if not os.path.exists(QE_MDS_cMDS_axes_output_path):
    print("======== QE normalization + MDS + cMDS ========")
    dir="./data/"
    data_path = dir+"hmp_v13lqphylotypeQuantE_rs_c.csv"

    df_hmp = np.loadtxt(data_path, delimiter=",")
    hmp_dis_sq=squareform(pdist(df_hmp))

    # Plot cMDS embedding using the pairs of axis from the four most significant axes 
    enforce_dim = 11 # enforcing dimension 11 to be consistent with QE+nsimplices+cMDS
    embedding = MDS(n_components=enforce_dim) 
    corr_coord = embedding.fit_transform(hmp_dis_sq)
    corr_dis_sq=squareform(pdist(corr_coord))
    _, _, Xe = cMDS(corr_dis_sq)

    np.savetxt(QE_MDS_cMDS_axes_output_path, Xe, fmt='%f')

figure_output_path = "./outputs/hmp_QE_MDS_cMDS_0_1.png"
if not os.path.exists(figure_output_path):
    print("======== plot pairwise 2D plot ========")
    Xe = np.loadtxt(QE_MDS_cMDS_axes_output_path)
    num_axes = 3
    for first_dim in range(num_axes):
        for second_dim in range(first_dim+1, num_axes):
            plt.figure()
            for i in range(Xe.shape[0]):
                plt.scatter(Xe[i, second_dim], Xe[i, first_dim], s=5, c=colors[i])
            plt.legend(["QuantE"])
            plt.savefig("./outputs/hmp_QE_MDS_cMDS_"+str(first_dim)+"_"+str(second_dim)+".png")

figure_output_path = "./outputs/hmp_QE_MDS_cMDS_subset_247_0_1.png"
if not os.path.exists(figure_output_path):
    print("======== plot pairwise 2D plot (subset) ========")
    Xe = np.loadtxt(QE_MDS_cMDS_axes_output_path)
    num_axes = 3
    for first_dim in range(num_axes):
        for second_dim in range(first_dim+1, num_axes):
            plt.figure()
            # only plot stool (blue), ears (black), throad (pink) points 
            for i in range(Xe.shape[0]):
                if colors[i] not in ["cornflowerblue", "black", "orange"]:
                    continue
                plt.scatter(Xe[i, second_dim], Xe[i, first_dim], s=5, c=colors[i])
            plt.legend(["QuantE"])
            plt.savefig("./outputs/hmp_QE_MDS_cMDS_subset_247_"+str(first_dim)+"_"+str(second_dim)+".png")

# combine all plots to one
axes_figure_output_path = "./outputs/hmp_axes_main.png"
fig, axes = plt.subplots(3, 2, figsize=(8,10))
if not os.path.exists(axes_figure_output_path):
    print("======== plot pairwise 2D plot (subset) in one plot ========")
    QE_nSimplices_cMDS_Xe = np.loadtxt(QE_nSimplices_cMDS_axes_output_path)
    row = 0
    col = 1
    num_axes = 3 # show pairwise 2D plot to decompose the 3D plot
    for first_dim in range(num_axes):
        label_list = []
        for second_dim in range(first_dim+1, num_axes):
            # only plot stool (blue), ears (black), throad (pink) points 
            for i in range(QE_nSimplices_cMDS_Xe.shape[0]):
                if colors[i] not in ["cornflowerblue", "black", "orange"]:
                    continue
                axes[row][col].scatter(QE_nSimplices_cMDS_Xe[i, second_dim], \
                    QE_nSimplices_cMDS_Xe[i, first_dim], s=5, c=colors[i], label = color_to_site(colors[i]))
            axes[row][col].set_xlabel('axes ' + str(first_dim))
            axes[row][col].set_ylabel('axes ' + str(second_dim))

            handles, labels = axes[row][col].get_legend_handles_labels()
            handle_list, label_list = [], []
            for handle, label in zip(handles, labels):
                if label not in label_list:
                    handle_list.append(handle)
                    label_list.append(label)
            axes[row][col].legend(handle_list, label_list, prop={'size': 6})
            row += 1
    axes[0][1].set_title("QE+nSimplices")


    QE_MDS_cMDS_Xe = np.loadtxt(QE_MDS_cMDS_axes_output_path)
    row = 0
    col = 0
    num_axes = 3 # show pairwise 2D plot to decompose the 3D plot
    for first_dim in range(num_axes):
        label_list = []
        for second_dim in range(first_dim+1, num_axes):
            # only plot stool (blue), ears (black), throad (pink) points 
            for i in range(QE_MDS_cMDS_Xe.shape[0]):
                if colors[i] not in ["cornflowerblue", "black", "orange"]:
                    continue
                axes[row][col].scatter(QE_MDS_cMDS_Xe[i, second_dim], \
                    QE_MDS_cMDS_Xe[i, first_dim], s=5, c=colors[i], label = color_to_site(colors[i]))
            axes[row][col].set_xlabel('axes ' + str(first_dim))
            axes[row][col].set_ylabel('axes ' + str(second_dim))

            handles, labels = axes[row][col].get_legend_handles_labels()
            handle_list, label_list = [], []
            for handle, label in zip(handles, labels):
                if label not in label_list:
                    handle_list.append(handle)
                    label_list.append(label)
            axes[row][col].legend(handle_list, label_list, prop={'size': 6})
            row += 1

    axes[0][0].set_title("QE+MDS")      

    plt.savefig(axes_figure_output_path)
    plt.close()


""" NB normalization + MDS + cMDS """ 
axes_output_path = "./outputs/hmp_NB_MDS_cMDS_axes.txt"
if not os.path.exists("./outputs/hmp_NB_MDS_cMDS_axes.txt"):
    print("======== NB normalization + MDS + cMDS ========")
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
    np.savetxt(axes_output_path, Xe, fmt='%f')
