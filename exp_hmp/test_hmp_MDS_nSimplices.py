import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import random as alea
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import plotly.express as px


""" Prepare colors (only once) """
if not os.path.exists("./data/hmp_colors.txt"):
    print("======== loading colors ========")
    color_df = pd.read_csv("./data/hmp_v13lqphylotypePheno_rs_c.csv", header=0)
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

    np.savetxt("./data/hmp_colors.txt", colors, fmt="%s")


""" Run DeCOr_MDS on HMP dataset """

colors = np.loadtxt("./data/hmp_colors.txt", dtype="str")
exec(open("./DeCOr_MDS.py").read())
alea.seed(42)

""" NB normalization + DeCOr_MDS + cMDS """ 
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
    outlier_indices, subspace_dim , corr_pairwise_dis, corr_coord = DeCOr_MDS(hmp_dis_sq, feature_num, dim_start, dim_end)
    print("subspace dimension is:", subspace_dim)

    # run cMDS to get the corrected coordinates in importance decreasing order
    _, _, Xe = cMDS(corr_pairwise_dis)
    np.savetxt(axes_output_path, Xe, fmt='%f')


""" QE normalization + DeCOr_MDS + cMDS """ 
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
    outlier_indices, subspace_dim , corr_pairwise_dis, corr_coord = DeCOr_MDS(hmp_dis_sq, feature_num, dim_start, dim_end)
    print("subspace dimension is:", subspace_dim)

    """ (1) Plot cMDS embedding using the pairs of axis from the four most significant axes """
    va, ve, Xe = cMDS(corr_pairwise_dis)

    np.savetxt(QE_nSimplices_cMDS_axes_output_path, Xe, fmt='%f')

    """ 
    Determine the number of additional outliers detected in the second round of nSimplices
    """
    normal_indices=[i for i in range(df_hmp.shape[0]) if i not in outlier_indices] # list of normal points 
    hmp_dis_sq_second = hmp_dis_sq[normal_indices, :][:, normal_indices]
    outlier_indices, subspace_dim , corr_pairwise_dis, corr_coord = DeCOr_MDS(hmp_dis_sq_second, feature_num, dim_start, dim_end)
    print("outlier_indices are:", outlier_indices)
    print("number of outliers detected in the second round is:", len(outlier_indices))



figure_output_path = "./outputs/hmp_QE_nSimplices_cMDS_0_1.png"
if not os.path.exists(figure_output_path):
    print("======== plot pairwise 2D plot ========")
    Xe = np.loadtxt(QE_nSimplices_cMDS_axes_output_path)
    num_axes = 3 # show pairwise 2D plot to decompose the 3D plot
    for first_dim in range(num_axes):
        for second_dim in range(first_dim+1, num_axes):
            plt.figure()
            # only plot stool (blue), ears (black), throat (pink) points 
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
            # only plot stool (blue), ears (black), throat (pink) points 
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
        return "MOUTH"
    if color == "orange":
        return "VAGINA"

def site_to_color(site):
    """
    Returns the site to color mapping
    """
    if site == "THROAT":
        return "deeppink"
    if site == "EARS":
        return "black"
    if site == "STOOL":
        return "cornflowerblue"
    if site == "NOSE":
        return "darkgreen"
    if site == "ELBOWS":
        return "red"
    if site == "MOUTH":
        return "gray"
    if site == "VAGINA":
        return "orange"

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
            # only plot stool (blue), ears (black), throat (pink) points 
            for i in range(Xe.shape[0]):
                if colors[i] not in ["cornflowerblue", "black", "orange"]:
                    continue
                plt.scatter(Xe[i, second_dim], Xe[i, first_dim], s=5, c=colors[i])
            plt.legend(["QuantE"])
            plt.savefig("./outputs/hmp_QE_MDS_cMDS_subset_247_"+str(first_dim)+"_"+str(second_dim)+".png")

# combine all plots to one
axes_figure_output_path = "./outputs/hmp_axes_main_127.png"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10*2/3,3))
fig.tight_layout(pad=2, w_pad=2, h_pad=2)
focus_sites = ["STOOL", "THROAT", "VAGINA"] 
focus_colors = [site_to_color(site) for site in focus_sites]
# corresponding colors: 
if not os.path.exists(axes_figure_output_path):
    print("======== plot pairwise 2D plot (subset) in one plot ========")

    QE_MDS_cMDS_Xe = np.loadtxt(QE_MDS_cMDS_axes_output_path)
    num_axes = 2 # show pairwise 2D plot to decompose the 3D plot
    for first_dim in range(num_axes):
        label_list = []
        for second_dim in range(first_dim+1, num_axes):
            # only plot stool (blue), vagina (orange), throat (pink) points 
            # filter by color indices 
            stool_indices = [i for i, e in enumerate(colors) if e == 'cornflowerblue']
            ears_indices = [i for i, e in enumerate(colors) if e == 'orange']
            throat_indices = [i for i, e in enumerate(colors) if e == 'deeppink']


            ax1.scatter(QE_MDS_cMDS_Xe[stool_indices, second_dim], \
                QE_MDS_cMDS_Xe[stool_indices, first_dim], s=5, c='cornflowerblue', label = "STOOL")
            ax1.scatter(QE_MDS_cMDS_Xe[ears_indices, second_dim], \
                QE_MDS_cMDS_Xe[ears_indices, first_dim], s=5, c='orange', label = "VAGINA")
            ax1.scatter(QE_MDS_cMDS_Xe[throat_indices, second_dim], \
                QE_MDS_cMDS_Xe[throat_indices, first_dim], s=5, c='deeppink', label = "THROAT")
            ax1.set_xlabel('axis ' + str(first_dim), size=10)
            ax1.set_ylabel('axis ' + str(second_dim), size=10)
            ax1.tick_params(labelsize=10)
            ax1.legend(fontsize=8)
    ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, size=10, weight='bold')
    ax1.set_title(r'$Q_{E}+MDS$', size=10)   

    QE_nSimplices_cMDS_Xe = np.loadtxt(QE_nSimplices_cMDS_axes_output_path)
    num_axes = 2 # show pairwise 2D plot to decompose the 3D plot
    for first_dim in range(num_axes):
        label_list = []
        for second_dim in range(first_dim+1, num_axes):
            stool_indices = [i for i, e in enumerate(colors) if e == 'cornflowerblue']
            ears_indices = [i for i, e in enumerate(colors) if e == 'orange']
            throat_indices = [i for i, e in enumerate(colors) if e == 'deeppink']


            ax2.scatter(QE_nSimplices_cMDS_Xe[stool_indices, second_dim], \
                QE_nSimplices_cMDS_Xe[stool_indices, first_dim], s=5, c='cornflowerblue', label = "STOOL")
            ax2.scatter(QE_nSimplices_cMDS_Xe[ears_indices, second_dim], \
                QE_nSimplices_cMDS_Xe[ears_indices, first_dim], s=5, c='orange', label = "VAGINA")
            ax2.scatter(QE_nSimplices_cMDS_Xe[throat_indices, second_dim], \
                QE_nSimplices_cMDS_Xe[throat_indices, first_dim], s=5, c='deeppink', label = "THROAT")
            ax2.set_xlabel('axis ' + str(first_dim), size=10)
            ax2.set_ylabel('axis ' + str(second_dim), size=10)
            ax2.tick_params(labelsize=10)
            ax2.legend(fontsize=8)
    ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, size=10, weight='bold')
    ax2.set_title(r'$Q_{E}+nSimplices$', size=10)   

    plt.savefig(axes_figure_output_path)
    plt.close()

# generate 3D plot of the first three axes
nSimplices_dynamic_figure_path = "./outputs/hmp_nSimplices_axes_3D.html"
MDS_dynamic_figure_path = "./outputs/hmp_MDS_axes_3D.html"

if not os.path.exists(nSimplices_dynamic_figure_path) or not os.path.exists(MDS_dynamic_figure_path):
    print("======== generate 3D plot of the first three axes ========")

    QE_nSimplices_cMDS_Xe = np.loadtxt(QE_nSimplices_cMDS_axes_output_path)
    QE_nSimplices_cMDS_Xe_df = pd.DataFrame(QE_nSimplices_cMDS_Xe[:,:3], columns = ["axis_0", "axis_1", "axis_2"])
    QE_nSimplices_cMDS_Xe_df['color'] = colors
    print(QE_nSimplices_cMDS_Xe_df.head())
    print("QE_nSimplices_cMDS_Xe_df shape is:", QE_nSimplices_cMDS_Xe_df.shape)

    QE_nSimplices_cMDS_Xe_df['label'] = \
        QE_nSimplices_cMDS_Xe_df.apply(lambda row: color_to_site(row['color']), axis=1)
    QE_nSimplices_cMDS_Xe_df = \
        QE_nSimplices_cMDS_Xe_df.loc[QE_nSimplices_cMDS_Xe_df['label'].isin(focus_sites)]
    

    color_discrete_map = {'STOOL': 'cornflowerblue', 'THROAT': 'deeppink', 'VAGINA': 'orange'}
    fig = px.scatter_3d(QE_nSimplices_cMDS_Xe_df, x='axis_0', y='axis_1', z='axis_2',
              color='label', color_discrete_map=color_discrete_map)
    fig.update_layout(scene = dict(
                    xaxis_title='axis 0',
                    yaxis_title='axis 1',
                    zaxis_title='axis 2'))
    fig.write_html(nSimplices_dynamic_figure_path)

    QE_MDS_cMDS_Xe = np.loadtxt(QE_MDS_cMDS_axes_output_path)
    QE_MDS_cMDS_Xe_df = pd.DataFrame(QE_MDS_cMDS_Xe[:,:3], columns = ["axis_0", "axis_1", "axis_2"])
    QE_MDS_cMDS_Xe_df['color'] = colors

    QE_MDS_cMDS_Xe_df['label'] = \
        QE_MDS_cMDS_Xe_df.apply(lambda row: color_to_site(row['color']), axis=1)
    QE_MDS_cMDS_Xe_df = \
        QE_MDS_cMDS_Xe_df.loc[QE_MDS_cMDS_Xe_df['label'].isin(focus_sites)]

    fig = px.scatter_3d(QE_MDS_cMDS_Xe_df, x='axis_0', y='axis_1', z='axis_2',
              color='label', color_discrete_map=color_discrete_map)
    fig.update_layout(scene = dict(
                xaxis_title='axis 0',
                yaxis_title='axis 1',
                zaxis_title='axis 2'))
    fig.write_html(MDS_dynamic_figure_path)



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
