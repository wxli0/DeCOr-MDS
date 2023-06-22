#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import os
import random as alea
from scipy.linalg import solve,pinv,pinv2
from scipy.spatial.distance import pdist, squareform
import time
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import time


# In[ ]:


# import nSimplices 
# get_ipython().run_line_magic('matplotlib', 'widget')
exec(compile(open(r"nsimplices.py", encoding="utf8").read(), "nsimplices.py", 'exec'))

# set matplotlib default savefig directory
plt.rcParams["savefig.directory"] = os.getcwd() # To save figures to directory
                                                #   defined above


""" Section 2.1.1: Cross dataset """

cross_fig_path = "./outputs/synthetic_cross_2D.pdf"
if not os.path.exists(cross_fig_path):
    print(" ====== Running cross dataset =====")
    # In[3]:


    ### Prepare the dataset
    df_cross = []
    # for x in range(-6,7,1):
    #     df_cross.append([x, 0, 0])
    # for y in range(-6,7,1):
    #     df_cross.append([0, y, 0])
    df_cross =     [[-6, 0, 0], [-5, 0, 0], [-4, 0, 0], [-3, 0, 0], 
        [-2, 0, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0], 
        [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0], 
        [6, 0, 0], [0, -6, 0], [0, -5, 0], [0, -4, 0], 
        [0, -3, 0], [0, -2, 0], [0, -1, 0],  
        [0, 1, 0], [0, 2, 0], [0, 3, 0], 
        [0, 4, 0], [0, 5, 0], [0, 6, 0]]
    df_cross = pd.DataFrame(df_cross)
    num_point = df_cross.shape[0]

    ori_dis=pdist(df_cross.copy()) # compute pairwise distance in data
    ori_dis_sq=squareform(ori_dis) # true pairwise distance in squared form


    # In[4]:


    df_cross =     [[-6, 0, -30/2], [-5, 0, 0], [-4, 0, 0], [-3, 0, 0], 
        [-2, 0, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0], 
        [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0], 
        [6, 0, 20/2], [0, -6, 0], [0, -5, 0], [0, -4, 45/2], 
        [0, -3, 0], [0, -2, 0], [0, -1, 0],  
        [0, 1, 0], [0, 2, 0], [0, 3, 0], 
        [0, 4, 0], [0, 5, 0], [0, 6, 0]]
    df_cross = pd.DataFrame(df_cross)


    # In[5]:


    ### Preparing pairwise distances

    """ euclidean distances """
    out_dis = pdist(df_cross)
    out_dis_sq = squareform(out_dis)


    # In[6]:



    ### Run nSimplices method
    feature_num = df_cross.shape[1]
    dim_start = 1
    dim_end = df_cross.shape[1]
    T1=time.time()
    # outlier_indices, subspace_dim, corr_dis_sq, corr_coord = nSimplices(out_dis_sq, feature_num, dim_start, dim_end, euc_coord=np.array(df_cross.copy()))

    outlier_indices, subspace_dim, corr_dis_sq, corr_coord = nsimplices(out_dis_sq, feature_num, dim_start, dim_end, euc_coord=np.array(df_cross.copy()))

    T2=time.time()
    print("runtime is:", T2-T1)
    print("subspace dimension is:", subspace_dim)


    # In[7]:


    """ Plot in 2D using the two largest eigenvalues - Fig.2 """

    normal_indices=[i for i in range(num_point) if i not in outlier_indices] # list of normal points 

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9,3))

    # plot original graph
    va, ve, Xe = cMDS(ori_dis_sq)
    ax1.plot(Xe[normal_indices,0],Xe[normal_indices,1],'.', color='black', label="normal")
    ax1.plot(Xe[outlier_indices,0],Xe[outlier_indices,1],'.',color='red', label="selected")
    ax1.set_title("True data")
    ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, 
            size=15, weight='bold')
    ax1.set_xlabel('axis 0', fontsize=9)
    ax1.set_ylabel('axis 1', fontsize=9)

    ax1.legend(fontsize=9)
    ax1.grid()

    # plot original graphs with outliers added 
    va, ve, Xe = cMDS(out_dis_sq)
    ax2.plot(Xe[normal_indices,0],Xe[normal_indices,1],'.', color='black', label="normal")
    ax2.plot(Xe[outlier_indices,0],Xe[outlier_indices,1],'.',color='red', label="outlier")
    ax2.legend(fontsize=9)
    ax2.grid()
    ax2.set_title("Outliers added")
    ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, 
            size=15, weight='bold')
    ax2.set_xlabel('axis 0', fontsize=9)
    ax2.set_ylabel('axis 1', fontsize=9)

    # plot correct outliers 
    va, ve, Xe = cMDS(corr_dis_sq)   
    ax3.plot(Xe[normal_indices,0],Xe[normal_indices,1],'.', color='black', label="normal")
    ax3.plot(Xe[outlier_indices,0],Xe[outlier_indices,1],'.',color='red', label="outlier")
    ax3.legend(fontsize=9)
    ax3.set_title("Corrected data")
    ax3.text(-0.1, 1.05, 'C', transform=ax3.transAxes, 
            size=14, weight='bold')
    ax3.set_xlabel('axis 0', fontsize=9)
    ax3.set_ylabel('axis 1', fontsize=9)
    ax3.grid()
    plt.tight_layout()
    plt.savefig(cross_fig_path)
    plt.close()


    """ Section 1.1.2: Plot in 3D using the first three dimensions for cross data """

    before_correction_dynamic_figure_path = "./outputs/synthetic_cross_before_correction_dynamic.html"
    after_correction_dynamic_figure_path = "./outputs/synthetic_cross_after_correction_dynamic.html"

    ori_coord=np.array(df_cross)

    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121, projection='3d')

    # plot the original coordinates

    for i in range(num_point):
        e=ori_coord[i]
        if (i in outlier_indices):
            print("outlier:", e)
            ax1.scatter(e[0],e[1],e[2], s=5, color='red', label="outlier")
        else:
            ax1.scatter(e[0],e[1],e[2], s=5, color='black', label="normal")
    ax1.set_title("Outliers added")
    ax1.text2D(0, 1, "A", transform=ax1.transAxes, size=15, weight='bold')

    # ax1.text(-10, 27, 'A', transform=ax1.transAxes, 
    #     size=15, weight='bold')

    def legend_without_duplicate_labels(ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))

    legend_without_duplicate_labels(ax1)

    outlier_array = []
    for i in range(corr_coord.shape[0]):
        if i in outlier_indices:
            outlier_array.append("outlier")
        else:
            outlier_array.append("normal")
    color_discrete_map = {'outlier': 'red', 'normal': 'black'}
    ori_coords_df = pd.DataFrame(ori_coord[:,:3], columns = ["axis_0", "axis_1", "axis_2"])
    ori_coords_df['label'] = outlier_array
    fig_dy1 = px.scatter_3d(ori_coords_df, x='axis_0', y='axis_1', z='axis_2',
            color='label', color_discrete_map=color_discrete_map)
    fig_dy1.update_layout(scene = dict(
                xaxis_title='axis 0',
                yaxis_title='axis 1',
                zaxis_title='axis 2'))
    fig_dy1.write_html(before_correction_dynamic_figure_path)


    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlim3d(-6, 6)
    ax2.set_ylim3d(-10, 5)
    ax2.set_zlim3d(-6, 6)
    ax2.set_title("Corrected data")
    ax2.text2D(0, 1, "B", transform=ax2.transAxes, size=15, weight='bold')

    # ax2.text(2, -10, 'B', transform=ax2.transAxes, 
    #     size=15, weight='bold')

    # plot the corrected coordinates

    for i in range(num_point):
        e=corr_coord[i]
        if (i in outlier_indices):
            print("outlier corrected:", e)
            ax2.scatter(e[0],e[1],e[2], s=5, color='red', label="outlier")
        else:
            ax2.scatter(e[0],e[1],e[2], s=5, color='black', label="normal")
    
    legend_without_duplicate_labels(ax2)
    plt.savefig("./outputs/synthetic_cross_3D.png")
    plt.close()


    print("original coord is:", df_cross.head(10))
    print("corr_coord is:", pd.DataFrame(corr_coord).head(10))

    coords_df = pd.DataFrame(corr_coord[:,:3], columns = ["axis_0", "axis_1", "axis_2"])
    coords_df['label'] = outlier_array
    fig_dy2 = px.scatter_3d(coords_df, x='axis_0', y='axis_1', z='axis_2',
            color='label', color_discrete_map=color_discrete_map)
    fig_dy2.update_layout(scene = dict(
                xaxis_title='axis 0',
                yaxis_title='axis 1',
                zaxis_title='axis 2'))
    fig_dy2.write_html(after_correction_dynamic_figure_path)


# 

""" Section 2.1.2: Main subspace of dimension 2 """

dim2_fig_path = "./outputs/synthetic_dim2_2D.png"
after_correction_dynamic_figure_path = "./outputs/synthetic_dim2_after_correction_dynamic.html"
before_correction_dynamic_figure_path = "./outputs/synthetic_dim2_before_correction_dynamic.html"
if not os.path.exists(dim2_fig_path) or  not os.path.exists(before_correction_dynamic_figure_path)  \
    or  not os.path.exists(after_correction_dynamic_figure_path) :
    print(" ====== Running dimension 2 dataset =====")

    # In[8]:

    ### test data, read in a dataset of main dimension 2
    df_dim2 = pd.read_csv(r'data/synthetic_rdim2.csv',sep=';',header=None)
    df_dim2.head()


    # In[9]:


    ### Processing datasets and computing pairwise distances
    num_point = df_dim2.shape[0]
    ori_dis = pdist(df_dim2.copy()) # compute pairwise distance in data
    ori_dis_sq = squareform(ori_dis) # true pairwise distance in squared form


    # In[10]:


    ### Add outliers
    prop = 0.05
    num_outliers = int(np.ceil(prop * num_point))

    # random draw of outliers 
    indices = np.sort(alea.sample(range(num_point),num_outliers))
    for n in indices:
        outlier = alea.uniform(-30,30)
        # only add outliers to the third dimension for the visualization purpose
        print(outlier)
        df_dim2.loc[n,2] = outlier 
        
    df_dim2.head(20)


    # In[11]:


    ### Preparing pairwise distances

    """ euclidean distances """
    out_dis=pdist(df_dim2) # pairwise distance in tab (with outliers added)
    out_dis_sq=squareform(out_dis) # squared matrix form of D


    # In[12]:


    """ Run n-Simplices method """
    T1=time.time()
    outlier_indices,rdim,corr_dis_sq,corr_coord = nsimplices(out_dis_sq, df_dim2.shape[1], dim_start = 1, dim_end = 7)
    T2=time.time()
    print("runtime is:", T2-T1)
    print("subspace dimension is:", rdim)


    # In[13]:


    """ Section 2.1.2: Plot in 3D using the first three dimensions - Fig.3(A) """

    ori_coord=np.array(df_dim2)

    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121, projection='3d')

    # plot the original coordinates

    for i in range(num_point):
        e=ori_coord[i]
        if (i in outlier_indices):
            print("outlier:", e)
            ax1.scatter(e[0],e[1],e[2], s=5, color='red', label="outlier")
        else:
            ax1.scatter(e[0],e[1],e[2], s=5, color='black', label="normal")
    ax1.set_title("Outliers added")
    ax1.text2D(0, 1, "A", transform=ax1.transAxes, size=15, weight='bold')

    # ax1.text(-10, 27, 'A', transform=ax1.transAxes, 
    #     size=15, weight='bold')

    def legend_without_duplicate_labels(ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))

    legend_without_duplicate_labels(ax1)

    outlier_array = []
    for i in range(corr_coord.shape[0]):
        if i in outlier_indices:
            outlier_array.append("outlier")
        else:
            outlier_array.append("normal")
    color_discrete_map = {'outlier': 'red', 'normal': 'black'}
    ori_coords_df = pd.DataFrame(ori_coord[:,:3], columns = ["axis_0", "axis_1", "axis_2"])
    ori_coords_df['label'] = outlier_array
    fig_dy1 = px.scatter_3d(ori_coords_df, x='axis_0', y='axis_1', z='axis_2',
            color='label', color_discrete_map=color_discrete_map)
    fig_dy1.update_layout(scene = dict(
                xaxis_title='axis 0',
                yaxis_title='axis 1',
                zaxis_title='axis 2'))
    fig_dy1.write_html(before_correction_dynamic_figure_path)


    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlim3d(-6, 6)
    ax2.set_ylim3d(-10, 5)
    ax2.set_zlim3d(-6, 6)
    ax2.set_title("Corrected data")
    ax2.text2D(0, 1, "B", transform=ax2.transAxes, size=15, weight='bold')
    

    # ax2.text(2, -10, 'B', transform=ax2.transAxes, 
    #     size=15, weight='bold')

    # plot the corrected coordinates

    for i in range(num_point):
        e=corr_coord[i]
        if (i in outlier_indices):
            print("outlier corrected:", e)
            ax2.scatter(e[0],e[1],e[2], s=5, color='red', label="outlier")
        else:
            ax2.scatter(e[0],e[1],e[2], s=5, color='black', label="normal")
    legend_without_duplicate_labels(ax2)
    plt.savefig("./outputs/synthetic_dim2_3D.png")
    plt.close()


    print("original coord is:", df_dim2.head(10))
    print("corr_coord is:", pd.DataFrame(corr_coord).head(10))

    coords_df = pd.DataFrame(corr_coord[:,:3], columns = ["axis_0", "axis_1", "axis_2"])
    coords_df['label'] = outlier_array
    fig_dy2 = px.scatter_3d(coords_df, x='axis_0', y='axis_1', z='axis_2',
            color='label', color_discrete_map=color_discrete_map)
    fig_dy2.update_layout(scene = dict(
                xaxis_title='axis 0',
                yaxis_title='axis 1',
                zaxis_title='axis 2'))
    fig_dy2.write_html(after_correction_dynamic_figure_path)



    # In[14]:


    """ Section 2.1.2: Plot in 2D using the two largest eigenvalues - Fig.3(B) """

    normal_indices=[i for i in range(num_point) if i not in outlier_indices] # list of normal points 

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,3))

    # plot original graph
    va, ve, Xe = cMDS(ori_dis_sq)
    ax1.plot(Xe[normal_indices,0],Xe[normal_indices,1],'.', color='black', label="normal")
    ax1.plot(Xe[outlier_indices,0],Xe[outlier_indices,1],'.',color='red',label="selected")
    ax1.set_title("True data")
    ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, 
        size=15, weight='bold')
    ax1.legend()
    ax1.grid()

    # plot original graphs with outliers added 
    va, ve, Xe = cMDS(out_dis_sq)
    ax2.plot(Xe[normal_indices,0],Xe[normal_indices,1],'.', color='black', label="normal")
    ax2.plot(Xe[outlier_indices,0],Xe[outlier_indices,1],'.',color='red', label="outlier")
    ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, 
        size=15, weight='bold')
    ax2.grid()
    ax2.legend()
    ax2.set_title("Outliers added")

    # plot corrected outliers 
    va, ve, Xe = cMDS(corr_dis_sq)   
    ax3.plot(Xe[normal_indices,0],Xe[normal_indices,1],'.', color='black', label="normal")
    ax3.plot(Xe[outlier_indices,0],Xe[outlier_indices,1],'.',color='red', label="outlier")
    ax3.set_title("Corrected data")
    ax2.text(-0.1, 1.05, 'C', transform=ax3.transAxes, 
        size=15, weight='bold')
    ax3.grid()
    ax3.legend()
    plt.savefig(dim2_fig_path)
    plt.close()


""" Section 2.1.3: Main subspace of higher dimensions """

dim10_fig_path = "./outputs/synthetic_dim10_1000.pdf"
if not os.path.exists(dim10_fig_path):
    print(" ====== Running dimension 10 dataset =====")

    # In[15]:

    ### Prepare for section 2.1.3

    ### test data, read in a dataset of main dimension 10
    df_dim10 = pd.read_csv(r'data/synthetic_rdim10_2500.csv',sep=';',header=None)
    df_dim10.head()


    # In[16]:


    ori_dis = pdist(df_dim10.copy())
    ori_dis_sq = squareform(ori_dis)
    num_point =df_dim10.shape[0]


    # In[17]:


    """ Add outliers """

    prop=0.05
    num_outliers=int(np.ceil(prop * num_point))
    # random draw of outliers 
    outlier_indices=np.sort(alea.sample(range(num_point),num_outliers))
    print("outlier_indices are:", outlier_indices)
    for n in outlier_indices:
        outlier=alea.uniform(-100,100)
        
        # for each row, add outliers to one of columns 10 to 15 (inclusive)
        # columns 10 to 15 are originally simulated with Guassian(2, 0.05)
        i=alea.randint(10,12) 
        df_dim10.loc[n,i] = outlier
    df_dim10.head(50)
    #  [ 10  19  92 106 126 145 158]


    # In[18]:


    ### Preparing pairwise distances for the dataset with outliers

    """ euclidean distances """
    out_dis=pdist(df_dim10)
    out_dis_sq=squareform(out_dis)


    # In[19]:


    ### Run nSimplices method
    T1=time.time()
    outlier_indices,subspace_dim,corr_dis_sq,corr_coord = nsimplices(out_dis_sq, df_dim10.shape[1], dim_start=1, dim_end=df_dim10.shape[1], num_groups=100)
    T2=time.time()
    print("runtime is:", T2-T1)
    print("subspace dimension is:", subspace_dim)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10*2.5,3*2.5))
    fig.tight_layout(pad=8, w_pad=10, h_pad=1.0)

    # In[20]:


    """ Importance of dimension correction in higher dimension - Fig.4(A) height distribution """

    hcolls = []
    start_dim = 2
    end_dim = 15
    for dim in range(start_dim, end_dim+1):
        heights = nsimplices_all_heights(num_point, out_dis_sq, dim, seed=dim+1)
        hcolls.append(heights)

    blues=np.array([[255,255,217,256*0.8],[199,233,180,256*0.8], [65,182,196,256*0.9], [34,94,168,256*0.9], [8,29,88,256*0.8]])/256

    # plt.figure()
    # select a few dimensions (i.e. 2,6,8,9,10) for demonstrating the distributions of heights
    max_x = 30
    bin_width=2
    ax1.hist(hcolls[2-start_dim],label='dim2',color=blues[0], \
        bins=np.arange(min(hcolls[2-start_dim]), 30 + bin_width, bin_width))
    ax1.hist(hcolls[6-start_dim],label='dim6',color=blues[1], \
        bins=np.arange(min(hcolls[6-start_dim]), 30 + bin_width, bin_width))
    ax1.hist(hcolls[8-start_dim],label='dim8',color=blues[2], \
        bins=np.arange(min(hcolls[8-start_dim]), 30 + bin_width, bin_width))
    ax1.hist(hcolls[9-start_dim],label='dim9',color=blues[3], \
        bins=np.arange(min(hcolls[9-start_dim]), 30 + bin_width, bin_width))
    ax1.hist(hcolls[10-start_dim],label='dim10',color=blues[4], \
        bins=np.arange(min(hcolls[10-start_dim]), 30 + bin_width, bin_width))

    ax1.set_xticks(np.arange(0, 35, 5))
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.set_ylabel("frequency", fontsize=25)
    ax1.set_xlabel(r'median of the heights $h^{n}_i$', fontsize=25)
    ax1.set_title("Distribution of medians of heights", fontsize=28)
    ax1.text(-0.1, 1.15, 'A', transform=ax1.transAxes, 
        size=37, weight='bold')
    ax1.legend(fontsize=25)


    # In[26]:


    """ Importance of dimension correction in higher dimension - Fig.4(B) dimensionality inference """

    # calculate median heights for tested dimension from start_dim to end_dim
    h_meds = []
    for hcoll in hcolls:
        h_meds.append(np.median(hcoll))

    # calculate the ratio, where h_med_ratios[i] corresponds to h_meds[i-1]/h_meds[i]
    # which is the (median height of dim (i-1+start_dim))/(median height of dim (i+start_dim))
    h_med_ratios = []
    for i in range(1, len(hcolls)):
        # print("dim", start_dim+i-1, "ratio is:", h_meds[i-1]/h_meds[i], h_meds[i-1], h_meds[i])
        h_med_ratios.append(h_meds[i-1]/h_meds[i])

    # plot the height scatterplot and the ratios

    # fig, sub_ax1 = plt.subplots()
    color = 'red'
    ax2.set_xlabel(r'dimension tested $n$', fontsize=25)
    ax2.set_ylabel(r'median of heights', color = color, fontsize=25)
    ax2.set_xticks(np.arange(2, 16, 1))
    ax2.scatter(list(range(start_dim, end_dim+1)), h_meds, color = color, s=10)
    ax2.plot(list(range(start_dim, end_dim+1)), h_meds, color=color)

    ax2.tick_params(axis ='y', labelcolor = color)
    ax2.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.text(-0.1, 1.15, 'B', transform=ax2.transAxes, 
        size=37, weight='bold')
    ax2.set_title("Dimensionality inference", fontsize=28)
    
    # Adding Twin Axes to plot using dataset_2
    sub_ax2 = ax2.twinx()
    
    color = 'black'
    sub_ax2.set_ylabel(r'heights median ratio: $h_{n-1}/h_n$', color = color, fontsize=25)
    sub_ax2.plot(list(range(start_dim+1, end_dim+1)), h_med_ratios, color = color)
    sub_ax2.tick_params(axis ='y', labelcolor = color)
    sub_ax2.tick_params(axis='x', labelsize=20)
    sub_ax2.tick_params(axis='y', labelsize=20)

    # In[24]:


    """ Importance of dimension correction in higher dimension - Fig.4(C) Shepard Diagram """

    out_dis_flat=out_dis_sq.flatten() # [200*200]
    ori_dis_flat=ori_dis_sq.flatten()
    corr_dis_flat=corr_dis_sq.flatten()

    # plt.figure()
    # ax3.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # ax3.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels

    ax3.scatter(ori_dis_flat,out_dis_flat,color='red',alpha=0.8,s=6, label="MDS")
    ax3.scatter(ori_dis_flat,corr_dis_flat,color='black',alpha=0.2,s=6, label="nSimplices")
    ax3.tick_params(axis='x', labelsize=20)
    ax3.tick_params(axis='y', labelsize=20)

    ax3.set_xlabel("original pairwise distance", fontsize=25)
    ax3.set_ylabel("embedded pairwise distance", fontsize=25)
    ax3.set_xlim(5,50)
    ax3.set_ylim(2,70)
    ax3.set_title("Shepard diagram", fontsize=28)
    ax3.text(-0.1, 1.15, 'C', transform=ax3.transAxes, 
        size=37, weight='bold')
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")

    ax3.legend(fontsize=25)

    plt.savefig(dim10_fig_path)
    plt.close()
