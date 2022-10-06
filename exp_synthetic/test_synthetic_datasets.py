#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import random as alea
from scipy.linalg import solve,pinv,pinv2
from scipy.spatial.distance import pdist, squareform
import pandas as pd
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

cross_fig_path = "./outputs/synthetic_cross_2D.png"
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
    print("running time is:", T2-T1)
    print("subspace dimension is:", subspace_dim)


    # In[7]:


    """ Plot in 2D using the two largest eigenvalues - Fig.2 """

    normal_indices=[i for i in range(num_point) if i not in outlier_indices] # list of normal points 

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,3))

    # plot original graph
    va, ve, Xe = cMDS(ori_dis_sq)
    ax1.plot(Xe[normal_indices,0],Xe[normal_indices,1],'.')
    ax1.plot(Xe[outlier_indices,0],Xe[outlier_indices,1],'.',color='red')
    ax1.set_title("True data")
    ax1.grid()

    # plot original graphs with outliers added 
    va, ve, Xe = cMDS(out_dis_sq)
    ax2.plot(Xe[normal_indices,0],Xe[normal_indices,1],'.', color='steelblue')
    ax2.plot(Xe[outlier_indices,0],Xe[outlier_indices,1],'.',color='red')
    ax2.set_title("Outliers added")

    # plot correct outliers 
    va, ve, Xe = cMDS(corr_dis_sq)   
    ax3.plot(Xe[normal_indices,0],Xe[normal_indices,1],'.', color='steelblue')
    ax3.plot(Xe[outlier_indices,0],Xe[outlier_indices,1],'.',color='red')
    ax3.set_title("Corrected data")
    ax3.grid()
    plt.savefig(cross_fig_path)
    plt.close()



# 

""" Section 2.1.2: Main subspace of dimension 2 """

dim2_fig_path = "./outputs/synthetic_dim2_2D.png"

if not os.path.exists(dim2_fig_path):
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
    print("running time is:", T2-T1)
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
            ax1.scatter(e[0],e[1],e[2], s=5, color='red')
        else:
            ax1.scatter(e[0],e[1],e[2], s=5, color='steelblue')
    # plt.show()

    ax2 = fig.add_subplot(122, projection='3d')

    # plot the corrected coordinates

    for i in range(num_point):
        e=corr_coord[i]
        if (i in outlier_indices):
            print("outlier corrected:", e)
            ax2.scatter(e[0],e[1],e[2], s=5, color='red')
        else:
            ax2.scatter(e[0],e[1],e[2], s=5, color='steelblue')
    plt.savefig("./outputs/synthetic_dim2_3D.png")
    plt.close()


    print("original coord is:", df_dim2.head(10))
    print("corr_coord is:", pd.DataFrame(corr_coord).head(10))


    # In[14]:


    """ Section 2.1.2: Plot in 2D using the two largest eigenvalues - Fig.3(B) """

    normal_indices=[i for i in range(num_point) if i not in outlier_indices] # list of normal points 

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,3))

    # plot original graph
    va, ve, Xe = cMDS(ori_dis_sq)
    ax1.plot(Xe[normal_indices,0],Xe[normal_indices,1],'.')
    ax1.plot(Xe[outlier_indices,0],Xe[outlier_indices,1],'.',color='red')
    ax1.set_title("True data")
    ax1.grid()

    # plot original graphs with outliers added 
    va, ve, Xe = cMDS(out_dis_sq)
    ax2.plot(Xe[normal_indices,0],Xe[normal_indices,1],'.', color='steelblue')
    ax2.plot(Xe[outlier_indices,0],Xe[outlier_indices,1],'.',color='red')
    ax2.set_title("Outliers added")

    # plot correct outliers 
    va, ve, Xe = cMDS(corr_dis_sq)   
    ax3.plot(Xe[normal_indices,0],Xe[normal_indices,1],'.', color='steelblue')
    ax3.plot(Xe[outlier_indices,0],Xe[outlier_indices,1],'.',color='red')
    ax3.set_title("Corrected data")
    ax3.grid()
    plt.savefig(dim2_fig_path)
    plt.close()


""" Section 2.1.3: Main subspace of higher dimensions """

dim10_fig_path = "./outputs/synthetic_dim10_shepard.png"
if not os.path.exists(dim10_fig_path):
    print(" ====== Running dimension 10 dataset =====")

    # In[15]:

    ### Prepare for section 2.1.3

    ### test data, read in a dataset of main dimension 10
    df_dim10 = pd.read_csv(r'data/synthetic_rdim10.csv',sep=';',header=None)
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
    outlier_indices,subspace_dim,corr_dis_sq,corr_coord = nsimplices(out_dis_sq, df_dim10.shape[1], dim_start=1, dim_end=df_dim10.shape[1])
    T2=time.time()
    print("running time is:", T2-T1)
    print("subspace dimension is:", subspace_dim)


    # In[20]:


    """ Importance of dimension correction in higher dimension - Fig.4(A) height distribution """

    hcolls = []
    start_dim = 2
    end_dim = 15
    for dim in range(start_dim, end_dim+1):
        heights = nsimplices_all_heights(num_point, out_dis_sq, dim, seed=dim+1)
        hcolls.append(heights)

    blues=np.array([[255,255,217,256*0.8],[199,233,180,256*0.8], [65,182,196,256*0.9], [34,94,168,256*0.9], [8,29,88,256*0.8]])/256

    plt.figure()
    # select a few dimensions (i.e. 2,6,8,9,10) for demonstrating the distributions of heights
    plt.hist(hcolls[2-start_dim],label='dim2',color=blues[0])
    plt.hist(hcolls[6-start_dim],label='dim6',color=blues[1])
    plt.hist(hcolls[8-start_dim],label='dim8',color=blues[2])
    plt.hist(hcolls[9-start_dim],label='dim9',color=blues[3])
    plt.hist(hcolls[10-start_dim],label='dim10',color=blues[4])

    plt.xticks(np.arange(0, 60, 5))
    plt.legend()
    plt.savefig("./outputs/synthetic_dim10_height_dist.png")
    plt.close()

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

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel(r'dimension tested $n$')
    ax1.set_ylabel(r'median of heights', color = color)
    ax1.scatter(list(range(start_dim, end_dim+1)), h_meds, color = color)
    ax1.tick_params(axis ='y', labelcolor = color)
    
    # Adding Twin Axes to plot using dataset_2
    ax2 = ax1.twinx()
    
    color = 'tab:green'
    ax2.set_ylabel(r'heights median ratio: $h_{n-1}/h_n$', color = color)
    ax2.plot(list(range(start_dim+1, end_dim+1)), h_med_ratios, color = color)
    ax2.tick_params(axis ='y', labelcolor = color)
    
    # Show plot
    plt.savefig("./outputs/synthetic_dim10_ratio.png")
    plt.close()

    # In[24]:


    """ Importance of dimension correction in higher dimension - Fig.4(C) Shepard Diagram """

    out_dis_flat=out_dis_sq.flatten() # [200*200]
    ori_dis_flat=ori_dis_sq.flatten()
    corr_dis_flat=corr_dis_sq.flatten()

    plt.figure()
    SMALL_SIZE=18
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.scatter(ori_dis_flat,out_dis_flat,color='red',alpha=0.2,s=10)
    plt.scatter(ori_dis_flat,corr_dis_flat,color='mediumblue',alpha=0.05,s=10)

    plt.xlabel(r"true $\delta_{ij}$")
    plt.ylabel(r'$\delta_{ij}$')
    axes = plt.gca()
    axes.set_xlim(5,50)
    axes.set_ylim(2,70)

    plt.savefig(dim10_fig_path)
    plt.close()

"""
Moved to test_synthetic_outlier_detection.py
"""
# # Section 2.1.4: Dimension correction in higher dimensions

# # In[2]:


# # Read in dataset of main dimension 40
# df_dim40 = pd.read_csv(r'data/synthetic_rdim40.csv',sep=';',header=None)
# df_dim40.head()


# # In[28]:


# prop = 0.04
# df_outlier = sim_outliers(df_dim40, prop, 40, 45)
# out_dis=pdist(df_outlier) # pairwise distance in tab (with outliers added)
# out_dis_sq=squareform(out_dis) # squared matrix form of D
# subspace_dim, _ = find_subspace_dim(out_dis_sq, 30, df_outlier.shape[1])

# print("subspace_dim is:", subspace_dim)


# # In[3]:


# outlier_indices_gap = 0.1
# res_outlier_indices_list = np.arange(0, 1, outlier_indices_gap)
# outlier_num = len(res_outlier_indices_list)
# prop_incre = 0.02 # simulate 2% more outliers per iteration
# dim_pred_diff = []
# dim_raw_diff= []
# true_dim = 40
# num_components = 50
# df_prev = df_dim40
# props = np.arange(prop_incre, prop_incre+outlier_num*prop_incre, prop_incre)
# for i in range(len(res_outlier_indices_list)):

#     res_outlier_indices =         range(int(res_outlier_indices_list[i] * df_prev.shape[0]),             int((res_outlier_indices_list[i]+outlier_indices_gap) * df_prev.shape[0]))
#     df_prev = sim_outliers(df_prev, prop_incre, 40, 45,         res_outlier_indices = res_outlier_indices)
#     df_prev.to_csv("outputs/iteration"+str(i)+".csv")
#     out_dis=pdist(df_prev) # pairwise distance in tab (with outliers added)
#     out_dis_sq=squareform(out_dis) # squared matrix form of D
#     subspace_dim, _ = find_subspace_dim(out_dis_sq, 30, df_prev.shape[1])
#     dim_pred_diff.append(subspace_dim - true_dim)
#     dim_raw_diff.append(subspace_dim + int((subspace_dim * prop_incre * i) - true_dim))
#     print("subspace_dim is:", subspace_dim)


# # In[4]:


# plt.figure()
# plt.plot(props, dim_pred_diff, c="red", label = "after correction")
# plt.plot(props, dim_raw_diff, c="blue", label = "before correction")
# plt.xticks(props)
# plt.xlabel(r'frarction of outliers $p$')
# plt.ylabel(r'$\bar{n}-d^{*}$')
# plt.savefig("./outputs/synthetic_prop.png")
# plt.close()


# # In[ ]:




