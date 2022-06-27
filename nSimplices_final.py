#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt
import os
import random as alea
import sys
from scipy.linalg import solve,pinv,pinv2
from scipy.spatial.distance import pdist, squareform
from scipy import optimize
from sklearn.decomposition import PCA
from sklearn import manifold
from scipy import stats
#


""" nSimplex Volume
    Cayley-Menger formula     
"""

def nSimplexVolume(indices,squareDistMat,exactdenominator=False):
    
    n = np.size(indices) - 1
    restrictedD = squareDistMat[:,indices][indices,:]
    CMmat = np.vstack(((n+1)*[1.] , restrictedD))
    CMmat = np.hstack((np.array([[0.]+(n+1)*[1.]]).T , CMmat))
    # NB : missing (-1)**(n+1)  ; but unnecessary here since abs() is taken afterwards
    if exactdenominator:
        denominator = float(2**n *(math.factorial(n))**2)
    else:
        # if calculation of n*Vn/Vn-1 then n*sqrt(denominator_n/denominator_{n-1})
        # simplify to 1/sqrt(2), to take into account in CALLING function.
        denominator = 1.
    VnSquare=np.linalg.det(CMmat**2)
    return np.sqrt(abs(VnSquare/denominator))


'''
Draw B groups of (n-1) points to create B nsimplices containing i, to calculate the heights of the point i
'''
def DrawNSimplices(data,N,B,i,n):

    
    hcollection=[]
    countVzero = 0
    for b in range(B):
        indices  = alea.sample( [x for x in range(N) if x != i] , (n)+1 )
        Vn       = nSimplexVolume( [i]+indices , data, exactdenominator=False)
        Vnm1     = nSimplexVolume( indices, data, exactdenominator=False)
        if Vnm1!=0:
            hcurrent =  Vn / Vnm1 / np.sqrt(2.) #*(n+1)*np.sqrt(2.)
            hcollection.append( hcurrent )
        else:
            hcollection.append(0.0)
    
    B = B - countVzero
    
    return B,hcollection

""" """
""" Determination of the height of each point   
                                                                  
Iteration on each point of the dataset, and the height is the median of heights of the points in B n-simplices

"""
def nSimplwhichh(N,data,n,seed=1,figpath=os.getcwd(), verbose=False):
    alea.seed(seed)
    h=N*[float('NaN')]
    
    
    # Computation of h_i for each i
    for i in range(N):
        
        B=100
        #we draw B groups of (n-1) points, to create n-Simplices and then compute the height median for i
        (B,hcollection)=DrawNSimplices(data,N,B,i,n)
        
        #we here get h[i] the median of heights of the data point i
        h[i] = np.median(hcollection)
        
    return h



""" 
    Classical multidimensional scaling
""" 

def cMDS(D,alreadyCentered=False):
    """
    D: distance / dissimilarity matrix, square and symetric, diagonal 0
    """
    (p,p2)=np.shape(D)
    if p != p2:
        sys.exit("D must be symetric...")
        
    # Double centering
    if not alreadyCentered:
        J=np.eye(p)-np.ones((p,p))/p
        Dcc=-0.5*np.dot(J,np.dot(D**2,J))
    else: 
        # allows robust centering with mediane and mad
        # outside this routine
        Dcc=D
    
    # Eigenvectors
    evals, evecs = np.linalg.eigh(Dcc)
    
    # Sort by eigenvalue in decreasing order, consider all the eigenvectors 
    idx = np.argsort(abs(evals))[::-1]
    evecst = evecs[:,idx]
    evalst= evals[idx] 
    
    # Undelying coordinates 
    idxPos,=np.where(evalst>0) # only  consider eigenvalues > 0
    Xe=np.dot(evecst[:,idxPos],np.diag(evalst[idxPos]**0.5))
    
    return evalst[idxPos], evecst[:,idxPos], Xe




"""
    Outlier correction
    
    Correct the coordinates matrix, by projecting the outliers on the subspace of dimensionality n_bar
"""

def correct_projection(euc_coord, outlier_indices, subspace_dim):
    """
    Correct the outliers index by outlier_indices in euclidean coordinates euc_coord \
        in a subspace of dimension subspace_dim
    Parameters
    ----------
    euc_coord: list[list[float]]
        Euclidean coordinates containing the outliers and normal points 
    outlier_indices: list[int]
        List of indices of outliers in euc_coord
    subspace_dim: int, 
        Dimension of the subspace

    Returns
    -------
    corr_pairwise_dis: list[list[[float]]]
        Correct pairwise distance matrix of the original points in euc_coord
    corr_coord: list[list[float]]
        Corrected coordinates
    """
    feature_num = euc_coord.shape[1] # number of features 
    corr_coord = euc_coord * 1.0
    
    normal_coord = np.delete(euc_coord, outlier_indices, 0) # delete outliers
    print("outliet_indices is:", outlier_indices)
    
    PCA_model = PCA(n_components=subspace_dim)
    _ = PCA_model.fit_transform(normal_coord) # do not need to correct non-outliers 
    PCA_components = PCA_model.components_ # find subspace components formed by Data_pca
    
    normal_mean = np.mean(normal_coord,0) # mean of the normal vectors per feature

    for comp in PCA_components:
        normal_mean = normal_mean - np.dot(normal_mean, comp) * comp 
        # standardize mean by PCA components, TODO: divide by |comp|^2
    print("normal_mean is:", normal_mean)
    
    for idx in outlier_indices:
        outlier = euc_coord[idx]
        print("original coord is:", outlier)
        proj_coord = np.zeros(feature_num)
        for comp in PCA_components:
            proj_coord += np.dot(outlier, comp) * comp
            print("proj_coord is:", proj_coord)
        print("+normal_mean is:", proj_coord + normal_mean)
        corr_coord[idx, :] = proj_coord + normal_mean
        print("corr_coord is:", pd.DataFrame(corr_coord).head(20))

    corr_pairwise_dis = squareform(pdist(corr_coord))
    #Then, the distances data is prepared for MDS.
    
    return corr_pairwise_dis, corr_coord
    

def nSimplices(pairwise_dis, feature_num, dim_start, dim_end, euc_coord=None):
    """
    The nSimplices method
    Parameters
    ----------
    pairwise_dis: int
        The squared matrix form of pairwise distancs
    feature_num: int
        Number of components in MDS
    dim_start: int, default 2
        Lowest dimension to test (inclusive)
    dim_end: int, default 6
        Largest dimension to test (inclusive)
    euc_coord: np 2D array
        Euclidean coordinates of the dataset containing the outliers, default None.\
        If provided, pass euc_coord directly into correct_projection; otherwise, use \
        MDS to transform pairwise_dis 

    Returns
    -------
    outlier_indices: list[int]
        A list of indices of the orthogonal outliers 
    subspace_dim: int
        The relevant dimension of the dataset
    corr_pairwise_dis: list[list[float]]
        The list of corrected pairwise distance 
    corr_coord: list[list[float]]
        The list corrected coordinates
    """
    
    point_num = np.shape(pairwise_dis)[0]
        
    med_height =np.zeros((dim_end-dim_start+1))
    dim_height_map = {}

    
    # Iteration on n: determination of the screeplot nb_outliers function of the dimension tested
    for dim in range(dim_start,dim_end+1):           
        cur_height = nSimplwhichh(point_num, pairwise_dis, dim, seed=dim+1)     
        cur_height = np.array(cur_height)
        med_height[dim-dim_start] = np.median(cur_height)
        dim_height_map[dim] = cur_height
    
    #Determination of the relevant dimension
    dims = np.array(range(dim_start, dim_end+1),dtype=float)
    print("med_height is:", med_height)
    subspace_dim = np.argmax(med_height[0:len(dims)-1]/med_height[1:len(dims)])+dim_start+1
    print("subspace_dim one is:", subspace_dim)
    
    #Detection of outliers in dimension subspace_dim
    subspace_heights = dim_height_map[subspace_dim]
    subspace_height_size = subspace_heights.size
    
    subspace_med=np.median(subspace_heights)
    subspace_std=stats.median_abs_deviation(subspace_heights)
    
    thres = subspace_med + 5 * subspace_std
    all_indices = np.array(range(subspace_height_size))
    outlier_indices = all_indices[subspace_heights > thres]
    print("outlier indices are:", outlier_indices)
    
    
    #Correction of the bias obtained on n_bar 
    outlier_prop = outlier_indices.shape[0]/subspace_height_size
    subspace_dim = subspace_dim - math.floor(subspace_dim * outlier_prop)
    
    # Correction of outliers using MDS, PCA
    corr_coord = None
    if euc_coord is not None: # no need to apply MDS
        print("no MDS")
        corr_pairwise_dis, corr_coord = correct_projection(euc_coord, outlier_indices, subspace_dim)
    else:
        MDS_model = manifold.MDS(n_components=feature_num, max_iter=100000000000,dissimilarity='precomputed')
        euc_coord = MDS_model.fit_transform(pairwise_dis)
        corr_pairwise_dis, corr_coord = correct_projection(euc_coord, outlier_indices, subspace_dim)
    
    return outlier_indices, subspace_dim , corr_pairwise_dis, corr_coord


def sim_outliers(df, prop, col_start, col_end, out_dist = alea.uniform(-100,100)):
    """
    Simulate p (in percentage) outliers in df from column col_start to column col_end

     Parameters
    ----------
    df: list[list[float]]
        The original dataframe 
    p: float
        The outlier fraction
    col_start: int
        The first column index to consider adding outliers (inclusive)
    col_end: int
        The last column index to consider adding outliers (inclusive)
    out_dist: function, default uniform(-100,100)
        The outlier distribution

    Returns
    -------
    df_new: list[list[float]]
        A new dataframe with outliers
    """
    N = df.shape[0]
    df_new = df.copy()
    num_outliers=math.floor(np.ceil(prop*N))
    # random draw of outliers 
    outlier_indices=np.sort(alea.sample(range(N),num_outliers))
    for n in outlier_indices:
        horsplan=out_dist
        # for each row, add outliers to one of columns 10 to 15 (inclusive)
        # columns 10 to 15 are originally simulated with Guassian(2, 0.05)
        i=alea.randint(col_start,col_end)
        df_new.loc[n,i] = horsplan
    return df_new