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
import pandas as pd


def simplex_volume(indices,dis_sq,use_deno=False):
    """ 
    Calculate simplex volumn using Cayley-Menger formula formed by specific indices  

    Parameters
    ----------
    indices: list[int]
        The indices to form the simplex
    dis_sq: list[float]
        The squared-form pairwise distances
    exact_dono: bool
        Use denominator or not in the final volumn, default false

    Returns
    -------
    float
        The simplex volumn
    """

    n = np.size(indices) - 1
    tar_dis_sq = dis_sq[:,indices][indices,:]
    CM_mat = np.vstack(((n+1)*[1.] , tar_dis_sq))
    CM_mat = np.hstack((np.array([[0.]+(n+1)*[1.]]).T , CM_mat))
    # NB : missing (-1)**(n+1)  ; but unnecessary here since abs() is taken afterwards
    if use_deno:
        deno = float(2**n *(math.factorial(n))**2)
    else:
        # if calculation of n*Vn/Vn-1 then n*sqrt(denominator_n/denominator_{n-1})
        # simplify to 1/sqrt(2), to take into account in CALLING function.
        deno = 1.
    VnSquare=np.linalg.det(CM_mat**2)
    return np.sqrt(abs(VnSquare/deno))



def nsimplices_heights(dis_sq, num_total_point, num_group, point_index, num_simplex_point):
    '''
    From a set of num_total_point points with pairwise distances (dis_sq), \
        draw num_group groups of (num_simplex-1) \
        points to create B nsimplices containing point_index point, \
        to calculate the heights of the point point_index

    Parameters
    ----------
    dis_sq: list[float]
        The squared-formed pairwise distances
    num_total_point: int
        The total number of points in dis_sq
    num_group: int
        The number of groups to draw
    point_index: int
        The target point index
    num_simplex_point: int
        The number of points for a simplex

    Returns
    -------
    heights: list[float]
        The heights for the drawn simplices
    '''
    
    heights=[]
    for _ in range(num_group):
        indices = \
            alea.sample([x for x in range(num_total_point) if x != point_index], \
                (num_simplex_point)+1 )
        Vn = simplex_volume([point_index]+indices , dis_sq)
        Vnm1 = simplex_volume(indices, dis_sq)
        if Vnm1!=0:
            hcurrent =  Vn / Vnm1 / np.sqrt(2.) #*(n+1)*np.sqrt(2.)
            heights.append( hcurrent )
        else:
            heights.append(0.0)
    return heights


def nsimplices_all_heights(num_total_point, dis_sq, num_simplex_point, \
    seed=1):
    """ 
    For a set of num_total point points with pairwise distances dis_sq, determine \
        the height of each point, by drawing 100 groups of simplices for each point, \
        with each simplex has num_simple_point.

    Parameters
    ----------
    num_total_point: int
        The total number of points in dis_sq
    dis_sq: list[float]
        Squared-from pairwise distances
    num_simplex_point: int
        The number of points for the drawn simplex
    seed: int
        The seed for picking points to form simplices, default 1

    Returns
    -------
    heights: list[float]
        The heights for all points
    """

    alea.seed(seed)
    heights=num_total_point * [float('NaN')]
    
    
    # computation of h_i for each i
    for idx in range(num_total_point):
        
        num_group = 100
        # we draw 100 groups of (num_simplex_point) points, \
        # to create n-simplices and then compute the height median for i
        idx_heights = \
            nsimplices_heights(dis_sq, num_total_point, num_group, idx, num_simplex_point)
        
        #w e here get h[i] the median of heights of the data point i
        heights[idx] = np.median(idx_heights)
        
    return heights


def MDS(dis_sq,already_centered=False):
    """
    Classical multidimensional scaling for pairwise distances dis_sq

    Parameters
    ----------
    dis_sq: list[float]
        The symetric and squared-form pairwise distances with diagonal being 0s
    already_centered: bool
        dis_sq is already centered, no need for double centering, default False.
    
    Returns
    ----------
    list[float]:
        The sorted eigenvalues
    2D np array of float:
        The eigenvectors sorted by their eigenvalues
    2D np array of float
        The underlying coordinates
    """

    width,height = np.shape(dis_sq)
    if width != height:
        sys.exit("D must be symetric...")
        
    # Double centering
    if not already_centered:
        jaco = np.eye(width)-np.ones((width,width))/width
        dis_sq_centered =-0.5*np.dot(jaco, np.dot(dis_sq ** 2, jaco))
    else: 
        # allows robust centering with mediane and mad
        # outside this routine
        dis_sq_centered = dis_sq
    
    # Eigenvectors
    evals, evecs = np.linalg.eigh(dis_sq_centered)
    
    # Sort by eigenvalue in decreasing order, consider all the eigenvectors 
    idx = np.argsort(abs(evals))[::-1]
    evecst = evecs[:,idx]
    evalst= evals[idx] 
    
    # Undelying coordinates 
    idx_pos, = np.where(evalst>0) # only  consider eigenvalues > 0
    coords = np.dot(evecst[:,idx_pos], np.diag(evalst[idx_pos]**0.5))
    
    return evalst[idx_pos], evecst[:,idx_pos], coords


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


def find_subspace_dim(pairwise_dis, dim_start, dim_end):
    """
    Find the subspace dimension formed by the pairwise distance matrix pairwise_dis
    Parameters
    ----------
    pairwise_dis: 2D np array of float
        The squared matrix form of pairwise distancs
    dim_start: int, default 2
        Lowest dimension to test (inclusive)
    dim_end: int, default 6
        Largest dimension to test (inclusive)

    Returns
    -------
    subspace_dim: int
        The relevant dimension of the dataset
    outlier_indices: list[int]
        A list of indices of the orthogonal outliers 
    """
    
    point_num = np.shape(pairwise_dis)[0]
        
    med_height =np.zeros((dim_end-dim_start+1))
    dim_height_map = {}

    
    # determine of the screeplot nb_outliers function of the dimension tested
    for dim in range(dim_start,dim_end+1):           
        cur_height = nsimplices_all_heights(point_num, pairwise_dis, dim, seed=dim+1)     
        cur_height = np.array(cur_height)
        med_height[dim-dim_start] = np.median(cur_height)
        dim_height_map[dim] = cur_height
    
    # determine of the subspace dimension
    dims = np.array(range(dim_start, dim_end+1),dtype=float)
    print("med_height is:", med_height)
    subspace_dim = np.argmax(med_height[0:len(dims)-1]/med_height[1:len(dims)])+dim_start+1
    print("subspace_dim one is:", subspace_dim)
    
    # detect outliers in dimension subspace_dim
    subspace_heights = dim_height_map[subspace_dim]
    print("subspace_heights for dimension", subspace_dim, "is:", subspace_heights)
    subspace_height_size = subspace_heights.size
    
    subspace_med = np.median(subspace_heights)
    subspace_std = stats.median_abs_deviation(subspace_heights)
    subspace_mean = np.mean(subspace_heights)
    
    thres = subspace_mean + 3 * subspace_std # TODO: consider make 5 a parameter
    print("thres is:", thres, "mean is:", subspace_mean, "std is:", subspace_std)
    all_indices = np.array(range(subspace_height_size))
    outlier_indices = all_indices[subspace_heights > thres]
    print("outlier indices are:", outlier_indices)
    for idx in outlier_indices:
        print("idx is:", idx, "height is:", subspace_heights[idx], "thres is:", thres)
    
    
    # correct the bias obtained by subspace dimension
    outlier_prop = outlier_indices.shape[0]/subspace_height_size
    subspace_dim = subspace_dim - int(subspace_dim * outlier_prop) # TODO: check this with Khanh, round vs. floor

    return int(subspace_dim), outlier_indices

def nsimplices(pairwise_dis, feature_num, dim_start, dim_end, euc_coord=None):
    """
    The nSimplices method
    Parameters
    ----------
    pairwise_dis: 2D np array of float
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
    
    subspace_dim, outlier_indices = find_subspace_dim(pairwise_dis, dim_start, dim_end)
    
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


def sim_outliers(df, prop, col_start, col_end, out_dist = alea.uniform(-100,100), \
    res_outlier_indices = None):
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
    res_outlier_indices: list[int]
        Only selects outliers from these restriccted outlier indices

    Returns
    -------
    df_new: list[list[float]]
        A new dataframe with outliers
    """

    # if there is no restriction on outlier indices, generate from all indices
    if res_outlier_indices is None:
        res_outlier_indices = range(df.shape[0])

    num_point = df.shape[0]
    df_new = df.copy()
    num_outliers=math.floor(np.ceil(prop * num_point))
    # random draw of outliers 
    outlier_indices=np.sort(alea.sample(res_outlier_indices,num_outliers))
    for n in outlier_indices:
        horsplan=out_dist
        i=alea.randint(col_start,col_end)
        df_new.loc[n,i] = horsplan
    return df_new