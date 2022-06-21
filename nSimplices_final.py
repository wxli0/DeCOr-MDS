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

def CorrectProjection(N,Data,list_outliers,n_bar):
    
    d=Data.shape[1]
    Data_corr=Data*1.0
    
    Data_pca=np.delete(Data,list_outliers,0)
    
    pca_method = PCA(n_components=n_bar)
    method = pca_method.fit_transform(Data_pca) 
    vectors=pca_method.components_
    # print(vectors)
    
    Mean=np.mean(Data_pca,0)
    # print(Mean)
    for v in vectors:
        Mean=Mean-np.dot(Mean,v)*v
    
    for i in list_outliers:
        outlier=Data[i]
        sum=0
        projection=np.zeros(d)
        for v in vectors:
            projection+=np.dot(outlier,v)*v
        Data_corr[i,:]=projection+Mean
    
    
    Distances_corr=squareform(pdist(Data_corr))
    #Then, the distances data is prepared for MDS.
    
    return Distances_corr,Data_corr
    

def nSimplices(D,d,n0=2,nf=6):
    """
    The nSimplices method
    Parameters
    ----------
    D: int
        The squared matrix form of pairwise distancs
    d: int
        Number of components in MDS
    n0: int
        Lowest dimension to test
    nf: int
        Largest dimension to test

    Returns
    -------
    O: list[int]
        The list of orthogonal outliers 
    n_bar: int
        The relevant dimension of the dataset
    D_coor: list[list[float]]
        The list of corrected pairwise distance 
    coord_corr: list[list[float]]
        The list corrected coordinates
    """
    
    N=np.shape(D)[0]
    dico_outliers   = {}
    dico_h          = {}
    
    stop=False
    
    nb_outliers = np.zeros((nf-n0))
    
    Med=np.median(D)
    # print(Med)
    
    hmed=np.zeros((nf-n0))
    
    # Iteration on n: determination of the screeplot nb_outliers function of the dimension tested
    for n in range(n0,nf):
        
        if not stop:
            
            h= nSimplwhichh(N,D,n,seed=n+1)
            
            h=np.array(h)
            hs1=np.std(h)
            hmed[n-n0]=np.median(h)
            
            dico_h[n] = h
    
    #Determination of the relevant dimension
    
    dimension=np.array(range(n0,nf),dtype=float)
    n_bar=np.argmax(hmed[0:len(dimension)-1]/hmed[1:len(dimension)])+n0+1
    
    #Detection of outliers in dimension n_bar
    
    heights=dico_h[n_bar]
    N=heights.size
    
    h_med=np.median(heights)
    h_std=stats.median_abs_deviation(heights)
    
    limit=h_med+5*h_std
    integers=np.array(range(N))
    O=integers[heights>limit]
    #print(O)
    
    
    #Correction of the bias obtained on n_bar
    
    p=O.shape[0]/N
    n_bar=n_bar-int(round(n_bar*p))
    # print("Dimensionality found after correction:")
    # print(n_bar)
    
    
    #Correction thanks to MDS and PCA on the distance matrix
    
    # print("correction of outliers")
    
    clf = manifold.MDS(n_components=d, max_iter=100000000000,dissimilarity='precomputed')
    eucl_coord = clf.fit_transform(D)
    
    D_corr,coord_corr=CorrectProjection(N,eucl_coord,O,n_bar)
    
    
    return O, n_bar , D_corr, coord_corr


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