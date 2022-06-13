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

use_kurtosis=False


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
    VnSquare=np.linalg.det(CMmat**2) # or numpy.linalg.slogdet
    return np.sqrt(abs(VnSquare/denominator))


""" Convert cm to inches     
"""

def cm2inch(value):
    return value/2.54

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
def nSimplwhichh(N,data,trim,n,seed=1,figpath=os.getcwd(), verbose=False):
    alea.seed(seed)
    h=N*[float('NaN')]
    hs=N*[float('NaN')]
    Jhn=[]
    
    
    # Computation of h_i for each i
    for i in range(N):
        
        B=100
        #we draw B groups of (n-1) points, to create n-Simplices and then compute the height median for i
        (B,hcollection)=DrawNSimplices(data,N,B,i,n)
        
        #we here get h[i] the median of heights of the data point i
        h[i] = np.median(hcollection)
        
    return h,hs,Jhn



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
    
    # Sort by eigenvalue in decreasing order
    idx = np.argsort(abs(evals))[::-1]
    #idx = np.argsort(evals)[::-1]
    evecst = evecs[:,idx]
    evalst= evals[idx]    
    
    # Undelying coordinates 
    idxPos,=np.where(evalst>0)
    Xe=np.dot(evecst[:,idxPos],np.diag(evalst[idxPos]**0.5))
    
    return evalst[idxPos], evecst[:,idxPos], Xe

"""
    Detect the outliers in dimension n using the defined criteria, and give the outlier list
"""

def DetectOutliers_n(N,data,trim,cutoff,n,Med):
    
    h,hs,Jhn = nSimplwhichh(N,data,trim,n,seed=n+1)
    
    h=np.array(h)
    hs1=np.std(h)
    Jhn=np.median(h)
    hw=h[h<2*hs1+Jhn]
    hs2=np.std(hw)
    hs=[hs1,hs2]
    
    #histogram of heights
    
    # plt.figure()
    # plt.title("Histogram of heights in dim"+str(n))
    # plt.hist(h,bins=50)
    # plt.axvline(x=np.sqrt(cutoff*np.sqrt(2.)/n**2*n/2)*Med, linewidth=3, color='k') 
    # plt.axvline(x=np.sqrt(cutoff/np.sqrt(2.)/1.5/2/1.5)*Med, linewidth=3, color='r')
    # plt.show()
    
    
    #Computation of h_i / median(delta_.,i), to detect outliers                                                   
    honmediandist = (h / Med)**2  
    
    list_outliers = list(np.where( honmediandist >cutoff*np.sqrt(2.)/2/1.5/2/1.5))[0] #/n**2 #*np.sqrt(2.)/n**2*n/2
    
    return list_outliers,h,hs,Jhn
    
"""
    Correct the distance matrix, to reduce the outlying-ness
"""

def CorrectDistances(N,distances,list_outliers,n,h):
    cdata=1.0*distances
    for i in list_outliers:
        
        for j in [x for x in range(N) if x!=i]:
            
            cdata[i,j] = cdata[j,i] = np.sqrt(np.max((0.,(cdata[i,j]**2 - h[i]**2))))
            
            
    return cdata


"""
    Other correction, most used:
    
    Correct the coordinates matrix, by projecting the outliers on the subspace of dimensionality rdim
"""

def CorrectProjection(N,Data,list_outliers,rdim):
    
    d=Data.shape[1]
    Data_corr=Data*1.0
    
    #remove outliers for the PCA?
    remove=True
    
    if remove:
        Data_pca=np.delete(Data,list_outliers,0)
    else:
        Data_pca=1.0*Data
    
    pca_method = PCA(n_components=rdim)
    method = pca_method.fit_transform(Data_pca) 
    vectors=pca_method.components_
    print(vectors)
    
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
    

def nSimpl_RelevantDim_ScreePlot(coord,data,cutoff,trim,n0=2,nf=6):
    
    N=np.shape(data)[0]
    dico_outliers   = {}
    dico_h          = {}
    
    stop=False
    
    nb_outliers = np.zeros((nf-n0))
    
    Med=np.median(data)
    print(Med)
    
    hmed=np.zeros((nf-n0))
    hstd1=np.zeros((nf-n0))
    hstd2=np.zeros((nf-n0))
    
    # Iteration on n: determination of the screeplot nb_outliers function of the dimension tested
    for n in range(n0,nf):
        
        if not stop:
            
            #outlier detection
            list_outliers,h,hs,Jhn = DetectOutliers_n(N,data,trim,cutoff,n,Med) 
            dico_outliers[n]=list_outliers
            nb=len(list_outliers)
            nb_outliers[n-n0]=nb
            dico_h[n] = h
            
            hmed[n-n0]=Jhn
            hstd1[n-n0]=hs[0]
            hstd2[n-n0]=hs[1]
            
            print ("Test dim"+str(n)+ ": Il y a "+str(nb)+" outliers : "+str(dico_outliers[n]))
            
            if (nb==0):
                stop=True
            
    
    dimension=np.array(range(n0,nf),dtype=float)
    
    plt.figure()
    plt.scatter(dimension,hstd1,label="std deviation of heights")
    plt.scatter(dimension,hstd2,label="std deviation of limited heights")
    plt.scatter(dimension,hmed,label="median of heights")
    plt.plot(dimension[0:len(dimension)-1],np.abs(np.diff(hmed)),label="pente mediane hauteurs")
    plt.plot(dimension[0:len(dimension)-1],np.abs(np.diff(hmed))/hmed[0:len(dimension)-1],label="% décroissance mediane hauteurs")
    plt.plot(dimension[0:len(dimension)-1],hmed[0:len(dimension)-1]/hmed[1:len(dimension)],label="décroissance mediane hauteurs hn/hn-1")
    plt.plot(dimension, [Med]*(nf-n0),label="median of distances")
    Dstd=np.std(data)
    plt.plot(dimension,[Dstd]*(nf-n0),label="std deviation of distances")
    plt.legend()
    plt.show()
    
    #Determination of the relevant dimension with a 2 piecewise linear function
    
    # def piecewise_linear(x, x0, y0, k1, k2):
    #     return np.piecewise(x, [x < x0, x >= x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
    # 
    # print(dimension)
    # print(nb_outliers)
    # 
    # p , e = optimize.curve_fit(piecewise_linear, dimension-dimension[0], nb_outliers)
    # xd = np.linspace(dimension[0], dimension[-1], 100)
    # 
    # plt.figure()
    # plt.plot(dimension,nb_outliers, "x")
    # plt.plot(xd, piecewise_linear(xd-dimension[0], *p))
    # plt.show()
    # 
    # if (p[1]<1.):
    #     rdim=round(n0+p[0]-1)
    # else:
    #     rdim=round(p[0]+dimension[0])
    # print(rdim)
    # rdim=2
    
    def sigmoid(x, A, lambd, B,C):
        
        return (A/(1.0+B*np.exp(lambd*x))+C)
    
    print(dimension)
    print(nb_outliers)
    
    p0=[nb_outliers[0],1,0,0]
    
    p , e = optimize.curve_fit(sigmoid, dimension, nb_outliers,p0,maxfev=50000)
    xd = np.linspace(dimension[0], dimension[-1], 100)
    
    plt.figure()
    plt.plot(dimension,nb_outliers, "x")
    plt.plot(xd, sigmoid(xd, *p))
    plt.show()
    
    print("A/(1+B*exp(lambda*x))")
    print("A="+str(p[0]))
    print("lambda="+str(p[1]))
    print("B="+str(p[2]))
    print("C="+str(p[3]))
    
    A=p[0]
    lambd=p[1]
    B=p[2]
    
    
    
    p=0.03
    rdim=int(round(1/lambd*np.log((1-p)/p/B)))
    print(1/lambd*np.log((1-p)/p/B))
    
    #Correction of the distances in the relevant dimension
    
    print("correction")
    cdata=1.0*data
    h=dico_h[rdim]
    cdata=CorrectDistances(N,cdata,dico_outliers[rdim],rdim,h)
    
    #correct by projection on the plan (if plan vectors known)
    print("correction")
    cdata_proj,coord_corr=CorrectProjection(N,coord,dico_outliers[rdim],rdim)
    
    
    return nb_outliers, dico_outliers, dico_h , rdim , cdata , cdata_proj,coord_corr
    

def nSimpl_RelevantDim_ScreePlot_v2(coord,data,cutoff,trim,d,n0=2,nf=6):
    
    N=np.shape(data)[0]
    dico_outliers   = {}
    dico_h          = {}
    
    stop=False
    
    nb_outliers = np.zeros((nf-n0))
    
    Med=np.median(data)
    print(Med)
    
    hmed=np.zeros((nf-n0))
    hstd1=np.zeros((nf-n0))
    hstd2=np.zeros((nf-n0))
    
    # Iteration on n: determination of the screeplot nb_outliers function of the dimension tested
    for n in range(n0,nf):
        
        if not stop:
            
            h,hs,Jhn = nSimplwhichh(N,data,trim,n,seed=n+1)
            
            h=np.array(h)
            hs1=np.std(h)
            Jhn=np.median(h)
            #hw=h[h<2*hs1+Jhn]
            #hs2=np.std(hw)
            #hs=[hs1,hs2]
            
            dico_h[n] = h
            
            #histogram of heights
            
            # plt.figure()
            # plt.title("Histogram of heights in dim"+str(n))
            # plt.hist(h,bins=50)
            # plt.axvline(x=np.sqrt(cutoff*np.sqrt(2.)/n**2*n/2)*Med, linewidth=3, color='k') 
            # plt.axvline(x=np.sqrt(cutoff/np.sqrt(2.)/1.5/2/1.5)*Med, linewidth=3, color='r')
            # plt.show()
            print(n)
            
            
            hmed[n-n0]=Jhn
            print(Jhn)
            #hstd1[n-n0]=hs[0]
            #hstd2[n-n0]=hs[1]
    
    
    #Determination of the relevant dimension
    
    dimension=np.array(range(n0,nf),dtype=float)
    
    plt.figure()
    
    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 26
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    
    
    # plt.scatter(dimension,hstd1,label="std deviation of heights")
    # plt.scatter(dimension,hstd2,label="std deviation of limited heights")
    plt.scatter(dimension[1:],hmed[1:],label="median of heights")
    
    pente=np.abs(np.diff(hmed))
    #plt.plot(dimension[0:len(dimension)-1],pente,label="slope of heights medians")
    # plt.plot(dimension[0:len(dimension)-1],np.abs(np.diff(hmed))/hmed[0:len(dimension)-1],label="% décroissance mediane hauteurs")
    plt.plot(dimension[1:len(dimension)-1],hmed[0:len(dimension)-2]/hmed[1:len(dimension)-1],label="heights median ratio: hn-1/hn")
    # plt.plot(dimension, [Med]*(nf-n0),label="median of distances")
    # Dstd=np.std(data)
    # plt.plot(dimension,[Dstd]*(nf-n0),label="std deviation of distances")
    plt.legend()
    plt.show()
    
    
    
    fig, ax1 = plt.subplots()
    #plt.title('Dimension detection',fontsize=28)
    ax1.scatter(dimension[1:],hmed[1:], c='royalblue')
    ax1.set_xlabel(r'dimension tested $n$')
    #ax1.xaxis.set_ticks(range(4,14,3))
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('median of heights', color='royalblue')
    for tl in ax1.get_yticklabels():
        tl.set_color('royalblue')
    
    
    ax2 = ax1.twinx()
    ax2.plot(dimension[1:len(dimension)],hmed[0:len(dimension)-1]/hmed[1:len(dimension)], 'k')
    ax2.set_ylabel(r'heights median ratio: $h_{n-1}/h_{n}$', color='k')
    for tl in ax2.get_yticklabels():
        tl.set_color('k')
    plt.show()

    
    rdim=np.argmax(hmed[0:len(dimension)-1]/hmed[1:len(dimension)])+n0+1
    print(rdim)
    
    
    #Detection of outliers in dimension rdim
    
    heights=dico_h[rdim]
    N=heights.size
    
    plt.figure()
    plt.title("Heights computed compared to real distances")
    # distances=np.zeros((1,N))
    # for i in range(10,13):
    #     cmed=np.median(coord[:,i])
    #     distances+=(coord[:,i]-cmed)*(coord[:,i]-cmed)
    # distances=np.sqrt(distances)
    # plt.scatter(distances,heights,s=4,color='k')
    #plt.plot(np.linspace(0,50,100),np.linspace(0,50,100),linewidth=1)
    plt.show()
    
    
    h_med=np.median(heights)
    h_std=stats.median_abs_deviation(heights)
    
    print(h_med,h_std)
    
    limit=h_med+5.4*h_std
    
    integers=np.array(range(N))
    outliers=integers[heights>limit]
    print(outliers)
    
    plt.figure()
    #plt.title("Histogram of heights in dim"+str(rdim))
    plt.hist(dico_h[rdim],bins=50,label="heights") 
    plt.axvline(x=limit, linewidth=1, color='r',label="limit outliers/non-outliers")
    plt.legend()
    plt.xlabel("heights")
    plt.ylabel("counts")
    plt.show()
    # 
    # plt.figure()
    # plt.title("Histogram of inliers heights")
    # plt.hist(heights[heights<5],bins=100,label="heights") 
    # plt.axvline(x=limit, linewidth=1, color='r',label="limit outliers/inliers")
    # plt.legend()
    # plt.show()
    
    blues=np.array([ [198,219,239,256*0.7], [158,202,225,256*0.7], [107,174,214,256*0.7], [66,146,198,256*0.7], [33,113,181,256*0.7], [8,69,148,256*0.7]])/256
    
    blues=np.array([[255,255,217,256*0.8],[199,233,180,256*0.8], [65,182,196,256*0.9], [34,94,168,256*0.9], [8,29,88,256*0.8]])/256
    
    plt.figure()
    
    SMALL_SIZE = 30
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 32
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    """
    #plt.title("Distribution of heights",fontsize=28)
    plt.hist(dico_h[2],bins=50,label="dim2",color=blues[0],edgecolor = 'black') 
    plt.hist(dico_h[6],bins=50,label="dim6",color=blues[1],edgecolor = 'black')
    plt.hist(dico_h[rdim-2],bins=50,label="dim8",color=blues[2],edgecolor = 'black')
    plt.hist(dico_h[rdim-1],bins=50,label="dim9",color=blues[3],edgecolor = 'black')
    plt.hist(dico_h[rdim],bins=50,label="dim10",color=blues[4],edgecolor = 'black')
    #plt.hist(dico_h[rdim+2],bins=100,label="dim11",color=blues[5])
    plt.legend()
    axes = plt.gca()
    
    axes.set_xlim(-2,20)
    axes.set_xlabel('height')
    axes.set_ylabel('counts')
    plt.show()
    """
    
    #Correction of the bias obtained on rdim
    
    p=outliers.shape[0]/N
    rdim=rdim-int(round(rdim*p))
    print(rdim)
    
    
    #Correction of the distances in the relevant dimension
    print("correction")
    cdata=1.0*data
    h=dico_h[rdim]
    cdata=CorrectDistances(N,cdata,outliers,rdim,h)
    
    #correct by projection on the plan (if plan vectors known)
    print("correction")
    cdata_proj,coord_corr=CorrectProjection(N,coord,outliers,rdim)
    
    #Correction thanks to MDS and PCA on the distance matrix
    
    clf = manifold.MDS(n_components=d, max_iter=100000000000,dissimilarity='precomputed')
    cdata2_coord = clf.fit_transform(data)
    #dico_h=cdata2_coord
    cdata_proj,coord_corr=CorrectProjection(N,cdata2_coord,outliers,rdim)
    
    
    
    
    nb_outliers=outliers.size
    
    return nb_outliers, outliers, dico_h , rdim , cdata , cdata_proj,coord_corr