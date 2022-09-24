#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Method by CARINE LEGRAND, in 2017 :
#
# Weighted Corrected Agreement between Counts. 
# This method was inspired by Oliehoek et al 2006.
#
#

from scipy.spatial.distance import pdist, squareform
import warnings
import numpy as np
import sys
    
"""
Computes the weighted corrected counts' similarity	(dist=False)
       		or distance (dist=True) matrix.
Tolerant to missing values

Parameters
----------
X :               2D ndarray, shape (n_samples, n_features)
                  Data from which to compute similarity or distance matrix
assume_centered : Boolean
                  Not used in WCS computation
                  Kept to maintain similar interface
dist :            distance matrix if true, similairyt matrix otherwise
Lmediane :        use mediane if true AND if simpl=FALSE, mean otherwise
Lmad :            use median absolute deviation if true AND if simpl=FALSE, 
                  use standard deviation otherwise.
form :            square matrix ('square') or distance matrix ('dist')
Returns
-------
WCS matrix : 2D ndarray, shape (n_features, n_features) 

-----------------------------------------------------------------------

Weighted Corrected Agreement between Counts:
a_xy= 1/W *sum_on_p(w_p*(S_xy,p - sp) / (max(abs)-sp))
where x, y are samples for which we calculate agreement a, 
        based on similarity per phylogeny S_xy,p, 
      p is a phylogeny 
      W=sum_on_p(w_p)
      w_p is the weight per phylogeny, w_p=(1-sp)^2/Scale(pairwise S_xy,p)
      s_p is the expected pairwise similarity among all S_xy,p, 
           for all (x,y) pairs, at phylogeny p.
           Here, s_p=median(S_xy,p) among (x,y) pairs.
      NB: luckily, for hmp data (restricted to v.introitus), S_xy,p ~ N(s_p,sigma)
      S_xy,p is the pairwise absolute difference between individual x and y.

Inspired from: Oliehoek et al 2006 equations (6) anf (7), page 486 + S_xy of equation (1), page 485
    Equation (6): r_xy= 2/W *sum_on_l(w_l*(S_xy,l - sl) / (1-sl))
    where r_xy is the relatedness between individuals x and y, W is the sum of w_l's over all loci,
          w_l is the weight for one locus, determined after equation (7),
          S_xy is the similarity for individuals x and y at locus l,
          and sl=1/nl = 1/(nb of alleles)
    Equation (7): 1/wl= sum_on_alleles_i(p_i^2)*(1-sum_on_alleles_i(p_i^2)) / (1-1/nl)^2
    where p_i is the allele i frequency. p_1 is p2, p_0 is 1-p2
    
    In biallelic case, equations (6) and (7) simplify to:  r_xy= 2/W * sum_on_l(w_l*S_xy,l/(1-1/nl)) - 2./(nl-1)
                                        w_l=  (1-1/nl)^2 / (sum_on_alleles_i(p_i^2)*(1-sum_on_alleles_i(p_i^2)))  
-----------------------------------------------------------------------

"""

def wcCounts_similDist(X,
          assume_centered=False,
          dist=False,
          Lmediane=False,
          Lmad=False,
          form='square',
          simpl=False,
          sdTrunc=None,
          weightCorrD=False
          ):
        
    X = np.asarray(X)
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))
        warnings.warn("Only one sample available. "
                      "You may want to reshape your data array")

    print("input data X shape is:", X.shape)
    # Initialisations
    # P: nb of phylotypes ; n: nb of samples
    P,n=np.shape(X)
    similDist=np.array(n*[n*[float('NaN')]])

    # Initialisations
    data=X  #(P,n) P:nb of phylogenies, n: nb of samples
    W=0.
    a_xy=np.array(0.)
    
    if not (simpl or weightCorrD):
        distDesSxy=True
    else:
        distDesSxy=False
    
    # Relatedness between samples for each phylogeny
    if simpl : 
        for p in range(P):
            # pairwise differences for phylogeny p
            S_xyp=[]
            for i in range((n-1)):
                S_xyp.extend(abs(X[p,i]-X[p,(i+1):]))
            S_xyp=np.array(S_xyp)
            # weight
            w_p=   1 / np.var(S_xyp)       
            # Weighted agreement for current phylogeny, cumulated.
            a_xy= a_xy + w_p*S_xyp
            # Weight, cumulated
            W=W+w_p
        # Pairwise agreement matrix
        similDist= 1/W * a_xy
        print("similDist in simpl shape is:", similDist.shape)
            
    elif distDesSxy:
        for p in range(P):
            # pairwise differences for phylogeny p
            S_xyp=[]
            for i in range((n-1)):
                # absolute distance for phylogeny p
                S_xyp.extend(abs(X[p,i]-X[p,(i+1):]))
            S_xyp=np.array(S_xyp)
            # center, scale and weighted corrections (weights) for current phylogeny
            if Lmediane:      s_p=   np.median(S_xyp)
            else:             s_p=   np.mean(S_xyp)
            if Lmad:          w_p=   1. / mad(S_xyp)**2
            else:             w_p=   1. / np.std(S_xyp)**2
            # Weighted agreement for current phylogeny, cumulated.
            a_xy= a_xy + w_p*(S_xyp-s_p)
            # Weight, cumulated
            W=W+w_p
        # Pairwise agreement matrix
        similDist= 1/W * a_xy
    
    # transform in similarity matrix or square matrix according to selected parameters
    if not dist:       similDist= 1-similDist/np.max(similDist) # assuming centering and scaling are done elsewhere
    #if dist:       similDist2= -similDist2 # assuming centering and scaling are done elsewhere
    if form=='square': similDist= squareform(similDist)
    #if form=='square': similDist2= squareform(similDist2)
    
    return similDist


