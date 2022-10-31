#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# Implementation by CARINE LEGRAND, in 2017, of the following method  :
#
# SMACOF - LAR : Stress MAjorization of a COmplex Function - Least Absolute Residuals
# LAR relies on l1-norm instead of l2
# The method was proposed by Heiser (1988) :
#   Heiser WJ (1988) Multidimensional scaling with least absolute residuals, pp. 455-462.
#   In: Bock HH: Classification and related methods of data analysis. Amsterdam.
#
# external libraries
import numpy as np
import random
from scipy.linalg import solve
from scipy.spatial.distance import pdist, squareform

def LARsmacof_D(D,n=10,NbIter=1000,graine=123456789):
    ## Set random seed
    random.seed(graine)
    #X: SNPsxIndividuals matrix, dimensions (n0,p)
    #n: nb of dimensions desired in final result
    p=  np.shape(D)[0]
    np.fill_diagonal(D,0.)       #Dmodified in-place
    ## Initialize companion matrices X2, Y ; Initialize stress (or loss function) and epsilon
    Y=X2=    np.array([random.random() for i in range(n*p)]).reshape((n,p))
    dX=      squareform(pdist(X2.T, 'euclidean'))
    Stress=  np.sum(abs(D-dX))
    epsilon= 0.000001
    previousStress= Stress+1000.
    
    for t in range(NbIter):
        print('Stress, previousStress= ',Stress, previousStress)
        if (previousStress-Stress)>epsilon:
        #if 1:
            print('iteration ',t,' of ',NbIter)
            #####################
            # Minimization step #
            ####################
            ## Calculation of matrix A in d(LAR+)/dX = XA+B = 0
            # A = L + tL where lii= np.sum(uik(Y), k<>i) for diagonal elements
            #                   lij= -uij(Y) for off-diagonal elements
            # Then, uij(Y)=1 / |Dij-dij(Y)|
            #                  where norm1=sum(abs(dij for i=1,p for j<>i)),
            #                        dij(Y)=np.sqrt(sum((Yli-Ylj)**2 for i=1,p for j<>i))
            # Case |Dij-dij(Y)|=0 or small: - Cf. Heiser27 and Ekblom28
            #                               - Replace x=|Dij-dij(Y)| with: sqrt(x**2+epsilon**2)
            dY= squareform(pdist(Y.T, 'euclidean'))
            uY= 1/np.sqrt(epsilon**2+(abs(D-dY))**2)
            L=  np.array(p*[p*[float('NaN')]])
            for i in range(p):
                L[i,i]= np.sum([uY[i,k] for k in [x for x in range(p) if x !=i]])
                for j in [x for x in range(p) if x!=i]:
                    L[i,j]=L[j,i]= -uY[i,j]
            
            A=  L+L.T
            #
            ## Calculation of matrix B in XA+B=0
            # B =  2.X2.tL2 where l2ii= np.sum(vik(Y), k<>i) for diagonal elements
            #                      l2ij= -vij(Y) for off-diagonal elements
            # Then, vij= uij(Y)*Dij/dij(X2)
            dX2= squareform(pdist(X2.T, 'euclidean'))
            for i in range(p):  # to avoid division by zero in next command
                dX2[i,i]= 1
            v=   uY*D/dX2                    # (check: (i,j)=(5,49) ; uY[i,j]*D[i,j]/dX2[i,j] ; v[i,j])
            L2=  np.array(p*[p*[float('NaN')]])
            for i in range(p):
                L2[i,i]= np.sum([v[i,k] for k in [x for x in range(p) if x !=i]])
                for j in [x for x in range(p) if x!=i]:
                    L2[i,j]= L2[j,i]= -v[i,j]
            B= 2*np.dot(X2,L2.T)
            
            #
            ## Minimization by finding root of derivative d(LAR+)/dX = XA+B
            #
            Y_new=X2_new=solve(A.T,-B.T).T
            
            ## Stress at current step
            #  Stress = sum_over_i_and_j>i(abs(delta_ij-d_ij))
            # current d_ij calculated over X2_new
            previousStress= Stress
            dX=             squareform(pdist(X2_new.T, 'euclidean'))
            Stress=         np.sum(abs(D-dX))
    
    return X2_new, dX, Stress, previousStress
