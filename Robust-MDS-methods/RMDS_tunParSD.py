#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# Implementation by CARINE LEGRAND, in 2017, of the following method :
#
# Robust MDS (RMDS), as introduced by Forero and Giannakis (2012).
# RMDS is an iterative procedure which aims at regularizing distance outliers 
# and therefore accommodates for them.
# 
# Full reference for Forero and Giannakis (2012) :
#   Forero PA, Giannakis GB (2012) Sparsity-exploiting robust multidimensional scaling. IEEE Trans Signal
#   Process 60:4118-4134

##################################################
#                     RMDS                       #
##################################################
# external libraries
import numpy as np
import random
from scipy.linalg import solve,pinv,pinv2
from scipy.spatial.distance import pdist, squareform
from statsmodels.robust.scale import mad
# own libraries
#execfile("/data/clegrand/Travail/diss/py/Lib/libPca_v1_2.py")

def RMDS_D(D,n=10,NbIter=1000,graine=987654321,tuningparameter=None):
    #### Initializations and inputs
    #Xdissim: SNPsxIndividuals matrix, dimensions (n0,p)
    #n: nb of dimensions desired in final result
    p=np.shape(D)[0]
    random.seed(graine)
    epsilon=0.000001    
    #Xt = np.array([random.random()- for i in range(n*p)]).reshape((n,p))
    Xt = np.array([random.uniform(a=0,b=12) for i in range(n*p)]).reshape((n,p))
    Ot = np.array(p*[p*[0.]])
    
    # Tuning Parameter
    #   lambda=tunPar as 2.69 * standard deviation (if known)
    
    # 
    #allDeltas=[D[i,j] for i in range(1,p) for j in range((i+1),p)]
    #stdDev=mad(allDeltas, c=(1./1.4826))  #stddev estimated from MAD
    #tunPar=2.69*stdDev
    # temporarily replaced by large value, to achieve classic l2 MDS solution :
    ##tunPar=10000.
    
    #   lambda=tunPar from Proposition1 -- see above --
    S=int(0.10*p)
    
    dXt= squareform(pdist(Xt.T, 'euclidean'))
    
    squaredResiduals=(D-dXt)**2
    uniqSquaredResiduals=[squaredResiduals[i,j] for i in range(1,p) for j in range((i+1),p)]
    rankedResiduals=np.sort(uniqSquaredResiduals)[::-1]
    # -- see above -- large tunPar for l2 solution ; 
    #tunPar=(rankedResiduals[(S-1)]+rankedResiduals[S])/2.


    # Tuning Parameter
    #   lambda=tunPar as 2.69 * standard deviation (if known)
    # 
    allDeltas=[D[i,j] for i in range(1,p) for j in range((i+1),p)]
    stdDev=np.std(allDeltas) # mad(allDeltas, c=(1./1.4826))  #stddev estimated from SD or MAD
    tunPar=2.69*stdDev
    
    
    if tuningparameter is not None:
        tunPar=tuningparameter
    
    print("tuning parameter = ",tunPar)
    
    
    
    # Init
    Xtp1=Xt
    Otp1=Ot
    Stress= np.sum((D-dXt-Ot)**2) + tunPar*np.sum(abs(Ot))
    Stress=1000. + Stress
    
    for t in range(NbIter):
        # Stress(X(t),O(t)) and previous stress
        Xt=np.float64(Xtp1)
        Ot=np.float64(Otp1)
        previousStress=Stress
        dXt= squareform(pdist(Xt.T, 'euclidean')) 
        Stress= np.sum((D-dXt-Ot)**2) + tunPar*np.sum(abs(Ot))
        #print 'Stress=',Stress
        # Next iteration
        if 1:
            #if t<10 or (previousStress-Stress)>epsilon:
            
            #1st half iteration O(t+1)=argmin g(O,X(t);X(t)) via (25)
            dd=D-dXt # dd = x = delta_n,m - d_n,m(Xt)
            Otp1=np.float64(np.sign(dd)*np.maximum(np.array(p*[p*[0.]]),(abs(dd)-tunPar/2.)))
            
            #2nd half iteration X(t+1)=argmin g(O(t+1),X;X(t)) via (27) with L1 as in (18a) and (17a) and L as described below(16)
            #   L: p-1 on diagonal, -1 off diagonal
            L=np.float64(np.array(p*[p*[float('NaN')]]))
            for i in range(p):
                L[i,i]= p-1
                for j in [x for x in range(p) if x!=i]:
                    L[i,j]=L[j,i]= -1.
            #Lcross=solve(L,np.identity(p)) #pinv(L) is less accurate
            # marche mais mis de côté par sûreté: Lcross=np.float64(np.diag(p*[1./float(p)])) #analytical solution instead of pinv2
            Lcross=pinv2(L)
            #print "L=",L
            #print "Lcross=",Lcross
            #   (17a) and (18a)
            #   NB: diag(A1) in (18a) is actually a degree matrix with the count of non-zero 
            #       connections between one individual and the others, on its diagonal.
            A1=np.float64(np.array(p*[p*[float('NaN')]]))
            degree_A1=np.float64(np.array(p*[p*[0.]]))
            for i in range(p):
                A1[i,i]= 0.
                for j in [x for x in range(p) if x!=i]:
                    if D[i,j]>Otp1[i,j] and dXt[i,j]>0. :
                        #if dXt[i,j]<Otp1[i,j] and D[i,j]>0. :
                        # For stability at dXt small, possibly replace dXt by sqrt(dXt^2+epsilon^2)
                        A1[i,j]=A1[j,i]=((D[i,j]-Otp1[i,j])/dXt[i,j])
                        #A1[i,j]=A1[j,i]=((dXt[i,j]-Otp1[i,j])/D[i,j])
                        degree_A1[i,i]=degree_A1[i,i]+1.
                    else :
                        A1[i,j]=A1[j,i]=0.
            
            L1=degree_A1-A1
            
            #   (27): X(t+1)= X(t) . L1(O(t+1),X(t)) . Lcross
            Xtp1=np.dot(Xt,np.dot(L1,Lcross))
            
        # If stress is not improved by further iterations, then stop
        else :
            break
    
    print('Final stress=',Stress)
    print("Xtp1 shape is:", Xtp1.shape)
    print("dXt shape is:", dXt.shape)
    print("Otp1 shape is:", Otp1.shape)
    return Xtp1,dXt, Otp1,Stress, previousStress
