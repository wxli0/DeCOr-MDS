#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# Implementation by CARINE LEGRAND, in 2017, of the following method :
#
# Robust MDS for Structured Outliers (RSMDS), as introduced by Forero and Giannakis (2012).
# RSMDS is, like RMDS, an iterative procedure which aims at regularizing distance outliers,
# and therefore accommodates for them.
# As compared to RMDS, RSMDS is specially designed for structured outliers. It
# takes advantage of outliers sparsity.
# 
# Full reference for Forero and Giannakis (2012) :
#   Forero PA, Giannakis GB (2012) Sparsity-exploiting robust multidimensional scaling. IEEE Trans Signal
#   Process 60:4118-4134

# external libraries
import numpy as np
import random
import scipy.linalg as LA
from scipy.spatial.distance import pdist, squareform
from statsmodels.robust.scale import mad
# own libraries
#execfile("/data/clegrand/Travail/diss/py/Lib/libPca_v1_2.py")

def RSMDS_D(D,n=10,NbIter=1000,graine=987654321,
          tunMu=2.,
          tunGamma=10.,
          tunLambda=10.):
    """ Initializations and inputs
    D      :            IndividualsxIndividuals distance matrix, dimensions (p,p)
    n:                  nb of dimensions desired in final result
    tunMu/Gamma/Lambda  Tuning parameters >0, to choose via grid cross-validation
    """
    
    random.seed(graine)
    epsilon=0.000001
    
    ## Read or set dissimilarities, initial coordinates matrix, Outliers indicator matrix
    p = np.shape(D)[0]
    np.fill_diagonal(D,0.)       #D modified in-place
    Xt = np.array([random.random() for i in range(n*p)]).reshape((n,p))
    dXt= squareform(pdist(Xt.T, 'euclidean'))
    Bt = Ct = np.array(p*[p*[0.]])
    
    # Init
    Xtp1=Xt
    Btp1=Bt
    Ctp1=Ct
    # Stress is f_GAMMA as given in eq.30 in Forero and Giannakis:
    #              f_GAMMA = 1/2*squared Frobenius norm of (delta - D -B - C) + GAMMA(B,C)+mu/2*squared Frobenius norm of (B-C.T)
    #              , with GAMMA as in eq.31:
    #              GAMMA(B,C) = gamma*sum(norm2(bn)+norm2(cn)) + lambda*sum(norm1(bn)+norm1(cn))
    # Nota bene: bn's are columns of B, whereas cn's are rows of C.
    Stress= (1./2.*(LA.norm((D-dXt-Btp1-Ctp1),ord='fro'))**2 +
                    ( tunGamma  *(sum([LA.norm(Btp1[:,j])       for j in range(p)]) +sum([LA.norm(Ctp1[i,:])       for i in range(p)])) +
                      tunLambda *(sum([LA.norm(Btp1[:,j],ord=1) for j in range(p)]) +sum([LA.norm(Ctp1[i,:],ord=1) for i in range(p)])) )
                    +tunMu/2.*(LA.norm(Btp1-Ctp1.T,ord='fro'))**2 )
    
    Stress= Stress+1000.
    
    for t in range(NbIter):
        # Stress(X(t),O(t)) and previous stress
        Xt=np.float64(Xtp1)
        previousStress=Stress
        dXt= squareform(pdist(Xt.T, 'euclidean'))
        # Stress function:
        Stress= (1./2.*(LA.norm((D-dXt-Btp1-Ctp1),ord='fro'))**2 +
                 ( tunGamma  *(sum([LA.norm(Btp1[:,j])       for j in range(p)]) +sum([LA.norm(Ctp1[i,:])       for i in range(p)])) +
                   tunLambda *(sum([LA.norm(Btp1[:,j],ord=1) for j in range(p)]) +sum([LA.norm(Ctp1[i,:],ord=1) for i in range(p)])) )
                 +tunMu/2.*(LA.norm(Btp1-Ctp1.T,ord='fro'))**2 )
        print('Stress=',Stress)
        # Next iteration
        if 1:
        #if (previousStress-Stress)>epsilon:
            
            #Update B via (38)
            Btp1=np.float64(np.array(p*[p*[float('NaN')]]))
            for i in range(p):
                u=D[:,i]-dXt[:,i]-Ctp1[:,i]+tunMu*Ctp1[i,:]
                S2lambda=np.sign(u)*np.maximum(np.array(p*[0.]),(abs(u)-tunLambda))
                for j in range(p):
                    if S2lambda[j]==0: S2lambda[j]=epsilon
                Btp1[:,i]=1./(1+tunMu) * np.maximum(np.array(p*[0.]),(1-tunGamma/LA.norm(S2lambda))) * S2lambda
                
            #Update C via (40)
            Ctp1=np.float64(np.array(p*[p*[float('NaN')]]))
            for i in range(p):
                u=D[:,i]-dXt[:,i]-Btp1[i,:]+tunMu*Btp1[:,i]
                S2lambda=np.sign(u)*np.maximum(np.array(p*[0.]),(abs(u)-tunLambda))
                for j in range(p):
                    if S2lambda[j]==0: S2lambda[j]=epsilon
                Ctp1[:,i]=1./(1+tunMu) * np.maximum(np.array(p*[0.]),(1-tunGamma/LA.norm(S2lambda))) * S2lambda
              
            # ----------------  
            # Update X via (43)
            # ----------------  
            BU=np.triu(Btp1,1) ; CU=np.triu(Ctp1,1) ; BL=np.tril(Btp1,-1) ; CL=np.tril(Ctp1,-1)
            OU=BU+BU.T+CU+CU.T
            OL=BL+BL.T+CL+CL.T
            AU1=AL1=AU2=AL2=np.float64(np.array(p*[p*[0.]]))
            degree_AU1=degree_AL1=degree_AU2=degree_AL2=np.float64(np.array(p*[p*[0.]]))
            L=np.float64(np.array(p*[p*[float('NaN')]]))
            # --
            # Get Ltild=L+L2tild, and Moore-Penrose pseudo-inverse of L1tild
            # --
            #   L1tild(Btp1+Ctp1,Xt)=L1(BUtp1+CUtp1,Xt)+L1(BLtp1+CLtp1,Xt)
            #     L1(BUtp1+CUtp1,Xt)=degree_A1(BUpt1+CUtp1,Xt) - A1(BUtp1+CUtp1,Xt)
            #     L1(BLtp1+CLtp1,Xt)=-----------L-----L--------------L-----L-------
            #   L2tild(Btp1+Ctp1,Xt)=L2(BUtp1+CUtp1,Xt)+L2(BLtp1+CLtp1,Xt)
            #     L2(BUtp1+CUtp1,Xt)=degree_A2(BUpt1+CUtp1,Xt) - A2(BUtp1+CUtp1,Xt)
            #     L2(BLtp1+CLtp1,Xt)=-----------L-----L--------------L-----L-------
            for i in range(p):
                L[i,i]= p-1
                for j in range((i+1),p):
                    L[i,j]=L[j,i]= -1.
                    # Possibly replace dXt[i,j] by np.sqrt(dXt[i,j]**2+epsilon**2) for stability at small values
                    if D[i,j]>OU[i,j] and dXt[i,j]>0. :
                        AU1[i,j]=AU1[j,i]=-((D[i,j]-OU[i,j])/dXt[i,j])
                        degree_AU1[i,i]=degree_AU1[i,i]+1.
                    if D[i,j]>OL[i,j] and dXt[i,j]>0. :
                        AL1[i,j]=AL1[j,i]=-((D[i,j]-OL[i,j])/dXt[i,j])
                        degree_AL1[i,i]=degree_AL1[i,i]+1.
                    if D[i,j]<=OU[i,j] and dXt[i,j]>0. :
                        AU2[i,j]=AU2[j,i]=-((D[i,j]-OU[i,j])/dXt[i,j])
                        degree_AU2[i,i]=degree_AU2[i,i]+1.
                    if D[i,j]<=OL[i,j] and dXt[i,j]>0. :
                        AL2[i,j]=AL2[j,i]=-((D[i,j]-OL[i,j])/dXt[i,j])
                        degree_AL2[i,i]=degree_AL2[i,i]+1.
                    
            L1tild= degree_AU1-AU1 + degree_AL1-AL1
            L2tild= degree_AU2-AU2 + degree_AL2-AL2
            # --
            # Ltild's Moore-Penrose inverse, and update of Xt to Xtp1
            # --
            LtildMP=LA.pinv2(np.float64(L+L2tild)) # LA:pinv2 supposed to work better on large matrices
            Xtp1=np.dot(Xt,np.dot(L1tild,LtildMP))
        # If stress is not improved by further iterations, then stop
        else :
            break
    print('Final stress=',Stress)
    # TODO: is Xtp1 the eigenvalues?
    print("dXt shape is:", dXt.shape)
    return Xtp1,dXt, Btp1, Ctp1, Stress, previousStress
