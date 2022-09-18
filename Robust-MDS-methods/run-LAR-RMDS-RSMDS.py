#!/usr/local/bin/python
# -*- coding: utf-8 -*-

D=squareform(pdist(data, metric='euclidean'))

    #"""
    # RMDS smacof
    execfile("/data/clegrand/Travail/diss/py/Lib/RMDS_tunParSD.py")
    Xtp1,dXtp1, Otp1,Stress, previousStress=RMDS_D(D,
      n=10,NbIter=100,graine=987654321,tuningparameter=None)
    
    # """
    # RSMDS smacof
    execfile("/data/clegrand/Travail/diss/py/Lib/RSMDS_D.py")
    Xtp1,dXtp1, Btp1, Ctp1, Stress, previousStress=RSMDS_D(D,
      n=10,NbIter=100,graine=987654321,
      tunMu=2.,tunGamma=10.,tunLambda=10.)   
    
    # """
    # LAR
    execfile("/data/clegrand/Travail/diss/py/Lib/smacof_LAR_D.py")
    X, dX, Stress, previousStress = LARsmacof_D(D,n=10,NbIter=100)
    
