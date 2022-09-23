from scipy.spatial.distance import pdist, squareform
import os
import numpy as np

data_path = os.path.join("./data/hmp_v13lqphylotypecounts_rs.csv")
df_hmp = np.loadtxt(data_path, delimiter=",")
D = squareform(pdist(df_hmp))


#"""
# RMDS smacof
exec(open("./Robust-MDS-methods/RMDS_tunParSD.py").read())
Xtp1,dXtp1, Otp1,Stress, previousStress=RMDS_D(D, n=10,NbIter=100,graine=987654321,tuningparameter=None)

# """
# RSMDS smacof
exec(open("./Robust-MDS-methods/RSMDS_D.py").read())
Xtp1,dXtp1, Btp1, Ctp1, Stress, previousStress=RSMDS_D(D, n=10,NbIter=100,graine=987654321, tunMu=2.,tunGamma=10.,tunLambda=10.)   

# """
# LAR
exec(open("./Robust-MDS-methods/smacof_LAR_D.py").read())
X, dX, Stress, previousStress = LARsmacof_D(D,n=10,NbIter=100)
    
