from scipy.spatial.distance import pdist, squareform
import os
import numpy as np

exec(open("./nsimplices.py").read())

data_path = os.path.join("./data/hmp_v13lqphylotypecounts_rs_c.csv")
df_hmp = np.loadtxt(data_path, delimiter=",")
D = squareform(pdist(df_hmp))


# # RMDS smacof
# exec(open("./Robust-MDS-methods/RMDS_tunParSD.py").read())
# _, similarity_matrix, _, _, _ = RMDS_D(np.transpose(D), n=10,NbIter=100,graine=987654321,tuningparameter=None)
# va, ve, Xe =  cMDS(similarity_matrix)

# np.savetxt("./outputs/hmp_RMDS_D_coord.txt", Xe, fmt='%f')


# RSMDS smacof
exec(open("./Robust-MDS-methods/RSMDS_D.py").read())
_,similarity_matrix, _, _, _, _ =RSMDS_D(np.transpose(D), n=10,NbIter=100,graine=987654321, tunMu=2.,tunGamma=10.,tunLambda=10.)   
va, ve, Xe =  cMDS(similarity_matrix)
np.savetxt("./outputs/hmp_RSMDS_D_coord.txt", Xe, fmt='%f')


# LAR
exec(open("./Robust-MDS-methods/smacof_LAR_D.py").read())
_,similarity_matrix, _, _ = LARsmacof_D(np.transpose(D),n=10,NbIter=100)
va, ve, Xe =  cMDS(similarity_matrix)
np.savetxt("./outputs/hmp_LAR_coord.txt", Xe, fmt='%f')

