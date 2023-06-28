
import numpy as np
import matplotlib.pyplot as plt
import os
import random as alea
from scipy.linalg import solve,pinv,pinv2
from scipy.spatial.distance import pdist, squareform
import time
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import time


# import nSimplices 
# get_ipython().run_line_magic('matplotlib', 'widget')
exec(compile(open(r"nsimplices.py", encoding="utf8").read(), "nsimplices.py", 'exec'))

# set matplotlib default savefig directory
plt.rcParams["savefig.directory"] = os.getcwd() # To save figures to directory
                                                #   defined above

df_dim40 = pd.read_csv(r'data/synthetic_rdim40_2500.csv',sep=';',header=None)

ori_dis = pdist(df_dim40.copy())
ori_dis_sq = squareform(ori_dis)
num_point =df_dim40.shape[0]

""" Add outliers """

prop=0.02
num_outliers=int(np.ceil(prop * num_point))
# random draw of outliers 
outlier_indices=np.sort(alea.sample(range(num_point),num_outliers))
print("outlier_indices are:", outlier_indices)
for n in outlier_indices:
    outlier=alea.uniform(-100,100)
    
    # for each row, add outliers to one of columns 40 to 45 (inclusive)
    # columns 10 to 15 are originally simulated with Guassian(2, 0.05)
    i=alea.randint(40,45) 
    df_dim40.loc[n,i] = outlier

""" euclidean distances """
out_dis=pdist(df_dim40)
out_dis_sq=squareform(out_dis)

### Run nSimplices method
T1=time.time()
outlier_indices,subspace_dim,corr_dis_sq,corr_coord = nsimplices(out_dis_sq, df_dim40.shape[1], dim_start=30, dim_end=50, num_groups=100)
T2=time.time()
print("runtime is:", T2-T1)
print("subspace dimension is:", subspace_dim)