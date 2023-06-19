import numpy as np
import pandas as pd 
from scipy.spatial.distance import pdist, squareform
exec(compile(open(r"./nsimplices.py", encoding="utf8").read(), "nsimplices.py", 'exec'))


df_Baron = pd.read_csv("~/nSimplices/data/sce_full_sce_Baron_scScope.csv", index_col=0, header=0)

### Run nSimplices method
feature_num = df_Baron.shape[1]
dim_start = 1
dim_end = df_Baron.shape[1]
print("dim_end is:", dim_end)
out_dis = pdist(df_Baron)
out_dis_sq = squareform(out_dis)

outlier_indices, subspace_dim, corr_dis_sq, corr_coord = nsimplices(out_dis_sq, feature_num, dim_start, dim_end, euc_coord=np.array(df_Baron.copy()))

