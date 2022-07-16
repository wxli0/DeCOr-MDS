from nsimplices import nsimplices

import numpy as np

### Run nSimplices method
feature_num = 200
dim_start = 190
dim_end = feature_num
dunn_control_matrix = np.loadtxt("datasets/dunn_control_matrix.txt")
print("dunn_control_matrix is:", dunn_control_matrix)
outlier_indices, subspace_dim, corr_dis_sq, corr_coord = \
    nsimplices(dunn_control_matrix, feature_num, dim_start, dim_end, euc_coord=np.array(dunn_control_matrix.copy()))

print("subspace dimension is:", subspace_dim)