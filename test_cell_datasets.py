from nsimplices import *


### Run nSimplices method
feature_num = 40
dim_start = 1
dim_end = feature_num
dunn_control_matrix = np.loadtxt("datasets/dunn_control_matrix.txt")
print("dunn_control_matrix is:", dunn_control_matrix)
outlier_indices, subspace_dim, corr_dis_sq, corr_coord = \
    nsimplices(dunn_control_matrix, feature_num, dim_start, dim_end)

print("subspace dimension is:", subspace_dim)

### Importance of dimension correction in higher dimension - Fig.4(A) height distribution 
num_point = dunn_control_matrix.shape[0]
hcolls = []
start_dim = 2
end_dim = 15
for dim in range(start_dim, end_dim+1):
    heights = nsimplices_all_heights(num_point, dunn_control_matrix, dim, seed=dim+1)
    hcolls.append(heights)
