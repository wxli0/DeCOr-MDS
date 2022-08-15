#!/usr/bin/env python
# coding: utf-8

# The data are adapted from \
# https://github.com/geomstats/challenge-iclr-2021/blob/main/Florent-Michel__Shape-Analysis-of-Cancer-Cells/submission_cell_shape_analysis.ipynb

# Running nSimplices

# In[1]:
"""Run nSimplices for cell datasets """

from nsimplices import *

### Run nSimplices method
feature_num = 40
dim_start = 1
dim_end = feature_num
file_id = "control"
target_matrix = np.loadtxt("data/cells_"+file_id+"_matrix.txt")
print("target_matrix is:", target_matrix)
outlier_indices, subspace_dim, corr_dis_sq, corr_coord = nsimplices(target_matrix, feature_num, dim_start, dim_end)

print("subspace dimension is:", subspace_dim)
print("outlier_indices is:", outlier_indices)
print("outlier_indices len is:", len(outlier_indices))

# # In[2]:


# """ Importance of dimension correction in higher dimension - Fig.4(A) height distribution """
# num_point = target_matrix.shape[0]
# hcolls = []
# start_dim = 1
# end_dim = 40
# for dim in range(start_dim, end_dim+1):
#     heights = nsimplices_all_heights(num_point, target_matrix, dim, seed=dim+1)
#     hcolls.append(heights)


# # In[3]:


# """ Importance of dimension correction in higher dimension - Fig.4(B) dimensionality inference """

# # calculate median heights for tested dimension from start_dim to end_dim
# h_meds = []
# for hcoll in hcolls:
#     h_meds.append(np.median(hcoll))

# # calculate the ratio, where h_med_ratios[i] corresponds to h_meds[i-1]/h_meds[i]
# # which is the (median height of dim (i-1+start_dim))/(median height of dim (i+start_dim))
# h_med_ratios = []
# for i in range(1, len(hcolls)):
#     h_med_ratios.append(h_meds[i-1]/h_meds[i])

# # plot the height scatterplot and the ratios
# plt.figure(0)
# fig, ax1 = plt.subplots()
# color = 'tab:red'
# ax1.set_xlabel(r'dimension tested $n$')
# ax1.set_ylabel(r'median of heights', color = color)
# ax1.scatter(list(range(start_dim, end_dim+1)), h_meds, color = color)
# ax1.tick_params(axis ='y', labelcolor = color)
 
# # Adding Twin Axes to plot using dataset_2
# ax2 = ax1.twinx()
 
# color = 'tab:green'
# ax2.set_ylabel(r'heights median ratio: $h_{n-1}/h_n$', color = color)
# ax2.plot(list(range(start_dim+1, end_dim+1)), h_med_ratios, color = color)
# ax2.tick_params(axis ='y', labelcolor = color)
 
# # Show plot
# plt.savefig("./outputs/cells_"+file_id+"_dim.png")


# # In[3]:
# """ Read in all cell shapes """

# cells_reshaped = np.loadtxt("./data/cells_reshaped.txt")
# cells_shape = [650, 100, 2]
# cells = cells_reshaped.reshape(
#     cells_reshaped.shape[0], cells_reshaped.shape[1] // cells_shape[2], cells_shape[2])


# # In[5]:

# """ Read in cell indices of a particular group """

# cells = np.array(cells)
# print(len(cells[0]))
# target_indexes = np.loadtxt("data/cells_"+file_id+"_indexes.txt")
# target_indexes = [int(x) for x in target_indexes]
# print(target_indexes)
# target_cells = cells[target_indexes,:,:]
normal_indices=[i for i in range(target_matrix.shape[0]) if i not in outlier_indices] # list of normal points 

# """ Plot outlier cells against normal cells """

# plt.figure(2)

# nb_cells = len(outlier_indices)
# print("outlier_indices are: ", outlier_indices)

# fig = plt.figure(figsize=(15, 8))

# for i in range(nb_cells):
#     outlier_idx = outlier_indices[i]
#     cell = target_cells[outlier_idx]
#     fig.add_subplot(2, nb_cells, i + 1)
#     plt.plot(cell[:, 0], cell[:, 1], color="red")
#     if i == nb_cells//2:
#         plt.title("Outlier cells")
#     plt.axis('equal')
#     plt.axis('off')
    
# for i in range(nb_cells):
#     normal_idx = normal_indices[i]
#     cell = target_cells[normal_idx]
#     fig.add_subplot(2, nb_cells, i + nb_cells + 1)
#     plt.plot(cell[:, 0], cell[:, 1], color="blue")
#     if i == nb_cells//2:
#         plt.title("normal cells")
#     plt.axis('equal')
#     plt.axis('off')

# plt.savefig("./outputs/cells_"+file_id+"_outlier_normal.png")
# plt.close()

# """ Plot MDS embedding in 2D using the two largest eigenvalues """

# plt.figure(3)
# blue_outlier_idx = -1
# _, _, Xe = MDS(target_matrix)
# plt.plot(Xe[normal_indices,0],Xe[normal_indices,1],'.', color='steelblue')
# plt.plot(Xe[outlier_indices,0],Xe[outlier_indices,1],'.',color='red')
# for i in range(Xe.shape[0]):
#     if 0.6 < Xe[i, 0] and Xe[i, 0] < 0.7:
#         blue_outlier_idx = i
# plt.title("MDS embedding")
# plt.savefig("outputs/cells_"+file_id+"_MDS.png")
# plt.close()

# fig = plt.figure(4)
# plt.plot(cell[:, 0], cell[:, 1], color="blue")
# plt.axis('equal')
# plt.axis('off')
# plt.title('blue MDS outlier but not detected')
# plt.savefig('./outputs/cells_'+str(blue_outlier_idx)+".png")

""" Computes the corrected coordinates after removing the abnormal outliers """
remove_indices = [42, 134, 203] # detected from cells_control_outlier_normal.png
remove_corr_dis_sq, _ = \
    remove_correct_proj(target_matrix, feature_num, subspace_dim, outlier_indices, remove_indices)
print("target_matrix shape is:", target_matrix.shape)
print("remove_corr_pairwise_dis shape is:", remove_corr_dis_sq.shape)
remove_outlier_indices = update_outlier_index(outlier_indices, remove_indices)
remove_normal_indices=[i for i in range(remove_corr_dis_sq.shape[0]) if i not in remove_outlier_indices] # list of normal points 

""" Plot MDS embedding in 2D using the two largest eigenvalues. The corrected \
    distance matrix are obtained by not removing the abnormal outliers """
plt.figure(5)
_, _, corr_Xe = MDS(corr_dis_sq)
plt.plot(corr_Xe[normal_indices,0],corr_Xe[normal_indices,1],'.', color='steelblue')
plt.plot(corr_Xe[outlier_indices,0],corr_Xe[outlier_indices,1],'.',color='red')
plt.title("MDS embedding (corrected and without abnormal outliers removed)")
plt.savefig("outputs/cells_"+file_id+"_MDS_corrected.png")
plt.close()

""" Plot MDS embedding in 2D using the two largest eigenvalues. The corrected \
    distance matrix are obtained by removing the abnormal outliers """
plt.figure(5)
_, _, remove_Xe = MDS(remove_corr_dis_sq)
plt.plot(remove_Xe[remove_normal_indices,0],remove_Xe[remove_normal_indices,1],'.', color='steelblue')
plt.plot(remove_Xe[remove_outlier_indices,0],remove_Xe[remove_outlier_indices,1],'.',color='red')
plt.title("MDS embedding (corrected and with abnormal outliers removed)")
plt.savefig("outputs/cells_"+file_id+"_MDS_removed_corrected.png")
plt.close()


