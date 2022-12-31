#!/usr/bin/env python
# coding: utf-8

# The data are adapted from \
# https://github.com/geomstats/challenge-iclr-2021/blob/main/Florent-Michel__Shape-Analysis-of-Cancer-Cells/submission_cell_shape_analysis.ipynb

import matplotlib.pyplot as plt

# Running nSimplices

# In[1]:
"""Run nSimplices for cell datasets """
# In[ ]:


# import nSimplices 
# get_ipython().run_line_magic('matplotlib', 'widget')
exec(compile(open(r"nsimplices.py", encoding="utf8").read(), "nsimplices.py", 'exec'))

# set matplotlib default savefig directory
plt.rcParams["savefig.directory"] = os.getcwd() # To save figures to directory
                                                #   defined above

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


""" Importance of dimension correction in higher dimension - Fig.4(A) height distribution """
num_point = target_matrix.shape[0]
hcolls = []
start_dim = 1
end_dim = 40
for dim in range(start_dim, end_dim+1):
    heights = nsimplices_all_heights(num_point, target_matrix, dim, seed=dim+1)
    hcolls.append(heights)


# In[3]:


""" Importance of dimension correction in higher dimension - Fig.4(B) dimensionality inference """

# calculate median heights for tested dimension from start_dim to end_dim
h_meds = []
for hcoll in hcolls:
    h_meds.append(np.median(hcoll))

# calculate the ratio, where h_med_ratios[i] corresponds to h_meds[i-1]/h_meds[i]
# which is the (median height of dim (i-1+start_dim))/(median height of dim (i+start_dim))
h_med_ratios = []
for i in range(1, len(hcolls)):
    h_med_ratios.append(h_meds[i-1]/h_meds[i])

# plot the height scatterplot and the ratios
plt.figure(0)
fig, ax1 = plt.subplots()
color = 'red'
ax1.set_xlabel(r'dimension tested $n$', fontsize=15)
ax1.set_ylabel(r'median of heights', color = color, fontsize=15)
ax1.scatter(list(range(start_dim, end_dim+1)), h_meds, color = color, s=10)
ax1.plot(list(range(start_dim, end_dim+1)), h_meds, color=color)
ax1.tick_params(axis ='y', labelcolor = color)
 
# Adding Twin Axes to plot using dataset_2
ax2 = ax1.twinx()
 
color = 'black'
ax2.set_ylabel(r'heights median ratio: $h_{n-1}/h_n$', color = color, fontsize=15)
ax2.plot(list(range(start_dim+1, end_dim+1)), h_med_ratios, color = color)
ax2.tick_params(axis ='y', labelcolor = color)
plt.tight_layout(pad=2)
 
# Show plot
plt.savefig("./outputs/cells_"+file_id+"_ratio.png")


# In[3]:
""" Read in all cell shapes """

cells_reshaped = np.loadtxt("./data/cells_reshaped.txt")
cells_shape = [650, 100, 2]
cells = cells_reshaped.reshape(
    cells_reshaped.shape[0], cells_reshaped.shape[1] // cells_shape[2], cells_shape[2])


# In[5]:

""" Read in cell indices of a particular group """

cells = np.array(cells)
print(len(cells[0]))
target_indexes = np.loadtxt("data/cells_"+file_id+"_indexes.txt")
target_indexes = [int(x) for x in target_indexes]
print(target_indexes)
target_cells = cells[target_indexes,:,:]
normal_indices=[i for i in range(target_matrix.shape[0]) if i not in outlier_indices] # list of normal points 
print("number of normal cells is:", len(normal_indices))

""" Plot selected outlier cells against normal cells """

plt.figure(2)

nb_cells = len(outlier_indices) # in the main manuscript, only select the first 10 outliers
print("outlier_indices are: ", outlier_indices)

fig = plt.figure(figsize=(15, 8))

invalid_outlier_indices = [0,4,7]

for i in range(nb_cells):
    outlier_idx = outlier_indices[i]
    cell = target_cells[outlier_idx]
    fig.add_subplot(2, nb_cells, i + 1)
    # plt.gca().set_title(i)
    if i in invalid_outlier_indices:
        plt.plot(cell[:, 0], cell[:, 1], color="cornflowerblue")
    else:
        plt.plot(cell[:, 0], cell[:, 1], color="red")
    # if i == nb_cells//2:
    #     plt.title("Outlier cells", fontdict={'fontsize': 20})
    plt.axis('equal')
    plt.axis('off')
    
for i in range(nb_cells):
    normal_idx = normal_indices[i]
    cell = target_cells[normal_idx]
    fig.add_subplot(2, nb_cells, i + nb_cells + 1)
    plt.plot(cell[:, 0], cell[:, 1], color="black")
    # if i == nb_cells//2:
    #     plt.title("Normal cells", fontdict={'fontsize': 20})
    plt.axis('equal')
    plt.axis('off')

plt.savefig("./outputs/cells_"+file_id+"_outlier_normal.png")
plt.close()

""" Plot all outlier cells """
cells_per_row = 10

nb_cells = len(outlier_indices) # in the main manuscript, only select the first 10 outliers
print("outlier_indices are: ", outlier_indices)

fig, axis = plt.subplots(2, cells_per_row)

for i in range(nb_cells):
    row = i // cells_per_row
    col = i % cells_per_row
    outlier_idx = outlier_indices[i]
    cell = target_cells[outlier_idx]
    axis[row, col].plot(cell[:, 0], cell[:, 1], color="cornflowerblue")
    axis[row, col].axis('equal')
    axis[row, col].axis('off')

for i in range(nb_cells, 2*cells_per_row):
    row = i // cells_per_row
    col = i % cells_per_row
    axis[row, col].axis('equal')
    axis[row, col].axis('off')

plt.savefig("./outputs/cells_"+file_id+"_outlier.png")
plt.close()


""" Plot the first 100 normal cells """

fig, axis = plt.subplots(10, cells_per_row)

nb_cells = len(normal_indices) # in the main manuscript, only select the first 10 outliers

for i in range(10*cells_per_row):
    row = i // cells_per_row
    col = i % cells_per_row
    normal_idx = normal_indices[i]
    cell = target_cells[normal_idx]
    axis[row, col].plot(cell[:, 0], cell[:, 1], color="black")
    axis[row, col].axis('equal')
    axis[row, col].axis('off')

plt.savefig("./outputs/cells_"+file_id+"_normal.png")
plt.close()


""" Plot cMDS embedding in 2D using the two largest eigenvalues """
# plot MDS vs MDS + nSimplices (with correct) + MDS in 2D

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))

# plot original graphs with outliers added 
va, ve, Xe = cMDS(target_matrix)
ax1.plot(Xe[normal_indices,0],Xe[normal_indices,1],'.', color='black', label="normal")
ax1.plot(Xe[outlier_indices,0],Xe[outlier_indices,1],'.',color='red', label="outlier")
ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, 
    size=15, weight='bold')
ax1.grid()
ax1.set_xlim([-0.5, 0.7])
ax1.set_ylim([-0.5, 0.5])
ax1.legend()
ax1.set_title("Outliers added")

# plot corrected outliers 
va, ve, Xe = cMDS(corr_dis_sq)   
ax2.plot(Xe[normal_indices,0],Xe[normal_indices,1],'.', color='black', label="normal")
ax2.plot(Xe[outlier_indices,0],Xe[outlier_indices,1],'.',color='red', label="outlier")
ax2.set_title("Corrected data")
ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, 
    size=15, weight='bold')
ax2.grid()
ax2.set_xlim([-0.5, 0.7])
ax2.set_ylim([-0.5, 0.5])
ax2.legend()
plt.savefig("./outputs/cells_"+file_id+"_2D.png")
plt.close()

""" Computes the corrected coordinates after removing the abnormal outliers """
cMDS_fig_path = "outputs/cells_"+file_id+"_cMDS_combined.png"
remove_indices = [42, 134, 203] # detected from cells_control_outlier_normal.png
remove_corr_dis_sq, _ = \
    remove_correct_proj(target_matrix, feature_num, subspace_dim, outlier_indices, remove_indices)
print("target_matrix shape is:", target_matrix.shape)
print("remove_corr_pairwise_dis shape is:", remove_corr_dis_sq.shape)
remove_outlier_indices = update_outlier_index(outlier_indices, remove_indices)
remove_normal_indices=[i for i in range(remove_corr_dis_sq.shape[0]) if i not in remove_outlier_indices] # list of normal points 

fig, (ax1, ax2 ) = plt.subplots(1, 2, figsize=(9,4))
""" Plot cMDS embedding in 2D using the two largest eigenvalues. The corrected \
    distance matrix are obtained by not removing the abnormal outliers """
_, _, corr_Xe = cMDS(corr_dis_sq)
ax1.plot(corr_Xe[normal_indices,0],corr_Xe[normal_indices,1],'.', color='black', label="normal")
ax1.plot(corr_Xe[outlier_indices,0],corr_Xe[outlier_indices,1],'.',color='red', label="outlier")
ax1.set_title("Correted cMDS embedding \n without invalid outliers removed)", fontsize=14)
ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, 
    size=15, weight='bold')
# ax1.axis('equal')
ax1.legend()
# plt.savefig("outputs/cells_"+file_id+"_cMDS_corrected.png")
# plt.close()




""" Plot cMDS embedding in 2D using the two largest eigenvalues. The corrected \
    distance matrix are obtained by removing the abnormal outliers """
# plt.figure(5)
_, _, remove_Xe = cMDS(remove_corr_dis_sq)
ax2.plot(remove_Xe[remove_normal_indices,0],remove_Xe[remove_normal_indices,1],'.', color='black', label="normal")
ax2.plot(remove_Xe[remove_outlier_indices,0],remove_Xe[remove_outlier_indices,1],'.',color='red', label="outlier")
ax2.set_title("Corrected cMDS embedding \n with invalid outliers removed)", fontsize=14)
ax1.text(-0.1, 1.05, 'B', transform=ax2.transAxes, 
    size=15, weight='bold')
ax2.legend()
plt.tight_layout(w_pad=2)
plt.savefig(cMDS_fig_path)
# plt.close()


