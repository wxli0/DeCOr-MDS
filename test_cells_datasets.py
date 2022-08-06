#!/usr/bin/env python
# coding: utf-8

# The data are adapted from \
# https://github.com/geomstats/challenge-iclr-2021/blob/main/Florent-Michel__Shape-Analysis-of-Cancer-Cells/submission_cell_shape_analysis.ipynb

# Running nSimplices

# In[1]:


from nsimplices import *

### Run nSimplices method
feature_num = 40
dim_start = 1
dim_end = feature_num
dunn_control_matrix = np.loadtxt("data/cells_dunn_control_matrix.txt")
print("dunn_control_matrix is:", dunn_control_matrix)
outlier_indices, subspace_dim, corr_dis_sq, corr_coord = nsimplices(dunn_control_matrix, feature_num, dim_start, dim_end)

print("subspace dimension is:", subspace_dim)


# In[2]:


### Importance of dimension correction in higher dimension - Fig.4(A) height distribution 
num_point = dunn_control_matrix.shape[0]
hcolls = []
start_dim = 1
end_dim = 40
for dim in range(start_dim, end_dim+1):
    heights = nsimplices_all_heights(num_point, dunn_control_matrix, dim, seed=dim+1)
    hcolls.append(heights)


# In[3]:


### Importance of dimension correction in higher dimension - Fig.4(B) dimensionality inference

# calculate median heights for tested dimension from start_dim to end_dim
h_meds = []
for hcoll in hcolls:
    h_meds.append(np.median(hcoll))

# calculate the ratio, where h_med_ratios[i] corresponds to h_meds[i-1]/h_meds[i]
# which is the (median height of dim (i-1+start_dim))/(median height of dim (i+start_dim))
h_med_ratios = []
for i in range(1, len(hcolls)):
    # print("dim", start_dim+i-1, "ratio is:", h_meds[i-1]/h_meds[i], h_meds[i-1], h_meds[i])
    h_med_ratios.append(h_meds[i-1]/h_meds[i])

# plot the height scatterplot and the ratios
plt.figure(0)
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel(r'dimension tested $n$')
ax1.set_ylabel(r'median of heights', color = color)
ax1.scatter(list(range(start_dim, end_dim+1)), h_meds, color = color)
ax1.tick_params(axis ='y', labelcolor = color)
 
# Adding Twin Axes to plot using dataset_2
ax2 = ax1.twinx()
 
color = 'tab:green'
ax2.set_ylabel(r'heights median ratio: $h_{n-1}/h_n$', color = color)
ax2.plot(list(range(start_dim+1, end_dim+1)), h_med_ratios, color = color)
ax2.tick_params(axis ='y', labelcolor = color)
 
# Show plot
plt.savefig("./outputs/cells_dim.png")


# In[3]:

cells_reshaped = np.loadtxt("./data/cells_reshaped.txt")
cells_shape = [650, 100, 2]
cells = cells_reshaped.reshape(
    cells_reshaped.shape[0], cells_reshaped.shape[1] // cells_shape[2], cells_shape[2])
# for i in range(len(cells)):

#     line = cells[i].split(" ")
#     print(line)
#     if line == '':
#         continue
#     line = [float(x) for x in line]

print(cells)
# print(np.array(cells).shape)


# In[5]:


nb_cells = 3

cells = np.array(cells)
print(len(cells[0]))
dunn_control_indexes = np.loadtxt("data/cells_dunn_control_indexes.txt")
dunn_control_indexes = [int(x) for x in dunn_control_indexes]
print(dunn_control_indexes)
dunn_control_cells = cells[dunn_control_indexes,:,:]

# fig = plt.figure(figsize=(15, 8))
plt.figure(2)
for i in outlier_indices:
    cell = dunn_control_cells[i]
    # fig.add_subplot(2, nb_cells, i + 1)
    plt.plot(cell[:, 0], cell[:, 1], color = "red")
    np.savetxt("./outputs/cells_tmp_"+str(i)+"_0.txt", cell[:, 0])
    np.savetxt("./outputs/cells_tmp_"+str(i)+"_1.txt", cell[:, 1])

    # if i == nb_cells//2:
    #     plt.title("Outliers")
    plt.axis('equal')
    plt.axis('off')
    
for i in range(0,3):
    cell = dunn_control_cells[i]
    # fig.add_subplot(2, nb_cells, i + nb_cells + 1)
    print("cell 0 is:", cell[:, 0])
    print("cell 1 is:", cell[:, 1])
    plt.plot(cell[:, 0], cell[:, 1], color = "blue")
    # if i == nb_cells//2:
    #     plt.title("Normal dunn control cells")
    plt.axis('equal')
    plt.axis('off')

plt.savefig("./outputs/cells_outlier_normal.png")

### Plot in 2D using the two largest eigenvalues

plt.figure(3)
# plot all cells
va, ve, Xe = MDS(dunn_control_matrix)
normal_indices=[i for i in range(dunn_control_matrix.shape[0]) if i not in outlier_indices] # list of normal points 
plt.plot(Xe[normal_indices,0],Xe[normal_indices,1],'.', color='steelblue')
plt.plot(Xe[outlier_indices,0],Xe[outlier_indices,1],'.',color='red')
plt.title("MDS embedding")
plt.savefig("outputs/cells_MDS.png")