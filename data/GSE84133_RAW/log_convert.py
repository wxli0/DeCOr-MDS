#!/usr/bin/env python
# coding: utf-8

# In[1]:


from math import log
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# In[2]:

base_path =  '/home/wanxinli/nSimplices/data/GSE84133_RAW'
file_name = 'GSM2230757_human1_umifm_counts.csv.gz'
save_file_name = 'GSM2230757_human1_umifm_log_counts.csv.gz'

file_path = os.path.join(base_path, file_name)
count_df = pd.read_csv(file_path, header=0, index_col=0)
count_array = np.array(count_df)
count_cell  = np.sum(count_array, axis=1)
print(count_cell)
print(max(count_cell))
plt.hist(count_cell)
plt.show()


# In[3]:


count_df


# In[4]:


def custom_log(count, total_count):
    if count == 0:
        return 0
    return log(count/total_count*(10**4)+1)


# In[5]:


for i in range(count_df.shape[0]):
    print(i)
    for j in range(count_df.shape[1]):
        # print(j)
        count_df.iat[i, j] = custom_log(count_df.iat[i, j], count_cell[i])


# In[65]:


count_df.head()


# In[ ]:





# In[31]:

print("save_file_name is:", save_file_name)
count_df.to_csv(os.path.join(base_path, save_file_name), header=True, index=True)


# 

# In[ ]:




