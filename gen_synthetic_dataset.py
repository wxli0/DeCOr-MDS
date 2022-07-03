#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def col(mean,var,nb): # loc, scale, size
    return pd.DataFrame(np.transpose(np.random.normal(mean,var,nb)))

df_dim3 = pd.concat([col(0,3,200),col(1,2,200), col(10,5,200),col(0,0,200),col(3,0.1,200),col(10,0.2,200),col(0,0.05,200)],axis=1)

df_dim2 = pd.concat([col(1,2,200), col(10,5,200),col(0,0,200),col(3,0.0003,200),col(10,0.0001,200),col(0,0.0001,200),col(2,0.0002,200)],axis=1) #

df_dim2point5 = pd.concat([col(1,5,200), col(10,5,200),col(2,0,200),col(2,0.01,200),col(2,0.01,200),col(2,0.02,200)],axis=1) #,col(3,0.1,200),col(10,0.2,200),col(0,0.05,200),col(2,0.1,200)

df_dim4 = pd.concat([col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),col(2,0.01,200),col(2,0.02,200),col(2,0.02,200)],axis=1)
df_dim5 = pd.concat([col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),col(7,6,200),col(2,0.01,200),col(2,0.02,200),col(2,0.02,200)],axis=1)
df_dim6 = pd.concat([col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),col(7,6,200),col(7,6,200),col(2,0.01,200),col(2,0.02,200),col(2,0.02,200)],axis=1)

N = 200
# df_dim10 = pd.concat([col(1,5,N), col(10,5,N),col(2,5,N),col(-2,4,N),col(7,6,N),col(7,6,N),col(1,5,N), col(10,5,N),col(2,5,N),col(-2,4,N),\
#     col(0,0.0001,N),col(0,0.0002,N),col(0,0.0003,N),col(0,0.0002,N),col(0,0.0001,N),col(0,0.0002,N),col(0,0.0003,N),col(0,0.0002,N),col(0,0.0001,N),col(0,0.0001,N)],axis=1)

df_dim10 = pd.concat([col(1,5,N), col(10,5,N),col(2,5,N),col(-2,4,N),col(7,6,N),col(7,6,N),col(1,5,N), col(10,5,N),col(2,5,N),col(-2,4,N),\
    col(0,0.0001,N), col(0,0.0001,N), col(2,0.0001,N)],axis=1)

df_dim20 = pd.concat([col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),col(7,6,200),col(7,6,200),col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),
    col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),col(7,6,200),col(7,6,200),col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),
    col(0,0.01,200),col(0,0.02,200),col(0,0.02,200),col(0,0.01,200),col(0,0.02,200),col(0,0.02,200),col(0,0.01,200),col(0,0.02,200),col(0,0.02,200),col(0,0.01,200)],axis=1)



df_dim30 = pd.concat([col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),col(7,6,200),col(7,6,200),col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),col(7,6,200),col(7,6,200),col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),col(7,6,200),col(7,6,200),col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),col(2,0.01,200),col(2,0.02,200),col(2,0.02,200),col(2,0.01,200),col(2,0.02,200),col(2,0.02,200),col(2,0.01,200),col(2,0.02,200),col(2,0.02,200),col(2,0.01,200),col(2,0.02,200),col(2,0.02,200),col(2,0.01,200),col(2,0.02,200),col(2,0.02,200),col(2,0.01,200),col(2,0.02,200),col(2,0.02,200)],axis=1)

df_dim40 = pd.concat([col(1,5,N), col(10,5,N),col(2,5,N),col(-2,4,N),col(7,6,N),col(7,6,N),col(1,5,N), col(10,5,N),col(2,5,N),col(-2,4,N),\
    col(1,5,N), col(10,5,N),col(2,5,N),col(-2,4,N),col(7,6,N),col(7,6,N),col(1,5,N), col(10,5,N),col(2,5,N),col(-2,4,N),\
    col(1,5,N), col(10,5,N),col(2,5,N),col(-2,4,N),col(7,6,N),col(7,6,N),col(1,5,N), col(10,5,N),col(2,5,N),col(-2,4,N),\
    col(1,5,N), col(10,5,N),col(2,5,N),col(-2,4,N),col(7,6,N),col(7,6,N),col(1,5,N), col(10,5,N),col(2,5,N),col(-2,4,N),\
    col(0,0.0001,N), col(0,0.0001,N), col(5,0.0001,N), col(0,0.0001,N), col(0,0.0001,N), col(2,0.0001,N)],axis=1)

df_dim40_2 = pd.concat([col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),col(7,6,200),col(7,6,200),col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),
    col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),col(7,6,200),col(7,6,200),col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),
    col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),col(7,6,200),col(7,6,200),col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),
    col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),col(7,6,200),col(7,6,200),col(1,5,200), col(10,5,200),col(2,5,200),col(-2,4,200),
    col(2,0,200),col(2,0,200),col(2,0,200),col(2,0,200),col(2,0,200),col(2,0,200),col(2,0,200),col(2,0,200),col(2,0,200)],axis=1)

df_noise_influence0=pd.concat([col(0,0.05,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200)],axis=1)
df_noise_influence1=pd.concat([col(0,0.05,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,0.05,200)],axis=1)
df_noise_influence5=pd.concat([col(0,0.05,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200)],axis=1)
df_noise_influence10=pd.concat([col(0,0.05,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200)],axis=1)
df_noise_influence50=pd.concat([col(0,0.05,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200),col(0,0.05,200)],axis=1)


df_noise2_influence2=pd.concat([col(0,0.05,200),col(0,5,200),col(0,5,200)],axis=1)
df_noise2_influence4=pd.concat([col(0,0.05,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200)],axis=1)
df_noise2_influence8=pd.concat([col(0,0.05,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200)],axis=1)
df_noise2_influence16=pd.concat([col(0,0.05,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200)],axis=1)
df_noise2_influence32=pd.concat([col(0,0.05,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200),col(0,5,200)],axis=1)

tab_dim2=np.array(df_dim2)
tab_dim3=np.array(df_dim3)
tab_dim2point5=np.array(df_dim2point5)
tab_dim4=np.array(df_dim4)
tab_dim5=np.array(df_dim5)
tab_dim6=np.array(df_dim6)
tab_dim10=np.array(df_dim10)
tab_dim20=np.array(df_dim20)
tab_dim30=np.array(df_dim30)
tab_dim40=np.array(df_dim40)
tab_dim40_2=np.array(df_dim40_2)

noise0=np.array(df_noise_influence0)
noise1=np.array(df_noise_influence1)
noise5=np.array(df_noise_influence5)
noise10=np.array(df_noise_influence10)
noise50=np.array(df_noise_influence50)

nois2=np.array(df_noise2_influence2)
nois4=np.array(df_noise2_influence4)
nois8=np.array(df_noise2_influence8)
nois16=np.array(df_noise2_influence16)
nois32=np.array(df_noise2_influence32)


np.savetxt(r'outputs/bdd_synthetic_rdim2.csv',tab_dim2,delimiter=';')

np.savetxt(r'outputs/bdd_synthetic_rdim3.csv',tab_dim3,delimiter=';')

np.savetxt(r'outputs/bdd_synthetic_rdim2point5.csv',tab_dim2point5,delimiter=';')

np.savetxt(r'outputs/bdd_synthetic_rdim4.csv',tab_dim4,delimiter=';')

np.savetxt(r'outputs/bdd_synthetic_rdim5.csv',tab_dim5,delimiter=';')

np.savetxt(r'outputs/bdd_synthetic_rdim6.csv',tab_dim6,delimiter=';')

np.savetxt(r'outputs/bdd_synthetic_rdim10.csv',tab_dim10,delimiter=';')

np.savetxt(r'outputs/bdd_synthetic_rdim30.csv',tab_dim30,delimiter=';')

np.savetxt(r'outputs/bdd_synthetic_rdim20.csv',tab_dim20,delimiter=';')

np.savetxt(r'outputs/bdd_synthetic_rdim40.csv',tab_dim40,delimiter=';')

np.savetxt(r'outputs/bdd_synthetic_rdim40_2.csv',tab_dim40_2,delimiter=';')

np.savetxt(r'outputs/noise0.csv',noise0,delimiter=';')
np.savetxt(r'outputs/noise1.csv',noise1,delimiter=';')
np.savetxt(r'outputs/noise5.csv',noise5,delimiter=';')
np.savetxt(r'outputs/noise10.csv',noise10,delimiter=';')
np.savetxt(r'outputs/noise50.csv',noise50,delimiter=';')

np.savetxt(r'outputs/nois2.csv',nois2,delimiter=';')
np.savetxt(r'outputs/nois4.csv',nois4,delimiter=';')
np.savetxt(r'outputs/nois8.csv',nois8,delimiter=';')
np.savetxt(r'outputs/nois16.csv',nois16,delimiter=';')
np.savetxt(r'outputs/nois32.csv',nois32,delimiter=';')
