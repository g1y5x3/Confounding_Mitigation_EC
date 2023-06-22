#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.io as sio
import pandas as pd

from mlconfound.stats import partial_confound_test
from mlconfound.plot import plot_null_dist, plot_graph
from IPython.display import Image


# In[2]:

filepath = "CPT_Test_Depression/"

sub_all = np.loadtxt("depression_subject_id.csv", delimiter=",", dtype=int)

df = pd.DataFrame(columns = ['Name','p-value','p-value GA'])


# In[3]:


for i in range(73):
    file = filepath + "%07d"%(sub_all[i])
    print(file)

    data = sio.loadmat(file+".mat")
    # data.keys()

    y_train    = np.ravel(data['y_train'])
    yhat_train = np.ravel(data['yhat_train'])
    c_train    = np.ravel(data['c_train'])

    # Note
    ret = partial_confound_test(y_train, yhat_train, c_train, 
                                cat_y=True, cat_yhat=True, cat_c=False,
                                cond_dist_method='gam',
                                progress=False)  
    graph = plot_graph(ret)
    graph.format = 'png'
    graph.render(filename=file)

    Image(filename=file+'.png') 

    file = filepath + "%07d_GA"%(sub_all[i])
    print(file)

    data = sio.loadmat(file+".mat")
    # data.keys()

    y_train    = np.ravel(data['y_train'])
    yhat_train = np.ravel(data['yhat_train'])
    c_train    = np.ravel(data['c_train'])

    # Note
    ret_GA = partial_confound_test(y_train, yhat_train, c_train, 
                                   cat_y=True, cat_yhat=True, cat_c=False,
                                   cond_dist_method='gam',
                                   progress=False)  
    graph = plot_graph(ret_GA)
    graph.format = 'png'
    graph.render(filename=file)

    Image(filename=file+'.png')

    df = df.append({'Name':'%07d'%(sub_all[i]), 
                    'p-value':ret.p, 
                    'p-value GA':ret_GA.p}, 
                    ignore_index=True)


# In[4]:


df_accuracy = pd.read_csv('CPT_Test_Depression/accuracy.csv', 
                          names=['training','training GA',
                                 'testing','testing GA'])


# In[5]:


df_summary = pd.concat([df, df_accuracy], axis=1)


# In[8]:


print(df_summary.mean(axis=0))


# In[10]:


df_summary.to_csv('CPT_Test_Depression/summary.csv')

