#!/usr/bin/env python
# coding: utf-8
import pygad
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from mlconfound.stats import partial_confound_test
from mlconfound.plot import plot_null_dist, plot_graph

# Just to eliminate the warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

DATA_ALL = sio.loadmat("data/subjects_40_v6.mat")

FEAT_N           = DATA_ALL['FEAT_N']            # Normalized features
LABEL            = DATA_ALL['LABEL']             # Labels
VFI_1            = DATA_ALL['SUBJECT_VFI']
SUBJECT_ID       = DATA_ALL['SUBJECT_ID']        # Sujbect ID
SUBJECT_SKINFOLD = DATA_ALL['SUBJECT_SKINFOLD']


leftout = 1
testing_acc  = np.zeros(40)
valid_acc    = np.zeros(40)
training_acc = np.zeros(40)
p_value      = np.zeros(40)


for sub_test in range(40):
    print('\n===Exp No. %d===\n'%(sub_test+1))
    
    sub_txt = "R%03d"%(int(SUBJECT_ID[sub_test][0][0]))
    print('Test Subject %s:'%(sub_txt))
    print('VFI-1:', (VFI_1[sub_test][0][0]))
    if int(VFI_1[sub_test][0][0]) > 10:
        sub_group = 'Fatigued'
    else:
        sub_group = 'Healthy'

    # ===== Load Testing Signals =====
    num_signal = np.shape(FEAT_N[sub_test,0])[0]    
    X_Temp = FEAT_N[sub_test,0]
    Y_Temp = LABEL[sub_test,0].flatten()

    num_leftout = round(leftout*num_signal)
    index_leftout = np.random.choice(range(num_signal), 
                                     size=num_leftout, 
                                     replace=False)
    print("Left-out Test samples: ", index_leftout.size)

    X_Test = X_Temp[index_leftout,:]
    Y_Test = Y_Temp[index_leftout]

    index_include = np.arange(num_signal)
    index_include = np.delete(index_include, index_leftout)
    print("Included Training samples: ", index_include.size)
    X_include = X_Temp[index_include,:]
    Y_include = Y_Temp[index_include]


    # ===== Load Traing Signals =====
    X_TV = np.zeros((0,48))
    Y_TV = np.zeros(0)    
    C_TV = np.zeros(0)
    for sub_train in range(40):
        if sub_train != sub_test:
            x_s = FEAT_N[sub_train,0]
            y_s = LABEL[sub_train,0].flatten()
            c_s = np.mean(np.mean(SUBJECT_SKINFOLD[sub_train,:]), axis=1)
            # ===== CAN BE CONVERTED INTO A FUNCTION =====
            X_TV = np.concatenate((X_TV, x_s), axis=0)
            Y_TV = np.concatenate((Y_TV, y_s), axis=0)
            C_TV = np.concatenate((C_TV, c_s), axis=0)       

    print('# of Healthy Samples: %d'%(np.sum(Y_TV == -1)))
    print('# of Fatigued Samples: %d'%(np.sum(Y_TV == 1)))    

    # ===== Data loading and preprocessing =====
    # Training and Validation
    X_Train, X_Valid, YC_Train, YC_Valid = train_test_split(X_TV, 
                                                            np.transpose([Y_TV, C_TV]), 
                                                            test_size=0.1, 
                                                            random_state=42)
    Y_Train, C_Train = YC_Train[:,0], YC_Train[:,1]
    Y_Valid, C_Valid = YC_Valid[:,0], YC_Valid[:,1]    
    
    clf = SVC(C=1.0, gamma='scale', kernel='rbf', class_weight='balanced', max_iter=1000, tol=0.001)
    clf.fit(X_Train, Y_Train)
    
    label_predict = clf.predict(X_Train)
    
    print('Training Acc: ', accuracy_score(label_predict, Y_Train))
    training_acc[sub_test] = accuracy_score(label_predict, Y_Train)

    label_predict = clf.predict(X_Valid)
    print('Validation Acc: ', accuracy_score(label_predict, Y_Valid))
    valid_acc[sub_test] = accuracy_score(label_predict, Y_Valid)

    label_predict = clf.predict(X_Test)
    print('Testing Acc: ', accuracy_score(label_predict, Y_Test))
    testing_acc[sub_test] = accuracy_score(label_predict, Y_Test)



data_array = np.array([training_acc, valid_acc, testing_acc]).T
df = pd.DataFrame(data_array, columns = ['Train', 'Valid', 'Test'])

print(df.mean(axis=0))
df.to_csv('results/RBF_SVM_LOO.csv')
