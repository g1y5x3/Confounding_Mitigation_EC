#!/usr/bin/env python
# coding: utf-8
import argparse, wandb
import numpy as np
import scipy.io as sio

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    DATA_ALL = sio.loadmat("data/subjects_40_vowels_v6.mat")
    # DATA_ALL = sio.loadmat("data/subjects_40_sen_slide_win1.0_overlap0.5.mat")
    # DATA_ALL = sio.loadmat("data/subjects_40_sen_fix_win1.0.mat")
    DATA_PARTIAL = sio.loadmat("data/subjects_40_vowels_v6.mat")

    FEAT_N           = DATA_ALL['FEAT_N']            # Normalized features
    LABEL            = DATA_ALL['LABEL']             # Labels
    VFI_1            = DATA_PARTIAL['SUBJECT_VFI']
    SUBJECT_ID       = DATA_PARTIAL['SUBJECT_ID']        # Sujbect ID
    SUBJECT_SKINFOLD = DATA_PARTIAL['SUBJECT_SKINFOLD']

    leftout = 1
    # valid_acc    = np.zeros(40)
    testing_acc  = np.zeros(40)
    training_acc = np.zeros(40)
    p_value      = np.zeros(40)

    testing_acc_ga  = np.zeros(40)
    training_acc_ga = np.zeros(40)
    p_value_ga      = np.zeros(40)

    project_name = 'LOO-Sentence-Classification'

    parser = argparse.ArgumentParser(description="SVM experiments")

    parser.add_argument('-s', type=int, default=0, help="start of the subjects")
    parser.add_argument('-nsub', type=int, default=1, help="number of subjects to be executed")
    parser.add_argument('-group', type=str, default='Vowels', help='Group name')

    args = parser.parse_args()

    # Default value for configurations and parameters that doesn't need
    # to be logged
    group_name = args.group
    start_sub  = args.s
    num_sub    = args.nsub

    for sub_test in range(start_sub, start_sub + num_sub):

        sub_txt = "R%03d"%(int(SUBJECT_ID[sub_test][0][0]))
        if int(VFI_1[sub_test][0][0]) > 10:
            sub_group = 'Fatigued'
        else:
            sub_group = 'Healthy'

        run = wandb.init(project  = project_name,
                         entity   = "vocalwell-vigir",
                         group    = group_name,
                         name     = sub_txt,
                         tags     = [sub_group, 'SVM'],
                         settings = wandb.Settings(_disable_stats=True, _disable_meta=True),
                         reinit   = True)

        print('\n===No.%d: %s===\n'%(sub_test+1, sub_txt))
        print('VFI-1:', (VFI_1[sub_test][0][0]))

        wandb.log({"subject_info/vfi_1"  : int(VFI_1[sub_test][0][0])})

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
        for sub_train in range(40):
            if sub_train != sub_test:
                x_s = FEAT_N[sub_train,0]
                y_s = LABEL[sub_train,0].flatten()
                c_s = np.mean(np.mean(SUBJECT_SKINFOLD[sub_train,:]), axis=1)
                X_TV = np.concatenate((X_TV, x_s), axis=0)
                Y_TV = np.concatenate((Y_TV, y_s), axis=0)

        print('# of Healthy Samples: %d'%(np.sum(Y_TV == -1)))
        print('# of Fatigued Samples: %d'%(np.sum(Y_TV == 1)))

        wandb.log({"exp_info/healthy_samples" : np.sum(Y_TV == -1),
                   "exp_info/fatigued_samples": np.sum(Y_TV ==  1),
                   "exp_info/total_samples"   : np.sum(Y_TV == -1) + np.sum(Y_TV ==  1)})

        # ===== Data loading and preprocessing =====
        # Training and Validation
        # NEED TO REMOVE THE VALIDATION DATA SINCE THEY ARE NOT BEING USED
        X_Train, X_Valid, Y_Train, Y_Valid = train_test_split(X_TV,
                                                              Y_TV,
                                                              test_size=0.1,
                                                              random_state=42)

        clf = SVC(C=1.0, gamma='scale', kernel='rbf', class_weight='balanced', max_iter=1000, tol=0.001)
        clf.fit(X_Train, Y_Train)

        label_predict = clf.predict(X_Train)

        print('Training Acc: ', accuracy_score(label_predict, Y_Train))
        training_acc[sub_test] = accuracy_score(label_predict, Y_Train)

        label_predict = clf.predict(X_Test)
        print('Testing  Acc: ', accuracy_score(label_predict, Y_Test))
        testing_acc[sub_test] = accuracy_score(label_predict, Y_Test)

        wandb.log({"metrics/train_acc" : training_acc[sub_test],
                   "metrics/test_acc"  : testing_acc[sub_test]})

        run.finish()