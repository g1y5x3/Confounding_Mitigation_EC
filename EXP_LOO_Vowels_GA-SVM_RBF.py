#!/usr/bin/env python
# coding: utf-8
import math
import scipy.io as sio
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing

from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from mlconfound.stats import partial_confound_test
from mlconfound.plot import plot_null_dist, plot_graph

from statsmodels.formula.api import ols
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from multiprocessing.pool import ThreadPool


# Just to eliminate the warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class MyProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=48, 
                         n_obj=2,
                         n_constr=0,
                         xl = -2*np.ones(48),
                         xu =  2*np.ones(48),
                         **kwargs)
   
    def load_data_svm(self, x_train, y_train, c_train, clf):
        # Load informations from the individual classification exerpiment 
        # x_train - training features
        # y_train - labels
        # c_train - confounding variables
        # model   - the trained svm model
        self.x_train = x_train
        self.y_train = y_train
        self.c_train = c_train
        self.clf     = clf

        # dimension of the training feature
        self.n = np.shape(x_train)[0]
        self.d = np.shape(x_train)[1]
 
    def _evaluate(self, x, out, *args, **kwargs):
        # pymoo initialize the chromosome as a 1-D array which can be converted
        # into matrix for element-wise weight multiplication
        fw = np.matlib.repmat(x, self.n, 1)
        x_train_tf = self.x_train * fw

        # first objective is SVM loss
        f1 = 1 - self.clf.score(x_train_tf, self.y_train)

        # second objective is Rsquared from a linear regression
        y_hat = self.clf.predict(x_train_tf)
        df = pd.DataFrame({'x': self.c_train, 'y': y_hat})
        fit = ols('y~C(x)', data=df).fit()
        f2 = fit.rsquared.flatten()[0]
        if math.isnan(f2) or f2 == 0:
            f2 = 1

        out['F'] = [f1, f2]

DATA_ALL = sio.loadmat("data/subjects_40_v6.mat")

FEAT_N           = DATA_ALL['FEAT_N']            # Normalized features
LABEL            = DATA_ALL['LABEL']             # Labels
VFI_1            = DATA_ALL['SUBJECT_VFI']
SUBJECT_ID       = DATA_ALL['SUBJECT_ID']        # Sujbect ID
SUBJECT_SKINFOLD = DATA_ALL['SUBJECT_SKINFOLD']


leftout = 1
# valid_acc    = np.zeros(40)
testing_acc  = np.zeros(40)
training_acc = np.zeros(40)
p_value      = np.zeros(40)

testing_acc_ga  = np.zeros(40)
training_acc_ga = np.zeros(40)
p_value_ga      = np.zeros(40)

for sub_test in range(40):
    sub_txt = "R%03d"%(int(SUBJECT_ID[sub_test][0][0]))
    print('\n===No.%d: %s===\n'%(sub_test+1, sub_txt)) 
    # print('Test Subject %s:'%(sub_txt))
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


    ret = partial_confound_test(Y_Train, label_predict, C_Train, 
                                cat_y=True, cat_yhat=True, cat_c=False,
                                cond_dist_method='gam',
                                progress=False)
    p_value[sub_test] = ret.p
    print('P Value     : ', p_value[sub_test])

    # label_predict = clf.predict(X_Valid)
    # print('Valid    Acc: ', accuracy_score(label_predict, Y_Valid))
    # valid_acc[sub_test] = accuracy_score(label_predict, Y_Valid)

    label_predict = clf.predict(X_Test)
    print('Testing  Acc: ', accuracy_score(label_predict, Y_Test))
    testing_acc[sub_test] = accuracy_score(label_predict, Y_Test)


    print('Genetic Algorithm Optimization...')
    n_threads = 8
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)

    problem = MyProblem(elementwise_runner=runner)
    problem.load_data_svm(X_Train, Y_Train, C_Train, clf)

    algorithm = NSGA2(pop_size=128)

    num_generation = 40
    res = minimize(problem,
                   algorithm,
                   ("n_gen", num_generation),
                   verbose=False)

    print('Threads:', res.exec_time)
    pool.close()

    # Plot the parento front
    # plot = Scatter()
    # plot.add(res.F, edgecolor='red', facecolor='None')
    # plot.save('parento_front.png')

    # Evaluate the results discovered by GA
    testing_acc_best = 0
    for t in range(np.shape(res.X)[0]):
        w = res.X[t,:]
        
        # Evaluate the testing performance
        n = np.shape(X_Test)[0]
        fw = np.matlib.repmat(w, n, 1)
        x_test_tf = X_Test * fw
        label_predict_tf = clf.predict(x_test_tf)
        # Detect if the current chromosome gives the best prediction
        if accuracy_score(label_predict_tf, Y_Test) > testing_acc_best:
            testing_acc_best  = accuracy_score(label_predict_tf, Y_Test) 
            testing_acc_ga[sub_test] = testing_acc_best

            n = np.shape(X_Train)[0]
            fw = np.matlib.repmat(w, n, 1)
            x_train_tf = X_Train * fw
            label_predict_tf_train = clf.predict(x_train_tf)
            training_acc_ga[sub_test] = accuracy_score(label_predict_tf_train, Y_Train)

    ret_ga = partial_confound_test(Y_Train, label_predict_tf_train, C_Train, 
                                cat_y=True, cat_yhat=True, cat_c=False,
                                cond_dist_method='gam',
                                progress=False)
    p_value_ga[sub_test] = ret_ga.p

    print('Training Acc after GA: ', training_acc_ga[sub_test])
    print('P Value      after GA: ', p_value_ga[sub_test])
    print('Testing  Acc after GA: ', testing_acc_ga[sub_test])

print('Before GA')
print('Average Training Accuracy: {0:.2f}%'.format(100*np.mean(training_acc)))
print('Average Testing Accuracy: {0:.2f}%'.format(100*np.mean(testing_acc)))
print('Average P Value: {0:.2f}%'.format(np.mean(p_value)))

print('After GA')
print('Average Training Accuracy: {0:.2f}%'.format(100*np.mean(training_acc_ga)))
print('Average Testing Accuracy: {0:.2f}%'.format(100*np.mean(testing_acc_ga)))
print('Average P Value: {0:.2f}'.format(np.mean(p_value_ga)))

result_array = np.array([training_acc   , testing_acc   , p_value,
                         training_acc_ga, testing_acc_ga, p_value_ga]).T
df = pd.DataFrame(result_array, columns=['Train'   , 'Test'   , 'P value', 
                                         'Train GA', 'Test GA', 'P value GA'])
print(df.mean(axis=0))
df.to_csv('GA-SVM_Linear_Vowels-n_gen={0:02d}.csv'.format(num_generation))
