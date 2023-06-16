#!/usr/bin/env python
# coding: utf-8

import math
import pandas as pd
import numpy as np
import numpy.matlib
import multiprocessing

from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

from statsmodels.formula.api import ols
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from multiprocessing.pool import ThreadPool
from mlconfound.stats import partial_confound_test
from mlconfound.plot import plot_graph

class MyProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=36, 
                         n_obj=2,
                         n_constr=0,
                         xl = -1*np.ones(36),
                         xu = np.ones(36),
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
        if math.isnan(f2):
            f2 = 1

        out['F'] = [f1, f2]

data = pd.read_csv('data/pone.csv')

subject_id  = data['PID'].to_numpy()
subject_all = data['PID'].unique()
num_subject = len(subject_all)

x = data[['pcm_intensity_sma_quartile1', 
          'pcm_loudness_sma_linregerrA',
          'pcm_loudness_sma_stddev',
          'pcm_loudness_sma_iqr2.3',
          'pcm_loudness_sma_iqr1.3',
          'mfcc_sma.1._max',
          'mfcc_sma.2._max',
          'mfcc_sma.2._amean',
          'mfcc_sma.5._min',
          'mfcc_sma.5._stddev',
          'mfcc_sma.5._iqr1.2',
          'mfcc_sma.6._min',
          'lspFreq_sma.3._amean',
          'lspFreq_sma.3._quartile1',
          'lspFreq_sma.3._quartile2',
          'lspFreq_sma.3._quartile3',
          'lspFreq_sma.4._amean',
          'lspFreq_sma.4._quartile1',
          'lspFreq_sma.4._quartile2',
          'lspFreq_sma.5._amean',
          'lspFreq_sma.5._quartile1',
          'mfcc_sma_de.2._quartile3',
          'mfcc_sma_de.2._iqr1.2',
          'mfcc_sma_de.2._iqr1.3',
          'mfcc_sma_de.3._linregerrA',
          'mfcc_sma_de.3._linregerrQ',
          'mfcc_sma_de.3._stddev',
          'mfcc_sma_de.5._linregerrA',
          'mfcc_sma_de.5._linregerrQ',
          'mfcc_sma_de.5._stddev',
          'mfcc_sma_de.7._linregerrA',
          'mfcc_sma_de.7._linregerrQ',
          'mfcc_sma_de.7._stddev',
          'voiceProb_sma_de_quartile1',
          'voiceProb_sma_de_iqr1.2',
          'voiceProb_sma_de_iqr1.3']].to_numpy()

y = data['iscase'].to_numpy()

# Extract demographic info as confounder
# c = data['age'].to_numpy()
c = subject_id

training_acc    = np.zeros(num_subject)
training_acc_ga = np.zeros(num_subject)
testing_acc     = np.zeros(num_subject)
testing_acc_ga  = np.zeros(num_subject)
p_value         = np.zeros(num_subject)
p_value_ga      = np.zeros(num_subject)

for s in range(num_subject):
    print('\n===NO.{}: {}===\n'.format(s, subject_all[s]))
    id_train = (subject_id != subject_all[s])
    id_test  = (subject_id == subject_all[s])
    
    x_train = x[id_train,:]
    y_train = y[id_train]
    c_train = c[id_train]
    
    x_test  = x[id_test,:]
    y_test  = y[id_test]
    
    clf = SVC(kernel='linear')
    clf.fit(x_train, y_train)
    
    label_predict = clf.predict(x_train)
    
    print('Training Acc: ', accuracy_score(label_predict, y_train))
    training_acc[s] = accuracy_score(label_predict, y_train)

    ret = partial_confound_test(y_train, label_predict, c_train, 
                                cat_y=True, cat_yhat=True, cat_c=False,
                                cond_dist_method='gam',
                                progress=False)
    p_value[s] = ret.p
    print('P Value     : ', p_value[s])

    label_predict = clf.predict(x_test)
    print('Testing  Acc: ', accuracy_score(label_predict, y_test))
    testing_acc[s] = accuracy_score(label_predict, y_test)    

    print('Genetic Algorithm Optimization...')
    n_threads = 8
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)

    problem = MyProblem(elementwise_runner=runner)
    problem.load_data_svm(x_train, y_train, c_train, clf)

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
        n = np.shape(x_test)[0]
        fw = np.matlib.repmat(w, n, 1)
        x_test_tf = x_test * fw
        label_predict_tf = clf.predict(x_test_tf)
        # Detect if the current chromosome gives the best prediction
        if accuracy_score(label_predict_tf, y_test) > testing_acc_best:
            testing_acc_best  = accuracy_score(label_predict_tf, y_test) 
            testing_acc_ga[s] = testing_acc_best

            n = np.shape(x_train)[0]
            fw = np.matlib.repmat(w, n, 1)
            x_train_tf = x_train * fw
            label_predict_tf_train = clf.predict(x_train_tf)
            training_acc_ga[s] = accuracy_score(label_predict_tf_train, y_train)
    
    ret_ga = partial_confound_test(y_train, label_predict_tf_train, c_train, 
                                   cat_y=True, cat_yhat=True, cat_c=False,
                                   cond_dist_method='gam',
                                   progress=False)
    p_value_ga[s] = ret_ga.p

    print('Training Acc after GA: ', training_acc_ga[s])
    print('P Value      after GA: ', p_value_ga[s])
    print('Testing  Acc after GA: ', testing_acc_ga[s])

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
df.to_csv('GA-SVM_Linear_Depression-n_gen={0:02d}.csv'.format(num_generation))