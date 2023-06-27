#!/usr/bin/env python
# coding: utf-8
import math
import wandb
import argparse
import numpy as np
import pandas as pd
import numpy.matlib
import multiprocessing

from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

from statsmodels.formula.api import ols
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display.multi import MultiObjectiveOutput

from multiprocessing.pool import ThreadPool
from mlconfound.stats import partial_confound_test

from ga.fitness import MyProblem, MyCallback

if __name__=="__main__":
    
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

    # Parsing the input arguments

    project_name = 'LOO Voice GA-SVM Linear'

    parser = argparse.ArgumentParser(description="GA-SVM experiments")

    parser.add_argument('-s', type=int, default=0, help="start of the subjects")
    parser.add_argument('-nsub', type=int, default=1, help="number of subjects to be executed")
    parser.add_argument('-ngen', type=int, default=3, help="Number of generation")
    parser.add_argument('-pop', type=int, default=16, help='Population size')
    parser.add_argument('-perm', type=int, default=100, help='Permutation value')
    parser.add_argument('-thread', type=int, default=8, help='Number of threads')
    parser.add_argument('-group', type=str, default='experiment_test', help='Group name')    
    parser.add_argument('-tag', type=str, default='rep_0', help='Tag for marking repetition')    

    args = parser.parse_args()

    # Default value for configurations and parameters that doesn't need
    # to be logged
    config = {"num_generation"  : args.ngen,
              "population_size" : args.pop,
              "permutation"     : args.perm,
              "threads"         : args.thread}
    group_name = args.group
    start_sub  = args.s 
    num_sub    = args.nsub
    tag        = args.tag

    for s in range(start_sub, start_sub + num_sub):
        print('\n===NO.{}: {}===\n'.format(s, subject_all[s]))
        id_train = (subject_id != subject_all[s])
        id_test  = (subject_id == subject_all[s])

        run = wandb.init(project  = project_name,
                         group    = group_name,
                         config   = config,
                         name     = str(id_test),
                         tags     = [tag],
                         settings = wandb.Settings(_disable_stats=True, _disable_meta=True),
                         reinit   = True)
        
        x_train = x[id_train,:]
        y_train = y[id_train]
        c_train = c[id_train]
        
        x_test  = x[id_test,:]
        y_test  = y[id_test]
        
        # Train the classifier
        clf = SVC(kernel='linear')
        clf.fit(x_train, y_train)
        
        label_predict = clf.predict(x_train)
        
        # Evaluate training accuracy
        print('Training Acc: ', accuracy_score(label_predict, y_train))
        training_acc[s] = accuracy_score(label_predict, y_train)

        # Evaluate conditional independence through p value
        ret = partial_confound_test(y_train, label_predict, c_train, 
                                    cat_y=True, cat_yhat=True, cat_c=False,
                                    cond_dist_method='gam',
                                    progress=False)
        p_value[s] = ret.p
        print('P Value     : ', p_value[s])

        # Evaluate Rsquare
        df = pd.DataFrame({'x': c_train, 'y': y_train})
        fit = ols('y~C(x)', data=df).fit()
        rsqrd = fit.rsquared.flatten()[0]

        # Evaluate testing accuracy
        label_predict = clf.predict(x_test)
        print('Testing  Acc: ', accuracy_score(label_predict, y_test))
        testing_acc[s] = accuracy_score(label_predict, y_test)

        wandb.log({"metrics/train_acc": training_acc[s],
                   "metrics/test_acc" : testing_acc[s],
                   "metrics/rsquare"  : rsqrd,
                   "metrics/p_value"  : p_value[s]})    

        print('Genetic Algorithm Optimization...')

        num_permu       = wandb.config["permutation"]
        num_generation  = wandb.config["num_generation"]
        population_size = wandb.config["population_size"]
        threads_count   = wandb.config["threads"]

        n_threads = threads_count
        pool = ThreadPool(n_threads)
        runner = StarmapParallelization(pool.starmap)

        problem = MyProblem(gielementwise_runner=runner,
                            n_var = 36,
                            xl    = -2*np.ones(36),
                            xu    =  2*np.ones(36))
        problem.load_data_svm(x_train, y_train, c_train, clf, num_permu)

        algorithm = NSGA2(pop_size  = population_size,
                          sampling  = FloatRandomSampling(),
                          crossover = SBX(eta=15, prob=0.9),
                          mutation  = PM(eta=20),
                          output    = MultiObjectiveOutput())

        res = minimize(problem,
                       algorithm,
                       ("n_gen", num_generation),
                       callback = MyCallback(),
                       verbose=False)

        print('Threads:', res.exec_time)
        pool.close()

        # Evaluate the results discovered by GA
        Xid = np.argsort(res.F[:,0])
        acc_best = 0
        for t in range(np.shape(res.X)[0]):
            w = res.X[Xid[t],:]
            
            # Evaluate the training performance
            n = np.shape(x_train)[0]
            fw = np.matlib.repmat(w, n, 1)
            x_train_tf = x_train * fw
            y_train_tf = clf.predict(x_train_tf)
            temp_train_acc = clf.score(x_train_tf, y_train)

            # Evaluate the rsquared
            df = pd.DataFrame({'x': c_train, 'y': y_train_tf})
            fit = ols('y~C(x)', data=df).fit()
            temp_rsqrd = fit.rsquared.flatten()[0]

            # Evaluate the p value from the current predicitions
            ret_ga = partial_confound_test(y_train, y_train_tf, c_train, 
                                        cat_y=True, cat_yhat=True, cat_c=False,
                                        cond_dist_method='gam',
                                        progress=False)
            temp_p_value = ret_ga.p

            # Evaluate the testing performance
            n = np.shape(x_test)[0]
            fw = np.matlib.repmat(w, n, 1)
            x_test_tf = x_test * fw
            temp_test_acc = clf.score(x_test_tf, y_test)

            wandb.log({"pareto-front/train_acc": temp_train_acc,
                       "pareto-front/rsquare"  : temp_rsqrd,
                       "pareto-front/p_value"  : temp_p_value,
                       "pareto-front/test_acc" : temp_test_acc})

            # Detect if the current chromosome gives the best prediction
            if temp_test_acc > acc_best:
                acc_best = temp_test_acc 

                training_acc_ga[s] = temp_train_acc 
                p_value_ga[s]      = temp_p_value
                rsqrd_best         = temp_rsqrd
                testing_acc_ga[s]  = temp_test_acc

        print('Training Acc after GA: ', training_acc_ga[s])
        print('P Value      after GA: ', p_value_ga[s])
        print('Testing  Acc after GA: ', testing_acc_ga[s])

        wandb.log({"metrics/train_acc_ga" : training_acc_ga[s],
                   "metrics/test_acc_ga"  : testing_acc_ga[s],
                   "metrics/p_value_ga"   : p_value_ga[s],
                   "metrics/rsquare_ga"   : rsqrd_best})

        run.finish()