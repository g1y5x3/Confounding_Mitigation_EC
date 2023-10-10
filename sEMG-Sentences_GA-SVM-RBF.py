#!/usr/bin/env python
# coding: utf-8
import argparse
import numpy.matlib
import multiprocessing
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from mlconfound.plot import plot_null_dist, plot_graph
from mlconfound.stats import partial_confound_test

from statsmodels.formula.api import ols

from multiprocessing.pool import ThreadPool

from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display.multi import MultiObjectiveOutput

from GA.fitness_log import MyProblem, MyCallback

# Just to eliminate the warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import wandb

if __name__ == "__main__":

    project_name = 'LOO-Sentence-Classification'

    parser = argparse.ArgumentParser(description="GA-SVM experiments")

    parser.add_argument('-win',    type=str, default='slide')
    parser.add_argument('-wandb',  type=int, default=0)
    parser.add_argument('-start',  type=int, default=0, help="start of the subjects")
    parser.add_argument('-nsub',   type=int, default=1, help="number of subjects to be executed")
    parser.add_argument('-ngen',   type=int, default=2, help="Number of generation")
    parser.add_argument('-pop',    type=int, default=16, help='Population size')
    parser.add_argument('-perm',   type=int, default=100, help='Permutation value')
    parser.add_argument('-thread', type=int, default=8, help='Number of threads')
    parser.add_argument('-group',  type=str, default='GA-SVM', help='Group name')

    args = parser.parse_args()

    # Default value for configurations and parameters that doesn't need
    # to be logged
    config = {"num_generation"  : args.ngen,
              "population_size" : args.pop,
              "permutation"     : args.perm,
              "threads"         : args.thread}
    
    group_name = args.group
    start_sub  = args.start 
    num_sub    = args.nsub

    if args.win == 'fix':
        DATA_ALL = sio.loadmat("data/subjects_40_sen_fix_win1.0.mat")
    else:
        DATA_ALL = sio.loadmat("data/subjects_40_sen_slide_win1.0_overlap0.5.mat")
    SUBJECT_INFO = pd.read_csv("data/subjects_40_info.csv")

    FEAT_N = DATA_ALL['FEAT_N']            # Normalized features
    LABEL  = DATA_ALL['LABEL']             # Labels

    leftout = 1
    testing_acc  = np.zeros(40)
    training_acc = np.zeros(40)
    p_value      = np.zeros(40)

    testing_acc_ga  = np.zeros(40)
    training_acc_ga = np.zeros(40)
    p_value_ga      = np.zeros(40)

    for sub_test in range(start_sub, start_sub + num_sub):

        sub_txt = "R%03d"%(int(SUBJECT_INFO.ID[sub_test]))
        sub_vfi = int(SUBJECT_INFO.VFI[sub_test])
        if sub_vfi > 10:
            sub_group = 'Fatigued'
        else:
            sub_group = 'Healthy'

        if args.wandb:
            run = wandb.init(project  = project_name,
                             group    = group_name,
                             config   = config,
                             name     = sub_txt,
                             tags     = [sub_group],
                             settings = wandb.Settings(_disable_stats=True, _disable_meta=True),
                             reinit   = True)

        print('\n===No.%d: %s==='%(sub_test+1, sub_txt))
        print('VFI-1:', sub_vfi)

        if args.wandb:
            wandb.log({"subject_info/vfi_1"  : sub_vfi})

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
                n = x_s.shape[0]
                # print(np.mean(np.array([SUBJECT_INFO.SKINFOLD_SUPRA[sub_train],SUBJECT_INFO.SKINFOLD_INFRA[sub_train]])))
                c_s = np.repeat(np.mean(np.array([SUBJECT_INFO.SKINFOLD_SUPRA[sub_train],SUBJECT_INFO.SKINFOLD_INFRA[sub_train]])), n, axis=0)
                X_TV = np.concatenate((X_TV, x_s), axis=0)
                Y_TV = np.concatenate((Y_TV, y_s), axis=0)
                C_TV = np.concatenate((C_TV, c_s), axis=0)
                # print(X_TV.shape, Y_TV.shape, C_TV.shape)

        print('# of Healthy Samples: %d'%(np.sum(Y_TV == -1)))
        print('# of Fatigued Samples: %d'%(np.sum(Y_TV == 1)))

        if args.wandb:
            wandb.log({"exp_info/healthy_samples" : np.sum(Y_TV == -1),
                       "exp_info/fatigued_samples": np.sum(Y_TV ==  1),
                       "exp_info/total_samples"   : np.sum(Y_TV == -1) + np.sum(Y_TV ==  1)})

        # ===== Data loading and preprocessing =====
        # Training and Validation
        # NEED TO REMOVE THE VALIDATION DATA SINCE THEY ARE NOT BEING USED
        X_Train, X_Valid, YC_Train, YC_Valid = train_test_split(X_TV,
                                                                np.transpose([Y_TV, C_TV]),
                                                                test_size=0.1,
                                                                random_state=42)
        Y_Train, C_Train = YC_Train[:,0], YC_Train[:,1]
        Y_Valid, C_Valid = YC_Valid[:,0], YC_Valid[:,1]

        clf = SVC(C=1.0, gamma='scale', kernel='rbf', class_weight='balanced', max_iter=1000, tol=0.001)
        clf.fit(X_Train, Y_Train)
        label_predict = clf.predict(X_Train)

        training_acc[sub_test] = accuracy_score(label_predict, Y_Train)
        print('Training Acc: ', training_acc[sub_test])

        ret = partial_confound_test(Y_Train, label_predict, C_Train,
                                    cat_y=True, cat_yhat=True, cat_c=False,
                                    cond_dist_method='gam',
                                    progress=False)
        p_value[sub_test] = ret.p
        print('P Value     : ', p_value[sub_test])

        # Evalute rsquared
        df = pd.DataFrame({'x': C_Train, 'y': Y_Train})
        fit = ols('y~C(x)', data=df).fit()
        rsqrd = fit.rsquared.flatten()[0]

        label_predict = clf.predict(X_Test)
        testing_acc[sub_test] = accuracy_score(label_predict, Y_Test)
        print('Testing  Acc: ', testing_acc[sub_test])

        if args.wandb:
            wandb.log({"metrics/train_acc" : training_acc[sub_test],
                       "metrics/test_acc"  : testing_acc[sub_test],
                       "metrics/rsquare"   : rsqrd,
                       "metrics/p_value"   : p_value[sub_test]})

        print('Genetic Algorithm Optimization...')

        if args.wandb:
            num_permu       = wandb.config["permutation"]
            num_generation  = wandb.config["num_generation"]
            population_size = wandb.config["population_size"]
            threads_count   = wandb.config["threads"]
        else:
            num_permu       = config["permutation"]
            num_generation  = config["num_generation"]
            population_size = config["population_size"]
            threads_count   = config["threads"]

        n_threads = threads_count
        pool = ThreadPool(n_threads)
        runner = StarmapParallelization(pool.starmap)

        problem = MyProblem(elementwise_runner=runner)
        problem.load_param(X_Train, Y_Train, C_Train, X_Test, Y_Test, clf, num_permu)

        # Genetic algorithm initialization
        algorithm = NSGA2(pop_size  = population_size,
                          sampling  = FloatRandomSampling(),
                          crossover = SBX(eta=15, prob=0.9),
                          mutation  = PM(eta=20),
                          output    = MultiObjectiveOutput())

        res = minimize(problem,
                       algorithm,
                       ("n_gen", num_generation),
                       callback = MyCallback(args.wandb),
                       verbose=False)

        print('Threads:', res.exec_time)
        pool.close()
        training_acc_ga[sub_test] = res.algorithm.callback.data["train_acc"][-1]
        p_value_ga[sub_test] = res.algorithm.callback.data["p_value"][-1]
        testing_acc_ga[sub_test] = res.algorithm.callback.data["test_acc"][-1]
        rsqrd_best = res.algorithm.callback.data["rsquare"][-1]

        print('Training Acc after GA: ', training_acc_ga[sub_test])
        print('P Value      after GA: ', p_value_ga[sub_test])
        print('Testing  Acc after GA: ', testing_acc_ga[sub_test])

        if args.wandb:
            wandb.log({"metrics/train_acc_ga" : training_acc_ga[sub_test],
                       "metrics/test_acc_ga"  : testing_acc_ga[sub_test],
                       "metrics/p_value_ga"   : p_value_ga[sub_test],
                       "metrics/rsquare_ga"   : rsqrd_best})
            run.finish()
