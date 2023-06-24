#!/usr/bin/env python
# coding: utf-8
import wandb
import sys
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

from multiprocessing.pool import ThreadPool

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.callback import Callback
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display.multi import MultiObjectiveOutput


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
   
    def load_data_svm(self, x_train, y_train, c_train, clf, permu):
        # Load informations from the individual classification exerpiment 
        # x_train - training features
        # y_train - labels
        # c_train - confounding variables
        # model   - the trained svm model
        self.x_train = x_train
        self.y_train = y_train
        self.c_train = c_train
        self.clf     = clf
        self.permu   = permu

        # dimension of the training feature
        self.n = np.shape(x_train)[0]
        self.d = np.shape(x_train)[1]
 
    def _evaluate(self, x, out, *args, **kwargs):
        # pymoo initialize the chromosome as a 1-D array which can be converted
        # into matrix for element-wise weight multiplication
        fw = np.matlib.repmat(x, self.n, 1)
        x_train_tf = self.x_train * fw

        # first objective is SVM training accuracy
        f1 = 1 - self.clf.score(x_train_tf, self.y_train)

        # second objective is P Value from CPT  
        y_hat = self.clf.predict(x_train_tf)
        ret = partial_confound_test(self.y_train, y_hat, self.c_train,
                                    cat_y=True, cat_yhat=True, cat_c=False,
                                    cond_dist_method='gam', 
                                    num_perms=self.permu, mcmc_steps=50,
                                    n_jobs=-1,
                                    progress=False)

        f2 = 1 - ret.p 

        out['F'] = [f1, f2]

class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm):
        self.data["best"].append(algorithm.pop.get("F")[0].min())
        wandb.log({"ga/n_gen"       : algorithm.n_gen,
                   "ga/1-train_acc" : algorithm.pop.get("F")[0].min(),
                   "ga/1-p_value"   : algorithm.pop.get("F")[1].min()})

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

# sub_test = int(sys.argv[1])

for sub_test in range(int(sys.argv[1]), int(sys.argv[1])+4): 

    sub_txt = "R%03d"%(int(SUBJECT_ID[sub_test][0][0]))
    if int(VFI_1[sub_test][0][0]) > 10:
        sub_group = 'Fatigued'
    else:
        sub_group = 'Healthy'

    # parameters for testing
    # config = {"num_generation"  : 1,
            #   "population_size" : 64,
            #   "permutation"     : 50,
            #   "threads"         : 16}

    config = {"num_generation"  : 20,
              "population_size" : 64,
              "permutation"     : 1000,
              "threads"         : 16}

    run = wandb.init(project  = 'LOO Vowels GA-SVM RBF',
                     group    = 'experiment_parallel_1',
                     config   = config,
                     name     = sub_txt,
                     tags     = [sub_group],
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
    C_TV = np.zeros(0)
    for sub_train in range(40):
        if sub_train != sub_test:
            x_s = FEAT_N[sub_train,0]
            y_s = LABEL[sub_train,0].flatten()
            c_s = np.mean(np.mean(SUBJECT_SKINFOLD[sub_train,:]), axis=1)
            X_TV = np.concatenate((X_TV, x_s), axis=0)
            Y_TV = np.concatenate((Y_TV, y_s), axis=0)
            C_TV = np.concatenate((C_TV, c_s), axis=0)       

    print('# of Healthy Samples: %d'%(np.sum(Y_TV == -1)))
    print('# of Fatigued Samples: %d'%(np.sum(Y_TV == 1)))    

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

    print('Training Acc: ', accuracy_score(label_predict, Y_Train))
    training_acc[sub_test] = accuracy_score(label_predict, Y_Train)

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
    print('Testing  Acc: ', accuracy_score(label_predict, Y_Test))
    testing_acc[sub_test] = accuracy_score(label_predict, Y_Test)

    wandb.log({"metrics/train_acc" : training_acc[sub_test],
               "metrics/test_acc"  : testing_acc[sub_test],
               "metrics/rsquare"   : rsqrd,
               "metrics/p_value"   : p_value[sub_test]})

    print('Genetic Algorithm Optimization...')

    num_permu       = wandb.config["permutation"]
    num_generation  = wandb.config["num_generation"]
    population_size = wandb.config["population_size"]
    threads_count   = wandb.config["threads"]

    n_threads = threads_count
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)

    problem = MyProblem(elementwise_runner=runner)
    problem.load_data_svm(X_Train, Y_Train, C_Train, clf, num_permu)

    # Genetic algorithm initialization
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

    # Plot the parento front
    plt.figure()
    plt.scatter(res.F[:,0], res.F[:,1], marker='o', 
                                        edgecolors='red', 
                                        facecolor='None' )
    plt.xlabel("1-train_acc")
    plt.ylabel("1-p value")
    wandb.log({"plots/scatter_plot": wandb.Image(plt)})

    # Log and save the weights
    fw_dataframe = pd.DataFrame(res.X)
    fw_table = wandb.Table(dataframe=fw_dataframe)
    run.log({"feature weights": fw_table})

    # Evaluate the results discovered by GA
    Xid = np.argsort(res.F[:,0])
    acc_best = 0
    for t in range(np.shape(res.X)[0]):
        w = res.X[Xid[t],:]

        # Evalute the training performance
        n = np.shape(X_Train)[0]
        fw = np.matlib.repmat(w, n, 1)
        x_train_tf = X_Train * fw
        Y_tf_train = clf.predict(x_train_tf)
        # temp_tr_acc = accuracy_score(Y_tf_train, Y_Train)
        temp_tr_acc = clf.score(x_train_tf, Y_Train)

        # Evaluate the r squared
        df = pd.DataFrame({'x': C_Train, 'y': Y_tf_train})
        fit = ols('y~C(x)', data=df).fit()
        temp_rsqrd = fit.rsquared.flatten()[0]

        # Evaluate the p value from the current predicitons
        ret_ga = partial_confound_test(Y_Train, Y_tf_train, C_Train, 
                                cat_y=True, cat_yhat=True, cat_c=False,
                                cond_dist_method='gam',
                                progress=False)
        temp_p_value = ret_ga.p

        # Evaluate the testing performance
        n = np.shape(X_Test)[0]
        fw = np.matlib.repmat(w, n, 1)
        x_test_tf = X_Test * fw
        Y_tf_test = clf.predict(x_test_tf)
        temp_te_acc = accuracy_score(Y_tf_test, Y_Test)

        wandb.log({"pareto-front/train_acc": temp_tr_acc,
                   "pareto-front/rsquare"  : temp_rsqrd,
                   "pareto-front/p_value"  : temp_p_value,
                   "pareto-front/test_acc" : temp_te_acc})

        # Detect if the current chromosome gives the best predictio`n
        if temp_te_acc > acc_best:
            acc_best = temp_te_acc 

            training_acc_ga[sub_test] = temp_tr_acc 
            p_value_ga[sub_test]      = temp_p_value
            rsqrd_best                = temp_rsqrd
            testing_acc_ga[sub_test]  = temp_te_acc


    print('Training Acc after GA: ', training_acc_ga[sub_test])
    print('P Value      after GA: ', p_value_ga[sub_test])
    print('Testing  Acc after GA: ', testing_acc_ga[sub_test])

    wandb.log({"metrics/train_acc_ga" : training_acc_ga[sub_test],
               "metrics/test_acc_ga"  : testing_acc_ga[sub_test],
               "metrics/p_value_ga"   : p_value_ga[sub_test],
               "metrics/rsquare_ga"   : rsqrd_best})

    run.finish()