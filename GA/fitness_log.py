import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from mlconfound.stats import partial_confound_test

class MyProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=48, 
                         n_obj=2,
                         n_constr=0,
                         xl = -2*np.ones(48),
                         xu =  2*np.ones(48),
                         **kwargs)

    def load_param(self, x_train, y_train, c_train, x_test, y_test, clf, permu):
        # Load informations from the individual classification exerpiment 
        # x_train - training features
        # y_train - labels
        # c_train - confounding variables
        # model   - the trained svm model
        self.x_train  = x_train
        self.y_train  = y_train
        self.c_train  = c_train
        self.x_test   = x_test
        self.y_test   = y_test
        self.clf      = clf
        self.permu    = permu

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
    def __init__(self, log_flag) -> None:
        super().__init__()
        self.data["train_acc"] = []
        self.data["test_acc"] = []
        self.data["p_value"] = []
        self.data["rsquare"] = []
        self.data["predict"] = []
        self.log_flag = log_flag

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        X = algorithm.pop.get("X")
        
        # Evaluate the results from GA
        Xid = np.argsort(F[:,0])
        acc_best = 0        
        tr_acc_best  = 0 
        p_value_best = 0 
        rsqrd_best   = 0 
        te_acc_best  = 0 
        predict_best = []
        for t in range(np.shape(X)[0]):
            w = X[Xid[t],:]

            # Evalute the training performance
            fw = np.matlib.repmat(w, algorithm.problem.n, 1)
            x_train_tf  = algorithm.problem.x_train * fw
            y_train_tf  = algorithm.problem.clf.predict(x_train_tf)
            temp_tr_acc = algorithm.problem.clf.score(x_train_tf, 
                                                      algorithm.problem.y_train)

            # Evaluate the r squared
            df = pd.DataFrame({'x': algorithm.problem.c_train, 'y': y_train_tf})
            fit = ols('y~C(x)', data=df).fit()
            temp_rsqrd = fit.rsquared.flatten()[0]

            # Evaluate the p value from the current predicitons
            ret_ga = partial_confound_test(algorithm.problem.y_train, 
                                           y_train_tf, 
                                           algorithm.problem.c_train, 
                                           cat_y=True, 
                                           cat_yhat=True, 
                                           cat_c=False,
                                           cond_dist_method='gam',
                                           progress=False)
            temp_p_value = ret_ga.p

            # Evaluate the testing performance
            n = np.shape(algorithm.problem.x_test)[0]
            fw = np.matlib.repmat(w, n, 1)
            x_test_tf = algorithm.problem.x_test * fw
            temp_te_acc = algorithm.problem.clf.score(x_test_tf, 
                                                      algorithm.problem.y_test)

            label_pred = algorithm.problem.clf.predict(x_test_tf)

            if self.log_flag:
                wandb.log({"pareto-front/train_acc-{}".format(algorithm.n_gen): temp_tr_acc,
                           "pareto-front/rsquare-{}".format(algorithm.n_gen)  : temp_rsqrd,
                           "pareto-front/p_value-{}".format(algorithm.n_gen)  : temp_p_value,
                           "pareto-front/test_acc-{}".format(algorithm.n_gen) : temp_te_acc})

            if temp_te_acc > acc_best:
                acc_best = temp_te_acc 

                tr_acc_best  = temp_tr_acc 
                p_value_best = temp_p_value
                rsqrd_best   = temp_rsqrd
                te_acc_best  = temp_te_acc
                predict_best = algorithm.problem.clf.predict(x_test_tf)

        if self.log_flag:
            wandb.log({"ga/n_gen"     : algorithm.n_gen,
                       "ga/train_acc" : tr_acc_best,
                       "ga/p_value"   : p_value_best,
                       "ga/rsquare"   : rsqrd_best,
                       "ga/test_acc"  : te_acc_best})

        self.data["train_acc"].append(tr_acc_best)
        self.data["p_value"].append(p_value_best)
        self.data["rsquare"].append(rsqrd_best)  
        self.data["test_acc"].append(te_acc_best)
        self.data["predict"].append(predict_best)
