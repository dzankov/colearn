import sys
sys.path.append('/home/zankov/dev')

import os
import shutil
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import product
from operator import itemgetter
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials
from CIMtools.model_selection.transformation_out import TransformationOut
from colearn.conjugated_nn.arrhenius_net import set_seed
from colearn.ridge_models import IndividualRidge, ArrheniusConjugatedRidge, ArrheniusIndividualRidge
from colearn.conjugated_nn.individual_net import IndividualNet
from colearn.conjugated_nn.arrhenius_net import ArrheniusConjugatedNet
from colearn.utils import arrhenius_cross_val


class HyperOpt:

    def __init__(self, algo='tpe'):

        self.algo = algo
        self.fitness = None

    def initialize(self, param_grid):

        if self.algo == 'tpe':
            self.method = tpe.suggest

        elif self.algo == 'random':
            self.method = rand.suggest

        self.param_grid = param_grid

        return self

    def set_fitness(self, func):
        self.fitness = func

    def fit(self, n_iter=100):

        def objective(params):
            loss = self.fitness(params)
            return {'loss': loss, 'params': params, 'status': STATUS_OK}

        trials = Trials()
        results = fmin(objective, self.param_grid, algo=self.method, trials=trials, max_evals=n_iter)
        self.results = trials.results
        return self

    def best_solution(self):
        best = min(self.results, key=itemgetter('loss'))['params']
        return best

    def best_score(self):
        score = res = min(self.results, key=itemgetter('loss'))['loss']
        return score

    def loss_scores(self):
        scores = [n_iter['loss'] for n_iter in self.results]
        return scores


class ModelBuider:

    def __init__(self, reacts=None, local_dir='./', hparams=None, init_cuda=None, random_state=42):

        self.hparams = hparams
        self.local_dir = local_dir
        self.init_cuda = init_cuda
        self.random_state = random_state

        self.reacts = reacts
        self.results = []

    def train_test_split(self, X, YK, YA, YE, T):

        self.ndim = (X.shape[1], 1024, 512, 256)

        solvs = [i.meta['additive.1'] for i in self.reacts]
        solvs_num = {v: n for n, v in enumerate(set(solvs))}
        groups = [solvs_num[i] for i in solvs]

        rkf = TransformationOut(n_splits=10, n_repeats=1, shuffle=True, random_state=self.random_state)
        self.train_idx, self.test_idx = list(rkf.split(self.reacts, groups=groups))[0]

        (self.X_train, self.YK_train,
         self.YA_train, self.YE_train, self.T_train) = (X[self.train_idx], YK[self.train_idx], YA[self.train_idx],
                                                        YE[self.train_idx], T[self.train_idx])

        (self.X_test, self.YK_test,
         self.YA_test, self.YE_test, self.T_test) = (X[self.test_idx], YK[self.test_idx],
                                                     YA[self.test_idx], YE[self.test_idx], T[self.test_idx])

        self.reacts_train, self.reacts_test = list(np.array(self.reacts)[self.train_idx]), list(np.array(self.reacts)[self.test_idx])

        scaler = MinMaxScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        return self

    def train_individual_model_ridge(self, task=None):

        model_dir = os.path.join(self.local_dir, 'ridge_individual_{}'.format(task))
        os.mkdir(model_dir)
        #
        tmp = []
        X_train = np.hstack((self.X_train, self.T_train.reshape(-1, 1)))
        for lmb in self.hparams['lmbE']:
            ridge = IndividualRidge(lmb=lmb, init_cuda=self.init_cuda)
            predictions, stat = arrhenius_cross_val(ridge, self.reacts_train, X_train, self.YK_train,
                                                    self.YA_train, self.YE_train, self.T_train, task=task)
            stat['lmb'] = lmb

            tmp.append(stat)
            res = pd.DataFrame(tmp)
            res.to_csv(os.path.join(model_dir, 'hyperopt.csv'), index=False)

        #
        opt = res.sort_values(by='{}_R2'.format(task), ascending=False).iloc[0]
        ridge = IndividualRidge(lmb=opt['lmb'], init_cuda=self.init_cuda)
        predictions, stat = arrhenius_cross_val(ridge, self.reacts_train, X_train, self.YK_train,
                                                self.YA_train, self.YE_train, self.T_train, task=task)
        pd.DataFrame(dict([(k, pd.Series(v)) for k, v in predictions.items()])).to_csv(os.path.join(model_dir, 'predictions_cv.csv'))
        #
        X_test = np.hstack((self.X_test, self.T_test.reshape(-1, 1)))
        if task == 'YK':
            X_train, X_test = X_train[np.isfinite(self.YK_train)], X_test[np.isfinite(self.YK_test)]
            Y_train, Y_test = self.YK_train[np.isfinite(self.YK_train)], self.YK_test[np.isfinite(self.YK_test)]
            T_train, T_test = self.T_train[np.isfinite(self.YK_train)], self.T_test[np.isfinite(self.YK_test)]
            train_idx, test_idx = self.train_idx[np.isfinite(self.YK_train)], self.test_idx[np.isfinite(self.YK_test)]
        elif task == 'YA':
            X_train, X_test = X_train[np.isfinite(self.YA_train)], X_test[np.isfinite(self.YA_test)]
            Y_train, Y_test = self.YA_train[np.isfinite(self.YA_train)], self.YA_test[np.isfinite(self.YA_test)]
            T_train, T_test = self.T_train[np.isfinite(self.YA_train)], self.T_test[np.isfinite(self.YA_test)]
            train_idx, test_idx = self.train_idx[np.isfinite(self.YA_train)], self.test_idx[np.isfinite(self.YA_test)]
        elif task == 'YE':
            X_train, X_test = X_train[np.isfinite(self.YE_train)], X_test[np.isfinite(self.YE_test)]
            Y_train, Y_test = self.YE_train[np.isfinite(self.YE_train)], self.YE_test[np.isfinite(self.YE_test)]
            T_train, T_test = self.T_train[np.isfinite(self.YE_train)], self.T_test[np.isfinite(self.YE_test)]
            train_idx, test_idx = self.train_idx[np.isfinite(self.YE_train)], self.test_idx[np.isfinite(self.YE_test)]

        ridge = IndividualRidge(lmb=opt['lmb'], init_cuda=self.init_cuda)
        ridge.fit(X_train, Y_train)
        #
        Y_pred = ridge.predict(X_test)
        pred_df = {'test_idx': test_idx,
                   '{}_pred'.format(task): Y_pred.flatten(),
                   '{}_true'.format(task): Y_test.flatten(),
                   'T_true': T_test.flatten()}
        pd.DataFrame(pred_df).to_csv(os.path.join(model_dir, 'predictions_test.csv'))
        #
        self.results.extend(
            [{'MODEL': 'ridge_individual_{}'.format(task),
              'SIZE': len(Y_train),
              'SET': 'CV',
              'YK_RMSE': opt['YK_RMSE'],
              'YA_RMSE': opt['YA_RMSE'],
              'YE_RMSE': opt['YE_RMSE'],
              'YK_R2': opt['YK_R2'],
              'YA_R2': opt['YA_R2'],
              'YE_R2': opt['YE_R2'],
              'LMB': opt['lmb']},

             {'MODEL': 'ridge_individual_{}'.format(task),
              'SET': 'TEST',
              'SIZE': len(Y_test),
              '{}_RMSE'.format(task): mean_squared_error(Y_test, Y_pred) ** 0.5,
              '{}_R2'.format(task): r2_score(Y_test, Y_pred)}])
        #
        pd.DataFrame(self.results).to_csv(os.path.join(self.local_dir, 'results.csv'), index=False)

    def hyperopt_optimize_conjugated(self, hparams, n_iter=1000, task='YKYAYE'):

        if task == 'YKYAYE':
            def objective(params):
                ridge = ArrheniusConjugatedRidge(**params)
                predictions, stat = arrhenius_cross_val(ridge, self.reacts_train, self.X_train, self.YK_train,
                                                        self.YA_train, self.YE_train,
                                                        self.T_train, task=None)

                return -(stat['YK_R2'] + stat['YA_R2'] + stat['YE_R2']) / 3

        elif task == 'YKYA':
            def objective(params):
                ridge = ArrheniusConjugatedRidge(**params)
                predictions, stat = arrhenius_cross_val(ridge, self.reacts_train, self.X_train, self.YK_train,
                                                        self.YA_train, self.YE_train,
                                                        self.T_train, task=None)

                return -(stat['YK_R2'] + stat['YA_R2']) / 2

        elif task == 'YKYE':
            def objective(params):
                ridge = ArrheniusConjugatedRidge(**params)
                predictions, stat = arrhenius_cross_val(ridge, self.reacts_train, self.X_train, self.YK_train,
                                                        self.YA_train, self.YE_train,
                                                        self.T_train, task=None)

                return -(stat['YK_R2'] + stat['YE_R2']) / 2

        elif task == 'YAYE':
            def objective(params):
                ridge = ArrheniusConjugatedRidge(**params)
                predictions, stat = arrhenius_cross_val(ridge, self.reacts_train, self.X_train, self.YK_train,
                                                        self.YA_train, self.YE_train,
                                                        self.T_train, task=None)

                return -(stat['YA_R2'] + stat['YE_R2']) / 2

        elif task == 'YK':
            def objective(params):
                ridge = ArrheniusConjugatedRidge(**params)
                predictions, stat = arrhenius_cross_val(ridge, self.reacts_train, self.X_train, self.YK_train,
                                                        self.YA_train, self.YE_train,
                                                        self.T_train, task=None)

                return -stat['YK_R2']

        elif task == 'YKZ':
            def objective(params):
                ridge = ArrheniusConjugatedRidge(**params)
                predictions, stat = arrhenius_cross_val(ridge, self.reacts_train, self.X_train, self.YK_train,
                                                        self.YA_train, self.YE_train,
                                                        self.T_train, task=None)

                return -stat['YK_R2']

        elif task == 'YAZ':
            def objective(params):
                ridge = ArrheniusConjugatedRidge(**params)
                predictions, stat = arrhenius_cross_val(ridge, self.reacts_train, self.X_train, self.YK_train,
                                                        self.YA_train, self.YE_train,
                                                        self.T_train, task=None)

                return -stat['YA_R2']

        elif task == 'YEZ':
            def objective(params):
                ridge = ArrheniusConjugatedRidge(**params)
                predictions, stat = arrhenius_cross_val(ridge, self.reacts_train, self.X_train, self.YK_train,
                                                        self.YA_train, self.YE_train,
                                                        self.T_train, task=None)

                return -stat['YE_R2']

        elif task == 'X':
            def objective(params):
                ridge = ArrheniusConjugatedRidge(**params)
                predictions, stat = arrhenius_cross_val(ridge, self.reacts_train, self.X_train, self.YK_train,
                                                        self.YA_train, self.YE_train,
                                                        self.T_train, task=None)

                return -(stat['YK_R2'] + stat['YA_R2'] + stat['YE_R2']) / 3

        hyper = HyperOpt(algo='tpe')
        hyper.set_fitness(objective)
        hyper.initialize(hparams)
        hyper.fit(n_iter=n_iter)

        return hyper.best_solution(), hyper.loss_scores()


    def train_arrhenius_conjugated_model_ridge(self, task='YKYAYE'):

        if task == 'YKYAYE':
            n_iter = 1000
            model_name = 'ridge_arrhenius_conjugated_YKYAYE'
            hparams = {'lmbA': hp.choice('lmbA', self.hparams['lmbA']),
                       'lmbE': hp.choice('lmbE', self.hparams['lmbE']),
                       'a': hp.uniform('a', 0, 1),
                       'b': hp.uniform('b', 0, 1),
                       'c': hp.uniform('c', 0, 1)}

        elif task == 'YKYA':
            n_iter = 500
            model_name = 'ridge_arrhenius_conjugated_YKYA'
            hparams = {'lmbA': hp.choice('lmbA', self.hparams['lmbA']),
                       'lmbE': hp.choice('lmbE', self.hparams['lmbE']),
                       'a': hp.uniform('a', 0, 1),
                       'b': hp.uniform('b', 0, 1),
                       'c': hp.choice('c', [0])}

        elif task == 'YKYE':
            n_iter = 500
            model_name = 'ridge_arrhenius_conjugated_YKYE'
            hparams = {'lmbA': hp.choice('lmbA', self.hparams['lmbA']),
                       'lmbE': hp.choice('lmbE', self.hparams['lmbE']),
                       'a': hp.uniform('a', 0, 1),
                       'b': hp.choice('b', [0]),
                       'c': hp.uniform('c', 0, 1)}

        elif task == 'YAYE':
            n_iter = 500
            model_name = 'ridge_arrhenius_conjugated_YAYE'
            hparams = {'lmbA': hp.choice('lmbA', self.hparams['lmbA']),
                       'lmbE': hp.choice('lmbE', self.hparams['lmbE']),
                       'a': hp.choice('a', [0]),
                       'b': hp.uniform('b', 0, 1),
                       'c': hp.uniform('c', 0, 1)}

        elif task == 'YK':
            n_iter = 100
            model_name = 'ridge_arrhenius_conjugated_YK'
            hparams = {'lmbA': hp.choice('lmbA', self.hparams['lmbA']),
                       'lmbE': hp.choice('lmbE', self.hparams['lmbE']),
                       'a': hp.choice('a', [1]),
                       'b': hp.choice('b', [0]),
                       'c': hp.choice('c', [0])}

        elif task == 'YKZ':
            n_iter = 1000
            model_name = 'ridge_arrhenius_conjugated_YKZ'
            hparams = {'lmbA': hp.choice('lmbA', self.hparams['lmbA']),
                       'lmbE': hp.choice('lmbE', self.hparams['lmbE']),
                       'a': hp.uniform('a', 0, 1),
                       'b': hp.uniform('b', 0, 1),
                       'c': hp.uniform('c', 0, 1)}

        elif task == 'YAZ':
            n_iter = 1000
            model_name = 'ridge_arrhenius_conjugated_YAZ'
            hparams = {'lmbA': hp.choice('lmbA', self.hparams['lmbA']),
                       'lmbE': hp.choice('lmbE', self.hparams['lmbE']),
                       'a': hp.uniform('a', 0, 1),
                       'b': hp.uniform('b', 0, 1),
                       'c': hp.uniform('c', 0, 1)}

        elif task == 'YEZ':
            n_iter = 1000
            model_name = 'ridge_arrhenius_conjugated_YEZ'
            hparams = {'lmbA': hp.choice('lmbA', self.hparams['lmbA']),
                       'lmbE': hp.choice('lmbE', self.hparams['lmbE']),
                       'a': hp.uniform('a', 0, 1),
                       'b': hp.uniform('b', 0, 1),
                       'c': hp.uniform('c', 0, 1)}

        elif task == 'X':
            n_iter = 100
            model_name = 'ridge_arrhenius_conjugated_X'
            hparams = {'lmbA': hp.choice('lmbA', self.hparams['lmbA']),
                       'lmbE': hp.choice('lmbE', self.hparams['lmbE']),
                       'a': hp.choice('a', [1]),
                       'b': hp.choice('b', [1]),
                       'c': hp.choice('c', [1])}

        model_dir = os.path.join(self.local_dir, model_name)
        os.mkdir(model_dir)
        #
        best_params, loss_scores = self.hyperopt_optimize_conjugated(hparams, n_iter=n_iter, task=task)
        res = pd.DataFrame()
        res['N_ITER'] = [i for i in range(len(loss_scores))]
        res['R2'] = loss_scores
        res.to_csv(os.path.join(model_dir, 'hyperopt.csv'), index=False)
        #
        ridge = ArrheniusConjugatedRidge(a=best_params['a'], b=best_params['b'], c=best_params['c'],
                                         lmbA=best_params['lmbA'], lmbE=best_params['lmbE'], init_cuda=self.init_cuda)
        predictions, opt = arrhenius_cross_val(ridge, self.reacts_train, self.X_train, self.YK_train,
                                               self.YA_train, self.YE_train, self.T_train, task=task)
        pd.DataFrame(dict([(k, pd.Series(v)) for k, v in predictions.items()])).to_csv(os.path.join(model_dir, 'predictions_cv.csv'))
        #
        ridge = ArrheniusConjugatedRidge(a=best_params['a'], b=best_params['b'], c=best_params['c'],
                                         lmbA=best_params['lmbA'], lmbE=best_params['lmbE'], init_cuda=self.init_cuda)

        ridge.fit(self.X_train[np.isfinite(self.YK_train)],
                  self.X_train[np.isfinite(self.YA_train)],
                  self.X_train[np.isfinite(self.YE_train)],
                  self.YK_train[np.isfinite(self.YK_train)],
                  self.YA_train[np.isfinite(self.YA_train)],
                  self.YE_train[np.isfinite(self.YE_train)],
                  self.T_train[np.isfinite(self.YK_train)])
        #
        testK = self.YK_test[np.isfinite(self.YK_test)]
        testA = self.YA_test[np.isfinite(self.YA_test)]
        testE = self.YE_test[np.isfinite(self.YE_test)]
        predK = ridge.predict_YK(self.X_test[np.isfinite(self.YK_test)], self.T_test[np.isfinite(self.YK_test)])
        predA = ridge.predict_YA(self.X_test[np.isfinite(self.YA_test)])
        predE = ridge.predict_YE(self.X_test[np.isfinite(self.YE_test)])

        pred_df = {'test_idx': self.test,
                   'YK_pred': predK.flatten(),
                   'YA_pred': predA.flatten(),
                   'YE_pred': predE.flatten(),
                   'YK_true': testK,
                   'YA_true': testA,
                   'YE_true': testE,
                   'T_true': self.T_test}
        pd.DataFrame(dict([(k, pd.Series(v)) for k, v in pred_df.items()])).to_csv(os.path.join(model_dir, 'predictions_test.csv'))
        #
        self.results.extend(
            [{'MODEL': model_name,
              'SET': 'CV',
              'LMBA': best_params['lmbA'],
              'LMBE': best_params['lmbE'],
              'A': best_params['a'],
              'B': best_params['b'],
              'C': best_params['c'],
              'YK_RMSE': opt['YK_RMSE'],
              'YA_RMSE': opt['YA_RMSE'],
              'YE_RMSE': opt['YE_RMSE'],
              'YK_R2': opt['YK_R2'],
              'YA_R2': opt['YA_R2'],
              'YE_R2': opt['YE_R2']},

             {'MODEL': model_name,
              'SET': 'TEST',
              'YK_RMSE': mean_squared_error(testK, predK) ** 0.5,
              'YK_R2': r2_score(testK, predK),

              'YA_RMSE': mean_squared_error(testA, predA) ** 0.5,
              'YA_R2': r2_score(testA, predA),

              'YE_RMSE': mean_squared_error(testE, predE) ** 0.5,
              'YE_R2': r2_score(testE, predE)}])
        #
        pd.DataFrame(self.results).to_csv(os.path.join(self.local_dir, 'results.csv'), index=False)


    def train_individual_model_net(self, task=None):

        set_seed(42)
        model_dir = os.path.join(self.local_dir, 'net_individual_{}'.format(task))
        os.mkdir(model_dir)
        #
        X_train = np.hstack((self.X_train, self.T_train.reshape(-1, 1)))
        net = IndividualNet(ndim=self.ndim, init_cuda=self.init_cuda)
        predictions, stat = arrhenius_cross_val(net, self.reacts_train, X_train, self.YK_train,
                                                self.YA_train, self.YE_train, self.T_train, task=task)
        pd.DataFrame(dict([(k, pd.Series(v)) for k, v in predictions.items()])).to_csv(os.path.join(model_dir, 'predictions_cv.csv'))
        #
        X_test = np.hstack((self.X_test, self.T_test.reshape(-1, 1)))
        if task == 'YK':
            X_train, X_test = X_train[np.isfinite(self.YK_train)], X_test[np.isfinite(self.YK_test)]
            Y_train, Y_test = self.YK_train[np.isfinite(self.YK_train)], self.YK_test[np.isfinite(self.YK_test)]
            T_train, T_test = self.T_train[np.isfinite(self.YK_train)], self.T_test[np.isfinite(self.YK_test)]
            reacts_train = list(np.array(self.reacts_train)[np.isfinite(self.YK_train)])
            train_idx, test_idx = self.train_idx[np.isfinite(self.YK_train)], self.test_idx[np.isfinite(self.YK_test)]
        elif task == 'YA':
            X_train, X_test = X_train[np.isfinite(self.YA_train)], X_test[np.isfinite(self.YA_test)]
            Y_train, Y_test = self.YA_train[np.isfinite(self.YA_train)], self.YA_test[np.isfinite(self.YA_test)]
            T_train, T_test = self.T_train[np.isfinite(self.YA_train)], self.T_test[np.isfinite(self.YA_test)]
            reacts_train = list(np.array(self.reacts_train)[np.isfinite(self.YA_train)])
            train_idx, test_idx = self.train_idx[np.isfinite(self.YA_train)], self.test_idx[np.isfinite(self.YA_test)]
        elif task == 'YE':
            X_train, X_test = X_train[np.isfinite(self.YE_train)], X_test[np.isfinite(self.YE_test)]
            Y_train, Y_test = self.YE_train[np.isfinite(self.YE_train)], self.YE_test[np.isfinite(self.YE_test)]
            T_train, T_test = self.T_train[np.isfinite(self.YE_train)], self.T_test[np.isfinite(self.YE_test)]
            reacts_train = list(np.array(self.reacts_train)[np.isfinite(self.YE_train)])
            train_idx, test_idx = self.train_idx[np.isfinite(self.YE_train)], self.test_idx[np.isfinite(self.YE_test)]

        net = IndividualNet(ndim=ndim, init_cuda=self.init_cuda)
        net.fit(X_train, Y_train, reacts=reacts_train)
        #
        Y_pred = net.predict(X_test)
        pred_df = {'test_idx': test_idx,
                   '{}_pred'.format(task): Y_pred.flatten(),
                   '{}_true'.format(task): Y_test.flatten(),
                   'T_true': T_test.flatten()}
        pd.DataFrame(pred_df).to_csv(os.path.join(model_dir, 'predictions_test.csv'))
        #
        self.results.extend(
            [{'MODEL': 'net_individual_{}'.format(task),
              'SIZE': len(Y_train),
              'SET': 'CV',
              'YK_RMSE': stat['YK_RMSE'],
              'YA_RMSE': stat['YA_RMSE'],
              'YE_RMSE': stat['YE_RMSE'],
              'YK_R2': stat['YK_R2'],
              'YA_R2': stat['YA_R2'],
              'YE_R2': stat['YE_R2'],
              'LMB': None},

             {'MODEL': 'net_individual_{}'.format(task),
              'SET': 'TEST',
              'SIZE': len(Y_test),
              '{}_RMSE'.format(task): mean_squared_error(Y_test, Y_pred) ** 0.5,
              '{}_R2'.format(task): r2_score(Y_test, Y_pred)}])
        #
        pd.DataFrame(self.results).to_csv(os.path.join(self.local_dir, 'results.csv'), index=False)

    def train_arrhenius_conjugated_model_net(self, task='YKYAYE'):

        set_seed(42)
        if task == 'YKYAYE':
            model_name = 'net_arrhenius_conjugated_YKYAYE'
            hparams = {'a': None, 'b': None, 'c': None, 'ndim': self.ndim, 'init_cuda': self.init_cuda}
        elif task == 'YKYA':
            model_name = 'net_arrhenius_conjugated_YKYA'
            hparams = {'a': None, 'b': None, 'c': 0, 'ndim': self.ndim, 'init_cuda': self.init_cuda}
        elif task == 'YKYE':
            model_name = 'net_arrhenius_conjugated_YKYE'
            hparams = {'a': None, 'b': 0, 'c': None, 'ndim': self.ndim, 'init_cuda': self.init_cuda}
        elif task == 'YAYE':
            model_name = 'net_arrhenius_conjugated_YAYE'
            hparams = {'a': 0, 'b': None, 'c': None, 'ndim': self.ndim, 'init_cuda': self.init_cuda}
        elif task == 'YK':
            model_name = 'net_arrhenius_conjugated_YK'
            hparams = {'a': None, 'b': 0, 'c': 0, 'ndim': self.ndim, 'init_cuda': self.init_cuda}
        elif task == 'X':
            model_name = 'net_arrhenius_conjugated_X'
            hparams = {'a': 1, 'b': 1, 'c': 1, 'ndim': self.ndim, 'init_cuda': self.init_cuda}

        #
        model_dir = os.path.join(self.local_dir, model_name)
        os.mkdir(model_dir)
        #
        net = ArrheniusConjugatedNet(**hparams)
        predictions, opt = arrhenius_cross_val(net, self.reacts_train, self.X_train, self.YK_train,
                                               self.YA_train, self.YE_train, self.T_train, task=task)
        pd.DataFrame(dict([(k, pd.Series(v)) for k, v in predictions.items()])).to_csv(os.path.join(model_dir, 'predictions_cv.csv'))
        #
        net = ArrheniusConjugatedNet(**hparams)
        net.fit(self.X_train, self.X_train, self.X_train, self.YK_train,
                self.YA_train, self.YE_train, self.T_train,reacts=self.reacts_train)

        testK = self.YK_test[np.isfinite(self.YK_test)]
        testA = self.YA_test[np.isfinite(self.YA_test)]
        testE = self.YE_test[np.isfinite(self.YE_test)]
        predK = net.predict_YK(self.X_test[np.isfinite(self.YK_test)], self.T_test[np.isfinite(self.YK_test)])
        predA = net.predict_YA(self.X_test[np.isfinite(self.YA_test)])
        predE = net.predict_YE(self.X_test[np.isfinite(self.YE_test)])

        pred_df = {'test_idx': self.test_idx,
                   'YK_pred': predK.flatten(),
                   'YA_pred': predA.flatten(),
                   'YE_pred': predE.flatten(),
                   'YK_true': testK,
                   'YA_true': testA,
                   'YE_true': testE,
                   'T_true': self.T_test}
        pd.DataFrame(dict([(k, pd.Series(v)) for k, v in pred_df.items()])).to_csv(os.path.join(model_dir, 'predictions_test.csv'))
        #
        self.results.extend(
            [{'MODEL': model_name,
              'SET': 'CV',
              'LMBA': None,
              'LMBE': None,
              'A': net.a,
              'B': net.b,
              'C': net.c,
              'YK_RMSE': opt['YK_RMSE'],
              'YA_RMSE': opt['YA_RMSE'],
              'YE_RMSE': opt['YE_RMSE'],
              'YK_R2': opt['YK_R2'],
              'YA_R2': opt['YA_R2'],
              'YE_R2': opt['YE_R2']},

             {'MODEL': model_name,
              'SET': 'TEST',
              'YK_RMSE': mean_squared_error(testK, predK) ** 0.5,
              'YK_R2': r2_score(testK, predK),

              'YA_RMSE': mean_squared_error(testA, predA) ** 0.5,
              'YA_R2': r2_score(testA, predA),

              'YE_RMSE': mean_squared_error(testE, predE) ** 0.5,
              'YE_R2': r2_score(testE, predE)}])

        pd.DataFrame(self.results).to_csv(os.path.join(self.local_dir, 'results.csv'), index=False)