#!/usr/bin/env python

import os
import os.path
import sys
import json
import random
import optuna
import joblib
import argparse

import numpy as np
import pandas as pd

from rdkit import Chem
from xgboost import XGBRegressor, XGBClassifier
from datetime import datetime

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.model_selection import StratifiedKFold, KFold

from drugex import DEFAULT_GPUS
from drugex.logs.utils import backUpFiles, enable_file_logger, commit_hash
# from drugex.environment.data import QSARDataset
# from drugex.environment.models import 

from drugex.training import models
from drugex.training.scorers.predictors import Predictor

class QSARDataset:
    """
        This class is used to prepare the dataset for QSAR model training. 
        It splits the data in train and test set, as well as creating cross-validation folds.
        Optionally low quality data is filtered out.
        For classification the dataset samples are labelled as active/inactive.
        ...

        Attributes
        ----------
        base_dir (str)            : base directory, needs to contain a folder data with .tsv file containing data
        input  (str)              : tsv file containing SMILES, target accesion & corresponding data
        target (str)              : target identifier, corresponding with accession in papyrus dataset
        reg (bool)                : if true, dataset for regression, if false dataset for classification
        timesplit (int), optional : Year to split test set on
        test_size (int or float), optional: Used when timesplit is None
                                            If float, should be between 0.0 and 1.0 and is proportion of dataset to
                                            include in test split. If int, represents absolute number of test samples.
        th (float)                : threshold for activity if classficiation model, ignored otherwise
        keep_low_quality (bool)   : if true low quality data is included in the dataset
        X (np.ndarray)            : m x n feature matrix for cross validation, where m is the number of samples
                                    and n is the number of features.
        y (np.ndarray)            : m-d label array for cross validation, where m is the number of samples and
                                    equals to row of X.
        X_ind (np.ndarray)        : m x n Feature matrix for independent set, where m is the number of samples
                                    and n is the number of features.
        y_ind (np.ndarray)        : m-l label array for independent set, where m is the number of samples and
                                    equals to row of X_ind, and l is the number of types.

        Methods
        -------
        split_dataset : A train and test split is made.
        create_folds: folds is an generator and needs to be reset after cross validation or hyperparameter optimization
        data_standardization: Performs standardization by centering and scaling
    """
    def __init__(self, base_dir, input, target, reg=True, timesplit=None, test_size=0.1, th=6.5, keep_low_quality=False):
        self.base_dir = base_dir
        self.input = input
        self.target = target
        self.reg = reg
        self.timesplit = timesplit
        self.test_size = test_size
        self.th = th
        self.keep_low_quality = keep_low_quality
        self.X = None
        self.y = None
        self.X_ind = None
        self.y_ind = None
        self.folds = None

    def split_dataset(self):
        """
        Splits the dataset in a train and temporal test set.
        Calculates the predictors for the QSAR models.
        """

        #read in the dataset
        df = pd.read_table('%s/data/%s' % (self.base_dir, self.input)).dropna(subset=['SMILES']) #drops if smiles is missing
        df = df[df['accession'] == self.target]
        df = df[['accession', 'SMILES', 'pchembl_value_Mean', 'Quality', 'Year']].set_index(['SMILES'])

        #Get indexes of samples test set based on temporal split
        if self.timesplit:
            year = df[['Year']].groupby(['SMILES']).min().dropna()
            test_idx = year[year['Year'] > self.timesplit].index

        # filter out low quality data if desired
        df = df if self.keep_low_quality else df[df.Quality.isin(['High','Medium'])] 

        #keep only pchembl values and make binary for classification
        df = df['pchembl_value_Mean']

        if not self.reg:
            df = (df > self.th).astype(float)

        #get test and train (data) set with set temporal split or random split
        df = df.sample(len(df)) 
        if self.timesplit:
            test_ix = set(df.index).intersection(test_idx)
            test = df.loc[test_ix].dropna()
        else:
            test = df.sample(int(round(len(df)*self.test_size))) if type(self.test_size) == float else df.sample(self.test_size)
        data = df.drop(test.index)

        #calculate ecfp and physiochemical properties as input for the predictors
        self.X_ind = Predictor.calculateDescriptors([Chem.MolFromSmiles(mol) for mol in test.index])
        self.X = Predictor.calculateDescriptors([Chem.MolFromSmiles(mol) for mol in data.index])

        self.y_ind = test.values
        self.y = data.values

        #Create folds for crossvalidation
        self.create_folds()

        #Write information about the trainingset to the log
        log.info('Train and test set created for %s %s:' % (self.target, 'REG' if self.reg else 'CLS'))
        log.info('    Total: train: %s test: %s' % (len(data), len(test)))
        if self.reg:
            log.info('    Total: active: %s not active: %s' % (sum(df >= self.th), sum(df < self.th)))
            log.info('    In train: active: %s not active: %s' % (sum(data >= self.th), sum(data < self.th)))
            log.info('    In test:  active: %s not active: %s\n' % (sum(test >= self.th), sum(test < self.th)))
        else:
            log.info('    Total: active: %s not active: %s' % (df.sum().astype(int), (len(df)-df.sum()).astype(int)))
            log.info('    In train: active: %s not active: %s' % (data.sum().astype(int), (len(data)-data.sum()).astype(int)))
            log.info('    In test:  active: %s not active: %s\n' % (test.sum().astype(int), (len(test)-test.sum()).astype(int)))
    
    def create_folds(self):
        """
            Create folds for crossvalidation
        """
        if self.reg:
            self.folds = KFold(5).split(self.X)
        else:
            self.folds = StratifiedKFold(5).split(self.X, self.y)
        log.debug("Folds created for crossvalidation")
        
    @staticmethod
    def data_standardization(data_x, test_x):
        """
        Perform standardization by centering and scaling

        Arguments:
                    data_x (list): descriptors of data set
                    test_x (list): descriptors of test set
        
        Returns:
                    data_x (list): descriptors of data set standardized
                    test_x (list): descriptors of test set standardized
        """
        scaler = Scaler(); scaler.fit(data_x)
        test_x = scaler.transform(test_x)
        data_x = scaler.transform(data_x)
        log.debug("Data standardized")
        return data_x, test_x

class QSARModel:
    """ Model initialization, fit, cross validation and hyperparameter optimization for classifion/regression models.
        ...

        Attributes
        ----------
        data: instance QSARDataset
        alg:  instance of estimator
        parameters (dict): dictionary of algorithm specific parameters
        search_space_bs (dict): search space for bayesian optimization
        search_space_gs (dict): search space for grid search
        save_m (bool): if true, save final model

        
        Methods
        -------
        init_model: initialize model from saved hyperparameters
        fit_model: build estimator model from entire data set
        objective: objective used by bayesian optimization
        bayes_optimization: bayesian optimization of hyperparameters using optuna
        grid_search: optimization of hyperparameters using grid_search

    """
    def __init__(self, data, alg, parameters, search_space_bs, search_space_gs, save_m=True):
        self.data = data
        self.alg = alg
        self.parameters = parameters
        self.search_space_bs = search_space_bs
        self.search_space_gs = search_space_gs
        self.save_m = save_m
        self.model = None

        d = '%s/envs' % data.base_dir
        self.out = '%s/%s_%s_%s' % (d, self.__class__.__name__, 'REG' if data.reg else 'CLS', data.target)
        log.info('Model intialized: %s' % self.out)

    def init_model(self, n_jobs=-1):
        """
            initialize model from saved or default hyperparameters
        """
        if os.path.isfile('%s_params.json' % self.out):
            
            with open('%s_params.json' % self.out) as j:
                self.parameters = json.loads(j.read())
            log.info('loaded model parameters from file: %s_params.json' % self.out)
        self.model = self.alg.set_params(n_jobs=n_jobs, **self.parameters)
        log.info('parameters: %s' % self.parameters)

    def fit_model(self):
        """
            build estimator model from entire data set
        """
        X_all = np.concatenate([self.data.X, self.data.X_ind], axis=0)
        y_all = np.concatenate([self.data.y, self.data.y_ind], axis=0)
        # KNN and PLS do not use sample_weight
        fit_set = {'X':X_all}
        if type(self.alg).__name__ not in ['KNeighborsRegressor', 'KNeighborsClassifier', 'PLSRegression']:
            fit_set['sample_weight'] = [1 if v >= 4 else 0.1 for v in y_all]
        
        if type(self.alg).__name__ == 'PLSRegression':
            fit_set['Y'] = y_all
        else:
            fit_set['y'] = y_all

        log.info('Model fit started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))  
        self.model.fit(**fit_set)
        log.info('Model fit ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        joblib.dump(self.model, '%s.pkg' % self.out, compress=3)

    def objective(self, trial):
        """
            objective for bayesian optimization
        """

        if type(self.alg).__name__ in ['XGBRegressor', 'XGBClassifier']:
            bayesian_params = {'verbosity': 0}
        else:
            bayesian_params = {}

        for key, value in self.search_space_bs.items():
            if value[0] == 'categorical':
                bayesian_params[key] = trial.suggest_categorical(key, value[1])
            elif value[0] == 'discrete_uniform':
                bayesian_params[key] = trial.suggest_discrete_uniform(key, value[1], value[2], value[3])
            elif value[0] == 'float':
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2])
            elif value[0] == 'int':
                bayesian_params[key] = trial.suggest_int(key, value[1], value[2])
            elif value[0] == 'loguniform':
                bayesian_params[key] = trial.suggest_loguniform(key, value[1], value[2])
            elif value[0] == 'uniform':
                bayesian_params[key] = trial.suggest_uniform(key, value[1], value[2])

        self.model = self.alg.set_params(**bayesian_params)

        if self.data.reg: 
            score = metrics.explained_variance_score(self.data.y, self.model_evaluation(save = False))
        else:
            score = metrics.roc_auc_score(self.data.y, self.model_evaluation(save = False))

        return score

    def bayes_optimization(self, n_trials):
        """
            bayesian optimization of hyperparameters using optuna
        """
        print('Bayesian optimization can take a while for some hyperparameter combinations')
        #TODO add timeout function
        study = optuna.create_study(direction='maximize')
        log.info('Bayesian optimization started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        study.optimize(lambda trial: self.objective(trial), n_trials)
        log.info('Bayesian optimization ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        trial = study.best_trial

        self.model = self.alg.set_params(**trial.params)
        
        if self.save_m:
            joblib.dump(self.model, '%s.pkg' % self.out, compress=3)

        self.data.create_folds()

        log.info('Bayesian optimization best params: %s' % trial.params)
        with open('%s_params.json' % self.out, 'w') as f:
            json.dump(trial.params, f)

    def grid_search(self):
        """
            optimization of hyperparameters using grid_search
        """          
        scoring = 'explained_variance' if self.data.reg else 'roc_auc'    
        grid = GridSearchCV(self.alg, self.search_space_gs, n_jobs=10, verbose=1, cv=self.data.folds, scoring=scoring, refit=self.save_m)
        
        
        #TODO maybe move the model fitting and saving to environment?
        fit_set = {'X':self.data.X}
        fit_set['y'] = self.data.y
        if type(self.alg).__name__ not in ['KNeighborsRegressor', 'KNeighborsClassifier', 'PLSRegression']:
            fit_set['sample_weight'] = [1 if v >= 4 else 0.1 for v in self.data.y]
        log.info('Grid search started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        grid.fit(**fit_set)
        log.info('Grid search ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        self.model = grid.best_estimator_
        
        if self.save_m:
            joblib.dump(self.model, '%s.pkg' % self.out, compress=3)

        self.data.create_folds()

        log.info('Grid search best parameters: %s' % grid.best_params_)
        with open('%s_params.json' % self.out, 'w') as f:
            json.dump(grid.best_params_, f)
            
    def model_evaluation(self, save=True):
        """
            Make predictions for crossvalidation and independent test set
            arguments:
                save (bool): don't save predictions when used in bayesian optimization
        """
        cvs = np.zeros(self.data.y.shape)
        inds = np.zeros(self.data.y_ind.shape)
        for i, (trained, valided) in enumerate(self.data.folds):
            log.info('cross validation fold %s started: %s' % (i, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            # use sample weight to decrease the weight of low quality datapoints
            fit_set = {'X':self.data.X[trained]}
            if type(self.alg).__name__ not in ['KNeighborsRegressor', 'KNeighborsClassifier', 'PLSRegression']:
                fit_set['sample_weight'] = [1 if v >= 4 else 0.1 for v in self.data.y[trained]]
            if type(self.alg).__name__ == 'PLSRegression':
                fit_set['Y'] = self.data.y[trained]
            else:
                fit_set['y'] = self.data.y[trained]
            self.model.fit(**fit_set)
            
            if type(self.alg).__name__ == 'PLSRegression':
                cvs[valided] = self.model.predict(self.data.X[valided])[:, 0]
                inds += self.model.predict(self.data.X_ind)[:, 0]
            elif self.data.reg:
                cvs[valided] = self.model.predict(self.data.X[valided])
                inds += self.model.predict(self.data.X_ind)
            else:
                cvs[valided] = self.model.predict_proba(self.data.X[valided])[:, 1]
                inds += self.model.predict_proba(self.data.X_ind)[:, 1]
            log.info('cross validation fold %s ended: %s' % (i, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        #save crossvalidation results
        if save:
            train, test = pd.Series(self.data.y).to_frame(name='Label'), pd.Series(self.data.y_ind).to_frame(name='Label')
            train['Score'], test['Score'] = cvs, inds / 5
            train.to_csv(self.out + '.cv.tsv', sep='\t')
            test.to_csv(self.out + '.ind.tsv', sep='\t')

        self.data.create_folds()

        return cvs

class RF(QSARModel):
    """ Random forest regressor and classifier initialization. Here the model instance is created 
        and parameters and search space can be defined
        ...

        Attributes
        ----------
        data: instance of QSARDataset
        parameters (dict): random forest specific parameters
        save_m (bool): if true, save final model

    """
    def __init__(self, data, save_m=True, parameters=None):
        self.alg = RandomForestRegressor() if data.reg else RandomForestClassifier()
        self.parameters=parameters if parameters != None else {'n_estimators': 1000}

        # set the search space for bayesian optimization
        self.search_space_bs = {
            'n_estimators': ['int', 10, 2000],
            'max_depth': ['int', 1, 100],
            'min_samples_leaf': ['int', 1, 25],
            'max_features': ['int', 1, 100], 
            'min_samples_split': ['int', 2, 12] 
        }
        if data.reg:
            self.search_space_bs.update({'criterion' : ['categorical', ['squared_error', 'poisson']]})
        else:
            self.search_space_bs.update({'criterion' : ['categorical', ['gini', 'entropy']]})

        # set the search space for grid search
        self.search_space_gs = {
            'max_depth': [None, 20, 50, 100],
            'max_features': ['auto', 'log2'],
            'min_samples_leaf': [1, 3, 5],
            'min_samples_split': [2, 5, 12],
            'n_estimators': [100, 200, 300, 1000]
        }
        super().__init__(data, self.alg, self.parameters, self.search_space_bs, self.search_space_gs, save_m=save_m)

class XGB(QSARModel):
    """ 
        XG Boost regressor and classifier initialization. Here the model instance is created 
        and parameters and search space can be defined
        ...

        Attributes
        ----------
        data: instance of QSARDataset
        parameters (dict): XGboost specific parameters
        save_m (bool): if true, save final model

    """
    def __init__(self, data, save_m=True, parameters=None):
        self.alg = XGBRegressor(objective='reg:squarederror') if data.reg else XGBClassifier(objective='binary:logistic',use_label_encoder=False, eval_metric='logloss')
        self.parameters=parameters if parameters != None else {'nthread': 4, 'n_estimators': 1000}
        self.search_space_bs = {
            'n_estimators': ['int', 100, 1000],
            'max_depth': ['int',3, 10],
            'learning_rate': ['uniform', 0.01, 0.1] #so called `eta` value
        }
        
        self.search_space_gs = {
            'nthread':[4], #when use hyperthread, xgboost may become slower
            'learning_rate': [0.01, 0.05, 0.1], #so called `eta` value
            'max_depth': [3,6,10],
            'n_estimators': [100, 500, 1000],
            'colsample_bytree': [0.3, 0.5, 0.7]
        }
        super().__init__(data, self.alg, self.parameters, self.search_space_bs, self.search_space_gs, save_m=save_m)

class SVM(QSARModel):
    """ Support vector regressor and classifier initialization. Here the model instance is created 
        and parameters and search space can be defined
        ...

        Attributes
        ----------
        data: instance of QSARDataset
        parameters (dict): SVM specific parameters
        save_m (bool): if true, save final model

    """
    def __init__(self, data, save_m=True, parameters=None):
        self.alg = SVR() if data.reg else SVC(probability=True)
        self.parameters=parameters if parameters != None else {}
        #parameter dictionary for bayesian optimization
        self.search_space_bs = {
            'C': ['loguniform', 2.0 ** -5, 2.0 ** 15],
            'kernel': ['categorical', ['linear', 'sigmoid', 'rbf']], # TODO: add poly kernel
            'gamma': ['uniform', 0, 20]
        }

        #parameter dictionary for grid search (might give error if dataset too small)
        self.search_space_gs = [{
            'kernel'      : ['rbf', 'sigmoid'],
            'C'           : [0.001,0.01,0.1,1,10,100,1000],
            'gamma'       : [0.001,0.01,0.1,1,10,100,1000]
            },
            {
            'kernel'      : ['linear'],
            'C'           : [0.001,0.01,0.1,1,10,100,1000]
            },
            {
            'kernel'      : ['poly'],
            'C'           : [0.001,0.01,0.1,1,10,100,1000],
            'gamma'       : [0.001,0.01,0.1,1,10,100,1000],
            'degree'      : [1,2,3,4,5]
            }]
        super().__init__(data, self.alg, self.parameters, self.search_space_bs, self.search_space_gs, save_m=save_m)

class KNN(QSARModel):
    """ K-nearest neighbor regressor and classifier initialization. Here the model instance is created 
        and parameters and search space can be defined
        ...

        Attributes
        ----------
        data: instance of QSARDataset
        parameters (dict): KNN specific parameters
        save_m (bool): if true, save final model

    """
    def __init__(self, data, save_m=True, parameters=None):
        self.alg = KNeighborsRegressor() if data.reg else KNeighborsClassifier()
        self.parameters=parameters if parameters != None else {}
        #parameter dictionary for bayesian optimization
        self.search_space_bs = {
            'n_neighbors': ['int', 1, 100],
            'weights': ['categorical', ['uniform', 'distance']],
            'metric': ['categorical', ["euclidean","manhattan",
                        "chebyshev","minkowski"]]
        }
        #parameter dictionary for grid search
        self.search_space_gs = {
            'n_neighbors' : list(range(1,31)),
            'weights'      : ['uniform', 'distance']
            }
        super().__init__(data, self.alg, self.parameters, self.search_space_bs, self.search_space_gs, save_m=save_m)

class NB(QSARModel):
    """ 
        Gaussian Naive Bayes model initialization. Here the model instance is created 
        and parameters and search space can be defined.
        ...

        Attributes
        ----------
        data: instance of QSARDataset
        parameters (dict): NB specific parameters
        save_m (bool): if true, save final model

    """
    def __init__(self, data, save_m=True, parameters=None):
        if data.reg:
            raise ValueError("NB should be constructed only with classification.")
        self.alg = GaussianNB()
        self.parameters=parameters if parameters != None else {}
        #parameter dictionaries for hyperparameter optimization
        self.search_space_bs = {
            'var_smoothing': ['loguniform', 1e-10, 1]
            }

        self.search_space_gs = {
            'var_smoothing': np.logspace(0,-9, num=100)
        }
        super().__init__(data, self.alg, self.parameters, self.search_space_bs, self.search_space_gs, save_m=save_m)

class PLS(QSARModel):
    """ 
        PLS Regression model initialization. Here the model instance is created 
        and parameters and search space can be defined
        ...

        Attributes
        ----------
        data: instance of QSARDataset
        parameters (dict): PLS specific parameters
        save_m (bool): if true, save final model

    """
    def __init__(self, data, save_m=True, parameters=None):
        if not data.reg:
            raise ValueError("PLS should be constructed only with regression.")
        self.alg = PLSRegression()
        self.parameters=parameters if parameters != None else {}
        self.search_space_bs = {
            'n_components': ['int', 1, 100],
            'scale': ['categorical', [True, False]]
            }
        self.search_space_gs = {
            'n_components': list(range(1, 100, 20))
        }
        super().__init__(data, self.alg, self.parameters, self.search_space_bs, self.search_space_gs,  save_m=save_m)

class DNN(QSARModel):
    """ 
        This class holds the methods for training and fitting a Deep Neural Net QSAR model initialization. 
        Here the model instance is created and parameters  can be defined
        ...

        Attributes
        ----------
        data: instance of QSARDataset
        parameters (dict): DNN specific parameters
        save_m (bool): if true, save final model
        batch_size (int): batch size
        lr (int): learning rate
        n_epoch (int): number of epochs

    """
    def __init__(self, data, save_m=True, parameters=None, batch_size=128, lr=1e-5, n_epoch=1000):
        self.alg = models.STFullyConnected
        self.parameters = parameters if parameters != None else {}
        super().__init__(data, self.alg, self.parameters, None, None, save_m=save_m)
        self.batch_size = batch_size
        self.lr = lr
        self.n_epoch = n_epoch
        self.y = self.data.y.reshape(-1,1)
        self.y_ind = self.data.y_ind.reshape(-1,1)

    def init_model(self):
        pass

    def fit_model(self):
        train_set = TensorDataset(torch.Tensor(self.data.X), torch.Tensor(self.y))
        train_loader = DataLoader(train_set, batch_size=self.batch_size)
        valid_set = TensorDataset(torch.Tensor(self.data.X_ind), torch.Tensor(self.y_ind))
        valid_loader = DataLoader(valid_set, batch_size=self.batch_size)
        net = self.alg(self.data.X.shape[1], self.y.shape[1], is_reg=self.data.reg)
        net.fit(train_loader, valid_loader, out=self.out, epochs=self.n_epoch, lr=self.lr)

    def model_evaluation(self):
        #Make predictions for crossvalidation and independent test set
        indep_set = TensorDataset(torch.Tensor(self.data.X_ind), torch.Tensor(self.y_ind))
        indep_loader = DataLoader(indep_set, batch_size=self.batch_size)
        cvs = np.zeros(self.y.shape)
        inds = np.zeros(self.y_ind.shape)
        for i, (trained, valided) in enumerate(self.data.folds):
            log.info('cross validation fold ' +  str(i))
            train_set = TensorDataset(torch.Tensor(self.data.X[trained]), torch.Tensor(self.y[trained]))
            train_loader = DataLoader(train_set, batch_size=self.batch_size)
            valid_set = TensorDataset(torch.Tensor(self.data.X[valided]), torch.Tensor(self.y[valided]))
            valid_loader = DataLoader(valid_set, batch_size=self.batch_size)
            net = self.alg(self.data.X.shape[1], self.y.shape[1], is_reg=self.data.reg)
            net.fit(train_loader, valid_loader, out='%s_%d' % (self.out, i), epochs=self.n_epoch, lr=self.lr)
            cvs[valided] = net.predict(valid_loader)
            inds += net.predict(indep_loader)
        train, test = pd.Series(self.y.flatten()).to_frame(name='Label'), pd.Series(self.y_ind.flatten()).to_frame(name='Label')
        train['Score'], test['Score'] = cvs, inds / 5
        train.to_csv(self.out + '.cv.tsv', sep='\t')
        test.to_csv(self.out + '.ind.tsv', sep='\t')
        self.data.create_folds()

    def grid_search(self):
        #TODO implement grid search for DNN
        log.warning("Grid search not yet implemented for DNN, will be skipped.")
    
    def bayes_optimization(self, n_trials):
        #TODO implement bayes optimization for DNN
        log.warning("bayes optimization not yet implemented for DNN, will be skipped.")


def EnvironmentArgParser(txt=None):
    """ 
        Define and read command line arguments
    """
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains a folder 'data' with input files")
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-ran', '--random_state', type=int, default=1, help="Seed for the random state")
    parser.add_argument('-i', '--input', type=str, default='dataset',
                        help="tsv file name that contains SMILES, target accession & corresponding data")
    parser.add_argument('-s', '--save_model', action='store_true',
                        help="If included then then the model will be trained on all data and saved")   
    parser.add_argument('-m', '--model_types', type=str, nargs='*', default=['RF', 'XGB', 'DNN', 'SVM', 'PLS', 'NB', 'KNN', 'MT_DNN'],
                        help="Modeltype, defaults to run all modeltypes, choose from: 'RF', 'XGB', 'DNN', 'SVM', 'PLS' (only with REG), 'NB' (only with CLS) 'KNN' or 'MT_DNN'") 
    parser.add_argument('-r', '--regression', type=str, default=None,
                        help="If True, only regression model, if False, only classification, default both")
    parser.add_argument('-t', '--targets', type=str, nargs='*', default=None, #TODO: maybe change this to all accession in the dataset?
                        help="Target indentifiers") 
    parser.add_argument('-a', '--activity_threshold', type=float, default=6.5,
                        help="Activity threshold")
    parser.add_argument('-l', '--keep_low_quality', action='store_true',
                        help="If included keeps low quality data")
    parser.add_argument('-y', '--year', type=int, default=None,
                        help="Temporal split limit")  
    parser.add_argument('-n', '--test_size', type=str, default="0.1",
                        help="Random test split fraction if float is given and absolute size if int is given, used when no temporal split given.")
    parser.add_argument('-o', '--optimization', type=str, default=None,
                        help="Hyperparameter optimization, if None no optimization, if grid gridsearch, if bayes bayesian optimization")    
    parser.add_argument('-c', '--model_evaluation', action='store_true',
                        help='If on, model evaluation through cross validation and independent test set is performed.')
    parser.add_argument('-ncpu', '--ncpu', type=int, default=8,
                        help="Number of CPUs")
    parser.add_argument('-gpu', '--gpu', type=str, default='1,2,3,4',
                        help="List of GPUs") 
    parser.add_argument('-bs', '--batch_size', type=int, default=2048,
                        help="Batch size for DNN")
    parser.add_argument('-e', '--epochs', type=int, default=1000,
                        help="Number of epochs for DNN")
    parser.add_argument('-ng', '--no_git', action='store_true',
                        help="If on, git hash is not retrieved") 
    
    if txt:
        args = parser.parse_args(txt)
    else:
        args = parser.parse_args()
    
    if args.targets is None:
        df = pd.read_table('%s/data/%s' % (args.base_dir, args.input)).dropna(subset=['SMILES'])
        args.targets = df.accession.unique().tolist()

    # If no regression argument, does both regression and classification
    if args.regression is None: 
        args.regression = [True, False]
    elif args.regression.lower() == 'true':
        args.regression = [True]
    elif args.regression.lower() == 'false':
        args.regression = [False]
    else:
        sys.exit("invalid regression arg given")

    if '.' in args.test_size:
        args.test_size = float(args.test_size) 
    else: 
        args.test_size = int(args.test_size)

    return args


def Environ(args):
    """ 
        Optimize, evaluate and train estimators
    """
    args.devices = eval(args.gpu) if ',' in args.gpu else [eval(args.gpu)]
    torch.cuda.set_device(DEFAULT_GPUS[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.environ['OMP_NUM_THREADS'] = str(args.ncpu)
    
    if not os.path.exists(args.base_dir + '/envs'):
        os.makedirs(args.base_dir + '/envs') 
    
    for reg in args.regression:
        args.learning_rate = 1e-4 if reg else 1e-5 
        for target in args.targets:
            #prepare dataset for training QSAR model
            mydataset = QSARDataset(args.base_dir, args.input, target, reg = reg, timesplit=args.year, test_size=args.test_size, th = args.activity_threshold, keep_low_quality=args.keep_low_quality)
            mydataset.split_dataset()
            
            for model_type in args.model_types:
                if model_type == 'MT_DNN': print('MT DNN is not implemented yet')
                elif model_type not in ['RF', 'XGB', 'DNN', 'SVM', 'PLS', 'NB', 'KNN']: 
                    log.warning(f'Model type {model_type} does not exist')
                    continue
                if model_type == 'NB' and reg:
                    log.warning("NB with regression invalid, skipped.")
                    continue
                if model_type == 'PLS' and not reg:
                    log.warning("PLS with classification invalid, skipped.")
                    continue

                #Create QSAR model object
                mymodel_class = getattr(sys.modules[__name__], model_type)
                mymodel = mymodel_class(mydataset)

                #if desired run parameter optimization
                if args.optimization == 'grid':
                    mymodel.grid_search()
                elif args.optimization == 'bayes':
                    mymodel.bayes_optimization(n_trials=20)
                
                #initialize models from saved or default parameters
                mymodel.init_model(n_jobs=args.ncpu)

                if args.optimization is None and args.save_model:
                    mymodel.fit_model()
                
                if args.model_evaluation:
                    mymodel.model_evaluation()

               
if __name__ == '__main__':
    args = EnvironmentArgParser()
    
    #Set random seeds
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    os.environ['TF_DETERMINISTIC_OPS'] = str(args.random_state)

    # Backup files
    tasks = [ 'REG' if reg == True else 'CLS' for reg in args.regression ]
    file_prefixes = [ f'{alg}_{task}_{target}' for alg in args.model_types for task in tasks for target in args.targets]
    backup_msg = backUpFiles(args.base_dir, 'envs', tuple(file_prefixes), cp_suffix='_params')

    if not os.path.exists(f'{args.base_dir}/envs'):
        os.mkdir(f'{args.base_dir}/envs')

    logSettings = enable_file_logger(
        os.path.join(args.base_dir, 'envs'),
        'environ.log',
        args.debug,
        __name__,
        commit_hash(os.path.dirname(os.path.realpath(__file__))) if not args.no_git else None,
        vars(args)
    )   

    log = logSettings.log
    log.info(backup_msg)

    # # Get logger, include this in every module
    # log = logging.getLogger(__name__)

    #Add optuna logging
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    # Create json log file with used commandline arguments 
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    with open(f'{args.base_dir}/envs/environ.json', 'w') as f:
        json.dump(vars(args), f)
        
    # logSettings for DNN model
    #TODO: set this only for DNN model, instead of global
    BATCH_SIZE = args.batch_size
    N_EPOCH = args.epochs
    
    #Optimize, evaluate and train estimators according to environment arguments
    Environ(args)
