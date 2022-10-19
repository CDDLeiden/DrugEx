from drugex.logs import logger
from drugex import DEFAULT_DEVICE, DEFAULT_GPUS

import sys
import os
import os.path
import json
import numpy as np
from datetime import datetime
import joblib
import pandas as pd
import optuna
import math

from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
from drugex.environment.interfaces import QSARModel
from drugex.environment.neural_network import STFullyConnected


class QSARsklearn(QSARModel):
    """ Model initialization, fit, cross validation and hyperparameter optimization for classifion/regression models.
        ...

        Attributes
        ----------
        data: instance QSARDataset
        alg:  instance of estimator
        parameters (dict): dictionary of algorithm specific parameters
        njobs (int): the number of parallel jobs to run
        
        Methods
        -------
        init_model: initialize model from saved hyperparameters
        fit: build estimator model from entire data set
        objective: objective used by bayesian optimization
        bayesOptimization: bayesian optimization of hyperparameters using optuna
        gridSearch: optimization of hyperparameters using gridSearch

    """
    def __init__(self, base_dir, data, alg, alg_name, parameters=None, n_jobs = 1):

        super().__init__(base_dir, data, alg, alg_name, parameters=parameters)
        
        #initialize models with defined parameters
        if self.parameters:
            if type(self.alg) in [GaussianNB, PLSRegression, SVR, SVC]:
                self.model = self.alg.set_params(**self.parameters)
            else:
                self.model = self.alg.set_params(n_jobs=n_jobs, **self.parameters)
        else:
            if type(self.alg) in [SVC, SVR]:
                logger.warning("parameter max_iter set to 10000 to avoid training getting stuck. \
                                 Manually set this parameter if this is not desired.")
                self.model = self.alg.set_params(max_iter=10000)
            elif type(self.alg) in [GaussianNB, PLSRegression]:
                self.model = self.alg
            else:
                self.model = self.alg.set_params(n_jobs=n_jobs)
    
        logger.info('parameters: %s' % self.parameters)
        logger.debug('Model intialized: %s' % self.out)

    def fit(self):
        """
            build estimator model from entire data set
        """
        X_all = np.concatenate([self.data.X, self.data.X_ind], axis=0)
        y_all = np.concatenate([self.data.y, self.data.y_ind], axis=0)
        
        # KNN and PLS do not use sample_weight
        fit_set = {'X': X_all}

        # weighting in original drugex v2 code, but was specific to data used there
        # use sample weight to decrease the weight of low quality datapoints
        # if type(self.alg).__name__ not in ['KNeighborsRegressor', 'KNeighborsClassifier', 'PLSRegression']:
        #     fit_set['sample_weight'] = [1 if v >= 4 else 0.1 for v in self.data.y[trained]]
        
        if type(self.alg).__name__ == 'PLSRegression':
            fit_set['Y'] = y_all
        else:
            fit_set['y'] = y_all

        logger.info('Model fit started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))  
        self.model.fit(**fit_set)
        logger.info('Model fit ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        joblib.dump(self.model, '%s.pkg' % self.out, compress=3)

    def evaluate(self, save=True):
        """
            Make predictions for crossvalidation and independent test set
            arguments:
                save (bool): don't save predictions when used in bayesian optimization
        """
        cvs = np.zeros(self.data.y.shape)
        inds = np.zeros(self.data.y_ind.shape)

        # cross validation
        for i, (trained, valided) in enumerate(self.data.folds):
            logger.info('cross validation fold %s started: %s' % (i, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            fit_set = {'X':self.data.X[trained]}

            # weighting in original drugex v2 code, but was specific to data used there
            # use sample weight to decrease the weight of low quality datapoints
            # if type(self.alg).__name__ not in ['KNeighborsRegressor', 'KNeighborsClassifier', 'PLSRegression']:
            #     fit_set['sample_weight'] = [1 if v >= 4 else 0.1 for v in self.data.y[trained]]
            
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
            logger.info('cross validation fold %s ended: %s' % (i, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        # fitting on whole trainingset and predicting on test set
        fit_set = {'X': self.data.X}
            
        if type(self.alg).__name__ == 'PLSRegression':
            fit_set['Y'] = self.data.y
        else:
            fit_set['y'] = self.data.y

        self.model.fit(**fit_set)
        
        if type(self.alg).__name__ == 'PLSRegression':
            inds = self.model.predict(self.data.X_ind)[:, 0]
        elif self.data.reg:
            inds = self.model.predict(self.data.X_ind)
        else:
            inds = self.model.predict_proba(self.data.X_ind)[:, 1]

        #save crossvalidation results
        if save:
            train, test = pd.Series(self.data.y).to_frame(name='Label'), pd.Series(self.data.y_ind).to_frame(name='Label')
            train['Score'], test['Score'] = cvs, inds
            train.to_csv(self.out + '.cv.tsv', sep='\t')
            test.to_csv(self.out + '.ind.tsv', sep='\t')

        self.data.createFolds()

        return cvs

    def gridSearch(self, search_space_gs):
        """
            optimization of hyperparameters using gridSearch
            arguments:
                search_space_gs (dict): search space for the grid search
        """          
        scoring = 'explained_variance' if self.data.reg else 'roc_auc'    
        grid = GridSearchCV(self.alg, search_space_gs, n_jobs=10, verbose=1, cv=self.data.folds,
                            scoring=scoring, refit=False)
        
        fit_set = {'X': self.data.X, 'y': self.data.y}
        # if type(self.alg).__name__ not in ['KNeighborsRegressor', 'KNeighborsClassifier', 'PLSRegression']:
        #     fit_set['sample_weight'] = [1 if v >= 4 else 0.1 for v in self.data.y]
        logger.info('Grid search started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        grid.fit(**fit_set)
        logger.info('Grid search ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        self.data.createFolds()

        logger.info('Grid search best parameters: %s' % grid.best_params_)
        with open('%s_params.json' % self.out, 'w') as f:
            json.dump(grid.best_params_, f)

        self.model = self.alg.set_params(**grid.best_params_)

    def bayesOptimization(self, search_space_bs, n_trials):
        """
            bayesian optimization of hyperparameters using optuna
            arguments:
                search_space_gs (dict): search space for the grid search
                n_trials (int): number of trials for bayes optimization
                save_m (bool): if true, after bayes optimization the model is refit on the entire data set
        """
        print('Bayesian optimization can take a while for some hyperparameter combinations')
        #TODO add timeout function
        study = optuna.create_study(direction='maximize')
        logger.info('Bayesian optimization started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        study.optimize(lambda trial: self.objective(trial, search_space_bs), n_trials)
        logger.info('Bayesian optimization ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        trial = study.best_trial

        self.data.createFolds()

        logger.info('Bayesian optimization best params: %s' % trial.params)
        with open('%s_params.json' % self.out, 'w') as f:
            json.dump(trial.params, f)

        self.model = self.alg.set_params(**trial.params)

    def objective(self, trial, search_space_bs):
        """
            objective for bayesian optimization
            arguments:
                trial (int): current trial number
                search_space_bs (dict): search space for bayes optimization
        """

        bayesian_params = {}

        for key, value in search_space_bs.items():
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

        print(bayesian_params)
        self.model = self.alg.set_params(**bayesian_params)

        if self.data.reg: 
            score = metrics.explained_variance_score(self.data.y, self.evaluate(save = False))
        else:
            score = metrics.roc_auc_score(self.data.y, self.evaluate(save = False))

        return score

class QSARDNN(QSARModel):
    """ 
        This class holds the methods for training and fitting a Deep Neural Net QSAR model initialization. 
        Here the model instance is created and parameters  can be defined
        ...

        Attributes
        ----------
        data: instance of QSARDataset
        parameters (dict): dictionary of parameters to set for model fitting
        device (cuda device): cuda device
        gpus (int/ list of ints): gpu number(s) to use for model fitting
        patience (int): number of epochs to wait before early stop if no progress on validiation set score
        tol (float): minimum absolute improvement of loss necessary to count as progress on best validation score

    """
    def __init__(self, base_dir, data, parameters = None, device=DEFAULT_DEVICE, gpus=DEFAULT_GPUS, patience = 50, tol = 0):

        
        super().__init__(base_dir, data, STFullyConnected(n_dim=data.X.shape[1], device=device, gpus=gpus, is_reg=data.reg),
                         "DNN", parameters=parameters)

        self.patience = patience
        self.tol = tol

        #Initialize model with defined parameters
        if self.parameters:
                self.model = self.alg.set_params(**self.parameters)
        else:
                self.model = self.alg
        logger.info('parameters: %s' % self.parameters)
        logger.debug('Model intialized: %s' % self.out)

        #transpose y data to column vector
        self.y = self.data.y.reshape(-1,1)
        self.y_ind = self.data.y_ind.reshape(-1,1)

        self.optimal_epochs = -1

    def fit(self):
        """
            train model on the trainings data, determine best model using test set, save best model

            ** IMPORTANT: evaluate should be run first, so that the average number of epochs from the cross-validation
                          with early stopping can be used for fitting the model.
        """
        if self.optimal_epochs == -1:
            logger.error('Cannot fit final model without first determining the optimal number of epochs for fitting. \
                          first run evaluate.')
            sys.exit()

        X_all = np.concatenate([self.data.X, self.data.X_ind], axis=0)
        y_all = np.concatenate([self.y, self.y_ind], axis=0)

        self.model = self.model.set_params(**{"n_epochs" : self.optimal_epochs})
        train_loader = self.model.get_dataloader(X_all, y_all)

        logger.info('Model fit started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.model.fit(train_loader, None, self.out, patience = -1)
        joblib.dump(self.model, '%s.pkg' % self.out, compress=3)
        logger.info('Model fit ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def evaluate(self, save=True, ES_val_size=0.1):
        """
            Make predictions for crossvalidation and independent test set
            arguments:
                save (bool): wether to save the cross validation predictions
                ES_val_size (float): validation set size for early stopping in CV
        """
        indep_loader = self.model.get_dataloader(self.data.X_ind)
        last_save_epoch = 0

        cvs = np.zeros(self.y.shape)
        inds = np.zeros(self.y_ind.shape)
        for i, (trained, valided) in enumerate(self.data.folds):
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(self.data.X[trained], self.y[trained], test_size=ES_val_size)
            logger.info('cross validation fold ' +  str(i))
            train_loader = self.model.get_dataloader(X_train_fold, y_train_fold)
            ES_valid_loader = self.model.get_dataloader(X_val_fold, y_val_fold)
            valid_loader = self.model.get_dataloader(self.data.X[valided])
            last_save_epoch += self.model.fit(train_loader, ES_valid_loader, '%s_temp' % self.out, self.patience, self.tol)
            os.remove('%s_temp_weights.pkg' % self.out)
            cvs[valided] = self.model.predict(valid_loader)
        os.remove('%s_temp.log' % self.out)

        if save:
            self.optimal_epochs = int(math.ceil(last_save_epoch / self.data.n_folds)) + 1
            self.model = self.model.set_params(**{"n_epochs" : self.optimal_epochs})

            train_loader = self.model.get_dataloader(self.data.X, self.y)
            self.model.fit(train_loader, None, '%s_temp' % self.out, patience = -1)
            os.remove('%s_temp_weights.pkg' % self.out)
            os.remove('%s_temp.log' % self.out)
            inds = self.model.predict(indep_loader)


            train, test = pd.Series(self.y.flatten()).to_frame(name='Label'), pd.Series(self.y_ind.flatten()).to_frame(name='Label')
            train['Score'], test['Score'] = cvs, inds
            train.to_csv(self.out + '.cv.tsv', sep='\t')
            test.to_csv(self.out + '.ind.tsv', sep='\t')
        self.data.createFolds()

        return cvs

    def gridSearch(self, search_space_gs, ES_val_size=0.1):
        """
            optimization of hyperparameters using gridSearch
            arguments:
                search_space_gs (dict): search space for the grid search, accepted parameters are:
                                        lr (int) ~ learning rate for fitting
                                        batch_size (int) ~ batch size for fitting
                                        n_epochs (int) ~ max number of epochs
                                        neurons_h1 (int) ~ number of neurons in first hidden layer
                                        neurons_hx (int) ~ number of neurons in other hidden layers
                                        extra_layer (bool) ~ whether to add extra (3rd) hidden layer
                save_m (bool): if true, after gs the model is refit on the entire data set
                ES_val_size (float): validation set size for early stopping in CV
        """          
        scoring = metrics.explained_variance_score if self.data.reg else metrics.roc_auc_score

        logger.info('Grid search started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        best_score = -np.inf
        for params in ParameterGrid(search_space_gs):
            logger.info(params)

            #do 5 fold cross validation and take mean prediction on validation set as score of parameter settings
            fold_scores = np.zeros(self.data.n_folds)
            for i, (trained, valided) in enumerate(self.data.folds):
                logger.info('cross validation fold ' +  str(i))
                X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(self.data.X[trained], self.y[trained], test_size=ES_val_size)
                train_loader = self.model.get_dataloader(X_train_fold, y_train_fold)
                ES_valid_loader = self.model.get_dataloader(X_val_fold, y_val_fold)
                valid_loader = self.model.get_dataloader(self.data.X[valided])
                self.model.set_params(**params)
                self.model.fit(train_loader, ES_valid_loader, '%s_temp' % self.out, self.patience, self.tol)
                os.remove('%s_temp_weights.pkg' % self.out)
                y_pred = self.model.predict(valid_loader)
                fold_scores[i] = scoring(self.y[valided], y_pred)
            os.remove('%s_temp.log' % self.out)
            param_score = np.mean(fold_scores)
            if param_score >= best_score:
                best_params = params
                best_score = param_score
            self.data.createFolds()
        
        logger.info('Grid search best parameters: %s' %  best_params)
        with open('%s_params.json' % self.out, 'w') as f:
            json.dump(best_params, f)
        logger.info('Grid search ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        self.model = self.alg.set_params(**best_params)

        self.data.createFolds()
    
    def bayesOptimization(self, search_space_bs, n_trials):
        """
            bayesian optimization of hyperparameters using optuna
            arguments:
                search_space_gs (dict): search space for the grid search
                n_trials (int): number of trials for bayes optimization
                save_m (bool): if true, after bayes optimization the model is refit on the entire data set
        """
        print('Bayesian optimization can take a while for some hyperparameter combinations')
        #TODO add timeout function
        study = optuna.create_study(direction='maximize')
        logger.info('Bayesian optimization started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        study.optimize(lambda trial: self.objective(trial, search_space_bs), n_trials)
        logger.info('Bayesian optimization ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        trial = study.best_trial

        self.data.createFolds()

        logger.info('Bayesian optimization best params: %s' % trial.params)
        with open('%s_params.json' % self.out, 'w') as f:
            json.dump(trial.params, f)

        self.model = self.alg.set_params(**trial.params)

    def objective(self, trial, search_space_bs):
        """
            objective for bayesian optimization
            arguments:
                trial (int): current trial number
                search_space_bs (dict): search space for bayes optimization
        """

        bayesian_params = {}

        for key, value in search_space_bs.items():
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
            score = metrics.explained_variance_score(self.data.y, self.evaluate(save = False))
        else:
            score = metrics.roc_auc_score(self.data.y, self.evaluate(save = False))

        return score

