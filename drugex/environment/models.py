from drugex.environment.interfaces import QSARModel
from drugex.logs import logger
import os
import os.path
import json
import numpy as np
from datetime import datetime
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
import optuna
from sklearn import metrics
import torch

from sklearn.model_selection import ParameterGrid
from drugex.environment.interfaces import QSARModel
from drugex.environment.classifier import STFullyConnected

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
        fit_model: build estimator model from entire data set
        objective: objective used by bayesian optimization
        bayes_optimization: bayesian optimization of hyperparameters using optuna
        grid_search: optimization of hyperparameters using grid_search

    """
    def fit_model(self):
        """
            build estimator model from entire data set
        """
        X_all = np.concatenate([self.data.X, self.data.X_ind], axis=0)
        y_all = np.concatenate([self.data.y, self.data.y_ind], axis=0)
        
        # KNN and PLS do not use sample_weight
        fit_set = {'X': X_all}
        
        if type(self.alg).__name__ == 'PLSRegression':
            fit_set['Y'] = y_all
        else:
            fit_set['y'] = y_all

        logger.info('Model fit started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))  
        self.model.fit(**fit_set)
        logger.info('Model fit ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        joblib.dump(self.model, '%s.pkg' % self.out, compress=3)

    def model_evaluation(self, save=True):
        """
            Make predictions for crossvalidation and independent test set
            arguments:
                save (bool): don't save predictions when used in bayesian optimization
        """
        cvs = np.zeros(self.data.y.shape)
        inds = np.zeros(self.data.y_ind.shape)
        for i, (trained, valided) in enumerate(self.data.folds):
            logger.info('cross validation fold %s started: %s' % (i, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            # use sample weight to decrease the weight of low quality datapoints
            fit_set = {'X':self.data.X[trained]}
            # weighting in original drugex v2 code, but was specific to data used there
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
        
        #save crossvalidation results
        if save:
            train, test = pd.Series(self.data.y).to_frame(name='Label'), pd.Series(self.data.y_ind).to_frame(name='Label')
            train['Score'], test['Score'] = cvs, inds / 5
            train.to_csv(self.out + '.cv.tsv', sep='\t')
            test.to_csv(self.out + '.ind.tsv', sep='\t')

        self.data.create_folds()

        return cvs

    def grid_search(self, search_space_gs, save_m=True):
        """
            optimization of hyperparameters using grid_search
            arguments:
                search_space_gs (dict): search space for the grid search
                save_m (bool): if true, after gs the model is refit on the entire data set
        """          
        scoring = 'explained_variance' if self.data.reg else 'roc_auc'    
        grid = GridSearchCV(self.alg, search_space_gs, n_jobs=10, verbose=1, cv=self.data.folds,
                            scoring=scoring, refit=save_m)
        
        fit_set = {'X':self.data.X}
        fit_set['y'] = self.data.y
        if type(self.alg).__name__ not in ['KNeighborsRegressor', 'KNeighborsClassifier', 'PLSRegression']:
            fit_set['sample_weight'] = [1 if v >= 4 else 0.1 for v in self.data.y]
        logger.info('Grid search started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        grid.fit(**fit_set)
        logger.info('Grid search ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        if save_m:
            self.model = grid.best_estimator_
            joblib.dump(self.model, '%s.pkg' % self.out, compress=3)

        self.data.create_folds()

        logger.info('Grid search best parameters: %s' % grid.best_params_)
        with open('%s_params.json' % self.out, 'w') as f:
            json.dump(grid.best_params_, f)

    def bayes_optimization(self, search_space_bs, n_trials, save_m):
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

        if save_m:
            self.model = self.alg.set_params(**trial.params)
            joblib.dump(self.model, '%s.pkg' % self.out, compress=3)

        self.data.create_folds()

        logger.info('Bayesian optimization best params: %s' % trial.params)
        with open('%s_params.json' % self.out, 'w') as f:
            json.dump(trial.params, f)

    def objective(self, trial, search_space_bs):
        """
            objective for bayesian optimization
            arguments:
                trial (int): current trial number
                search_space_bs (dict): search space for bayes optimization
        """

        if type(self.alg).__name__ in ['XGBRegressor', 'XGBClassifier']:
            bayesian_params = {'verbosity': 0}
        else:
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
                bayesian_params[key] = trial.suggest_loggeruniform(key, value[1], value[2])
            elif value[0] == 'uniform':
                bayesian_params[key] = trial.suggest_uniform(key, value[1], value[2])

        self.model = self.alg.set_params(**bayesian_params)

        if self.data.reg: 
            score = metrics.explained_variance_score(self.data.y, self.model_evaluation(save = False))
        else:
            score = metrics.roc_auc_score(self.data.y, self.model_evaluation(save = False))

        return score

class QSARDNN(QSARModel):
    """ 
        This class holds the methods for training and fitting a Deep Neural Net QSAR model initialization. 
        Here the model instance is created and parameters  can be defined
        ...

        Attributes
        ----------
        data: instance of QSARDataset
        batch_size (int): batch size
        lr (int): learning rate
        n_epoch (int): number of epochs

    """
    def __init__(self, base_dir, data, parameters = None):
        
        super().__init__(base_dir, data, STFullyConnected(n_dim=data.X.shape[1]), "DNN", parameters=parameters)
        
        #transpose y data to column vector
        self.y = self.data.y.reshape(-1,1)
        self.y_ind = self.data.y_ind.reshape(-1,1)

    def fit_model(self):
        """
            train model on the trainings data, determine best model using test set, save best model
        """
        train_loader = self.model.get_dataloader(self.data.X, self.y)
        indep_loader = self.model.get_dataloader(self.data.X_ind, self.y_ind)

        logger.info('Model fit started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.model.fit(train_loader, indep_loader, out=self.out)
        logger.info('Model fit ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def model_evaluation(self, save=True):
        """
            Make predictions for crossvalidation and independent test set
        """
        indep_loader = self.model.get_dataloader(self.data.X_ind, self.y_ind)
        
        cvs = np.zeros(self.y.shape)
        inds = np.zeros(self.y_ind.shape)
        for i, (trained, valided) in enumerate(self.data.folds):
            logger.info('cross validation fold ' +  str(i))
            train_loader = self.model.get_dataloader(self.data.X[trained], self.y[trained])
            valid_loader = self.model.get_dataloader(self.data.X[valided], self.y[valided])
            self.model.fit(train_loader, valid_loader, out='%s_%d' % (self.out, i))
            cvs[valided] = self.model.predict(valid_loader)
            inds += self.model.predict(indep_loader)
        
        if save:
            train, test = pd.Series(self.y.flatten()).to_frame(name='Label'), pd.Series(self.y_ind.flatten()).to_frame(name='Label')
            train['Score'], test['Score'] = cvs, inds / 5
            train.to_csv(self.out + '.cv.tsv', sep='\t')
            test.to_csv(self.out + '.ind.tsv', sep='\t')
        self.data.create_folds()

    def grid_search(self, search_space_gs, save_m):
        """
            optimization of hyperparameters using grid_search
            arguments:
                search_space_gs (dict): search space for the grid search, accepted parameters are:
                                        lr (int) ~ learning rate for fitting
                                        batch_size (int) ~ batch size for fitting
                                        n_epochs (int) ~ max number of epochs
                                        neurons_h1 (int) ~ number of neurons in first hidden layer
                                        neurons_hx (int) ~ number of neurons in other hidden layers
                                        extra_layer (bool) ~ whether to add extra (3rd) hidden layer
                save_m (bool): if true, after gs the model is refit on the entire data set
        """          
        scoring = metrics.explained_variance_score if self.data.reg else metrics.roc_auc_score

        logger.info('Grid search started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        best_score = 0
        for params in ParameterGrid(search_space_gs):
            logger.info(params)

            #do 5 fold cross validation and take mean prediction on validation set as score of parameter settings
            fold_scores = np.zeros(self.data.n_folds)
            for i, (trained, valided) in enumerate(self.data.folds):
                logger.info('cross validation fold ' +  str(i))
                train_loader = self.model.get_dataloader(self.data.X[trained], self.y[trained])
                valid_loader = self.model.get_dataloader(self.data.X[valided], self.y[valided])
                self.model.set_params(**params)
                self.model.fit(train_loader, valid_loader, out='%s_temp' % self.out)
                os.remove('%s_temp.pkg' % self.out)
                y_pred = self.model.predict(valid_loader)
                fold_scores[i] = scoring(self.y[valided], y_pred)
            param_score = np.mean(fold_scores)
            if param_score > best_score:
                best_params = params
            self.data.create_folds()
        
        logger.info('Grid search best parameters: %s' %  best_params)
        with open('%s_params.json' % self.out, 'w') as f:
            json.dump(best_params, f)
        logger.info('Grid search ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        if save_m:
            self.model.set_params(**best_params)
            self.model.fit()

        self.data.create_folds()
    
    def bayes_optimization(self, n_trials):
        #TODO implement bayes optimization for DNN
        logger.warning("bayes optimization not yet implemented for DNN, will be skipped.")


