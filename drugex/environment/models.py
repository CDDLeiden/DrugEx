from drugex.logs import logger
import os
import os.path
import sys
import json
import numpy as np
from datetime import datetime
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
import optuna
from sklearn import metrics
import torch
from torch.utils.data import DataLoader, TensorDataset
from drugex.environment import models

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
    def __init__(self, data, alg, parameters=None, n_jobs=-1):
        """
            initialize model from saved or default hyperparameters
        """
        self.data = data
        self.alg = alg
        self.parameters = parameters

        d = '%s/envs' % data.base_dir
        self.out = '%s/%s_%s_%s' % (d, self.__class__.__name__, 'REG' if data.reg else 'CLS', data.target)
        
        if os.path.isfile('%s_params.json' % self.out):
            
            with open('%s_params.json' % self.out) as j:
                parameters = json.loads(j.read())
            logger.info('loaded model parameters from file: %s_params.json' % self.out)

        self.model = self.alg.set_params(n_jobs=n_jobs, **parameters)
        logger.info('parameters: %s' % self.parameters)
        logger.debug('Model intialized: %s' % self.out)

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
            logger.info('cross validation fold %s ended: %s' % (i, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        #save crossvalidation results
        if save:
            train, test = pd.Series(self.data.y).to_frame(name='Label'), pd.Series(self.data.y_ind).to_frame(name='Label')
            train['Score'], test['Score'] = cvs, inds / 5
            train.to_csv(self.out + '.cv.tsv', sep='\t')
            test.to_csv(self.out + '.ind.tsv', sep='\t')

        self.data.create_folds()

        return cvs

    def grid_search(self, search_space_gs, save_m):
        """
            optimization of hyperparameters using grid_search
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
        
        self.model = grid.best_estimator_
        
        if save_m:
            joblib.dump(self.model, '%s.pkg' % self.out, compress=3)

        self.data.create_folds()

        logger.info('Grid search best parameters: %s' % grid.best_params_)
        with open('%s_params.json' % self.out, 'w') as f:
            json.dump(grid.best_params_, f)

    def bayes_optimization(self, search_space_bs, n_trials):
        """
            bayesian optimization of hyperparameters using optuna
        """
        print('Bayesian optimization can take a while for some hyperparameter combinations')
        #TODO add timeout function
        study = optuna.create_study(direction='maximize')
        logger.info('Bayesian optimization started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        study.optimize(lambda trial: self.objective(trial, search_space_bs), n_trials)
        logger.info('Bayesian optimization ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        trial = study.best_trial

        self.model = self.alg.set_params(**trial.params)
        
        if self.save_m:
            joblib.dump(self.model, '%s.pkg' % self.out, compress=3)

        self.data.create_folds()

        logger.info('Bayesian optimization best params: %s' % trial.params)
        with open('%s_params.json' % self.out, 'w') as f:
            json.dump(trial.params, f)

    def objective(self, trial, search_space_bs):
        """
            objective for bayesian optimization
            arguments:
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
            elif value[0] == 'loggeruniform':
                bayesian_params[key] = trial.suggest_loggeruniform(key, value[1], value[2])
            elif value[0] == 'uniform':
                bayesian_params[key] = trial.suggest_uniform(key, value[1], value[2])

        self.model = self.alg.set_params(**bayesian_params)

        if self.data.reg: 
            score = metrics.explained_variance_score(self.data.y, self.model_evaluation(save = False))
        else:
            score = metrics.roc_auc_score(self.data.y, self.model_evaluation(save = False))

        return score
    
    @staticmethod
    def load_params_grid(fname, optim_type, model_types):
        """
            Load parameter grids for bayes or grid search parameter optimization from json file
            fname (str): file name of json file containing array with three columns containing modeltype, optimization
                         type (grid or bayes) and model type
            optim_type (str): optimization type ('grid' or 'bayes')
            model_types (list of str): model type for hyperparameter optimization (e.g. RF)
        """

        if fname:
            try:
                with open(fname) as json_file:
                    optim_params = np.array(json.load(json_file), dtype=object)
            except:
                logger.error("Search space file (%s) not found" % fname)
                sys.exit()
        else:
            with open('drugex/environment/search_space.json') as json_file:
                optim_params = np.array(json.load(json_file), dtype=object)
        
        # select either grid or bayes optimization parameters from param array
        optim_params = optim_params[optim_params[:,2]==optim_type, :]
        
        #check all modeltypes to be used have parameter grid
        if not set(list(model_types)).issubset(list(optim_params[:,0])):
            logger.error("model types %s missing from models in search space dict (%s)" % (model_types, optim_params[:,0]))
            sys.exit()
        logger.info("search space loaded from file")
        
        return optim_params

class QSARDNN(QSARModel):
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
            logger.info('cross validation fold ' +  str(i))
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
        logger.warning("Grid search not yet implemented for DNN, will be skipped.")
    
    def bayes_optimization(self, n_trials):
        #TODO implement bayes optimization for DNN
        logger.warning("bayes optimization not yet implemented for DNN, will be skipped.")
