import os
import sys
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import json

from drugex.logs import logger
from sklearn.naive_bayes import GaussianNB
from drugex.environment.classifier import STFullyConnected
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from xgboost import XGBRegressor, XGBClassifier

class QSARModel(ABC):
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
    def __init__(self, base_dir, data, alg, alg_name, parameters=None, n_jobs = 1):
        """
            initialize model from saved or default hyperparameters
        """
        self.data = data
        self.alg = alg
        self.parameters = parameters

        d = '%s/envs' % base_dir
        self.out = '%s/%s_%s_%s' % (d, alg_name, 'REG' if data.reg else 'CLS', data.target)
        
        if os.path.isfile('%s_params.json' % self.out) and not self.parameters:    
            with open('%s_params.json' % self.out) as j:
                self.parameters = json.loads(j.read())
            logger.info('loaded model parameters from file: %s_params.json' % self.out)

        if self.parameters:
            if type(self.alg) in [GaussianNB, PLSRegression, SVR, SVC, STFullyConnected]:
                self.model = self.alg.set_params(**self.parameters)
            else:
                self.model = self.alg.set_params(n_jobs=n_jobs, **self.parameters)
        else:
            if type(self.alg) in [GaussianNB, PLSRegression, SVR, SVC, STFullyConnected]:
                self.model = self.alg
            else:
                self.model = self.alg.set_params(n_jobs=n_jobs)
        logger.info('parameters: %s' % self.parameters)
        logger.debug('Model intialized: %s' % self.out)


    @abstractmethod
    def fit_model(self):
        """
            build estimator model from entire data set
        """
        pass

    @abstractmethod
    def model_evaluation(self, save=True):
        """
            Make predictions for crossvalidation and independent test set
            arguments:
                save (bool): don't save predictions when used in bayesian optimization
        """
        pass
    
    @staticmethod
    def load_params_grid(fname, optim_type, model_types):
        """
            Load parameter grids for bayes or grid search parameter optimization from json file
            arguments:
                fname (str): file name of json file containing array with three columns containing modeltype,
                             optimization type (grid or bayes) and model type
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
        model_types = [model_types] if type(model_types) == str else model_types

        if not set(list(model_types)).issubset(list(optim_params[:,0])):
            logger.error("model types %s missing from models in search space dict (%s)" % (model_types, optim_params[:,0]))
            sys.exit()
        logger.info("search space loaded from file")
        
        return optim_params
