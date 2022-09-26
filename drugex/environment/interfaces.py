import os
import os.path as op
import sys
from abc import ABC, abstractmethod

import numpy as np
import json

from drugex.logs import logger
from drugex.environment.neural_network import STFullyConnected
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR


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
        fit: build estimator model from entire data set
        objective: objective used by bayesian optimization
        bayesOptimization: bayesian optimization of hyperparameters using optuna
        gridSearch: optimization of hyperparameters using gridSearch

    """
    def __init__(self, base_dir, data, alg, alg_name, parameters={}):
        """
            initialize model from saved or default hyperparameters
        """
        self.data = data
        self.alg = alg
        self.parameters = parameters
        self.alg_name = alg_name

        d = '%s/envs' % base_dir
        self.out = '%s/%s_%s_%s' % (d, alg_name, 'REG' if data.reg else 'CLS', data.target)
        
        if os.path.isfile('%s_params.json' % self.out):    
            with open('%s_params.json' % self.out) as j:
                if self.parameters:
                    self.parameters = json.loads(j.read()).update(self.parameters)
                else:
                    self.parameters = json.loads(j.read())
            logger.info('loaded model parameters from file: %s_params.json' % self.out)


    @abstractmethod
    def fit(self):
        """
            build estimator model from entire data set
        """
        pass

    @abstractmethod
    def evaluate(self, save=True):
        """
            Make predictions for crossvalidation and independent test set
            arguments:
                save (bool): don't save predictions when used in bayesian optimization
        """
        pass

    @abstractmethod
    def gridSearch(self):
        """
            optimization of hyperparameters using gridSearch
            arguments:
                search_space_gs (dict): search space for the grid search
                save_m (bool): if true, after gs the model is refit on the entire data set
        """          
        pass

    @abstractmethod
    def bayesOptimization(self):
        """
            bayesian optimization of hyperparameters using optuna
            arguments:
                search_space_gs (dict): search space for the grid search
                n_trials (int): number of trials for bayes optimization
                save_m (bool): if true, after bayes optimization the model is refit on the entire data set
        """
        pass
    
    @staticmethod
    def loadParamsGrid(fname, optim_type, model_types):
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
            with open(op.join(op.dirname(__file__), "search_space.json")) as json_file:
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
