import os
import sys
from abc import ABC, abstractmethod

import numpy as np
import json

from drugex.logs import logger
from drugex.environment.neural_network import STFullyConnected
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR

# IDEA:  since this abstract class assumes that the implementing class will use either bayes_optimization or grid_search, it might make sense to define these abstract methods as well to make sure they really exist when they are invoked from the environ.py script or otherwise
# IDEA (optional): I think even better would be to remove this from the QSARModel and move it to i.e. a separate HyperParamOptim class that does this (same with the evaluation maybe), but let's save this for later
# IDEA (optional): the definition of the model could also be independent of the data -> so incorporating the idea above we would separately initialize QSARDataset, QSARModel, HyperParamOptim and ModelEvaluation instances and then we would call fit on the QSARModel instance giving it the QSARDataset and HyperParamOptim to obtain the best model, which we would then supply to the evaluate method of ModelEvaluation -> but this is religiously following separation of concerns and in our small case it probably does not matter much yet. But just imagine we want to add more hyperparameter optimization techniques or evaluation mechanisms: it would soon become cumbersome to add all these new methods to all implementations of QSARModel and then also add their calls to environ.py, but if we have it separated and encapsulated then we just add new implementations of HyperParamOptim and ModelEvaluation and just plug those into the workflow very easily (in theory)
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

        # IDEA: would it maybe make more sense to set the parameters of the actual model in the subclasses and let this class only implement the general acquisition of model parameters to self.parameters? It seems this class really does not have to care about sklearn or anything like that -> it would be the responsibility of the subclass to parse and initialize self.model instances appropriate from the parameters given.
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


    # IDEA: could we simply call this method 'fit'?
    @abstractmethod
    def fit_model(self):
        """
            build estimator model from entire data set
        """
        pass

    # IDEA: could we simply call this method 'evaluate'?
    @abstractmethod
    def model_evaluation(self, save=True):
        """
            Make predictions for crossvalidation and independent test set
            arguments:
                save (bool): don't save predictions when used in bayesian optimization
        """
        pass
    
    # IDEA: I would change this to the camelCase style, most methods and class members in DrugEx are like that so we could keep the same style (I would gradually try to refactor all methods and other class members to use this)
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
