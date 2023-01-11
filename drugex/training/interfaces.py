"""
interfaces

Created by: Martin Sicho
On: 01.06.22, 11:29
"""
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from scipy.stats import gmean

import torch
from torch import nn

from drugex import DEFAULT_GPUS, DEFAULT_DEVICE
from drugex.logs import logger
from drugex.utils import gpu_non_dominated_sort, cpu_non_dominated_sort



class ModelEvaluator(ABC):
    """
    A simple function to score a model based on the generated molecules and input fragments if applicable.
    """

    @abstractmethod
    def __call__(self, mols, frags=None):
        """
        Score molecules.

        Args:
            mols: molecules to score
            frags: given input fragments

        Returns:
            scores (DataFrame): a data frame with columns name 'VALID' and 'DESIRE' indicating the validity of the SMILES and the degree of desirability
        """

        pass

class ScoreModifier(ABC):
    """
    Defines a function to modify a score value.

    """
    @abstractmethod
    def __call__(self, x):
        """
        Apply the modifier on x.
        Args:
            x: float or np.array to modify
        Returns:
            float or np.array (depending on the type of x) after application of the distance function.
        """

class Scorer(ABC):
    """
    Used by the `Environment` to calculate customized scores.

    """

    def __init__(self, modifier=None):
        self.modifier = modifier

    @abstractmethod
    def getScores(self, mols, frags=None):
        """
        Returns scores for the input molecules.

        Args:
            mols: molecules to score
            frags: input fragments

        Returns:
            scores (list): `list` of scores for "mols"
        """

        pass

    def __call__(self, mols, frags=None):
        """
        Actual call method. Modifies the scores before returning them.

        Args:
            mols: molecules to score
            frags: input fragments

        Returns:
            scores (DataFrame): a data frame with columns name 'VALID' and 'DESIRE' indicating the validity of the SMILES and the degree of desirability
        """

        return self.getModifiedScores(self.getScores(mols, frags))

    def getModifiedScores(self, scores):
        """
        Modify the scores with the given `ScoreModifier`.

        Args:
            scores:

        Returns:

        """

        if self.modifier:
            return self.modifier(scores)
        else:
            return scores

    @abstractmethod
    def getKey(self):
        pass

    def setModifier(self, modifier):
        self.modifier = modifier

    def getModifier(self):
        return self.modifier

class RankingStrategy(ABC):
    """
    Ranks the given molecules according to their scores.

    The implementing classes can get a paretor from by calling `RankingStrategy.getParetoFronts()` on the input scores.

    """

    def __init__(self, device=DEFAULT_DEVICE):
        """
        Constructor allows to specify a GPU or CPU device for Pareto fronts calculation.

        Args:
            device:
        """

        self.device = device

    def getParetoFronts(self, scores):
        """
        Returns Pareto fronts.

        Args:
            scores: matrix of scores for the multiple objectives

        Returns:
            fronts (list): `list` of Pareto fronts.
        """

        if self.device == torch.device('cuda'):
            swarm = torch.Tensor(scores).to(self.device)
            return gpu_non_dominated_sort(swarm)
        else:
            return cpu_non_dominated_sort(scores)

    @abstractmethod
    def __call__(self, smiles, scores):
        """
        Return ranks of the molecules based on the given scores.

        Args:
            smiles: SMILES of the molecules
            scores: the matrix of scores ("len(smiles) x len(objectives)")

        Returns:

        """

        pass


class RewardScheme(ABC):
    """
    Reward scheme that enables ranking of molecules based on the calculated objectives and other criteria.
    """


    class RewardException(Exception):
        """
        Exception to catch errors in the calculation of rewards.
        """

        pass

    def __init__(self, ranking=None):
        """
        The `RankingStrategy` function to use for ranking solutions.

        Args:
            ranking: a `RankingStrategy`
        """

        self.ranking = ranking

    @abstractmethod
    def __call__(self, smiles, scores, valid, desire, undesire, thresholds):
        """
        Calculate the rewards for generated molecules and rank them according to teh given `RankingStrategy`.

        Args:
            smiles (list): SMILES strings of the generated molecules
            scores (DataFrame): the full scoring table ("len(smiles) x len(thresholds)")
            valid (list):  ratio of valid molecules
            desire (list): ratio of desired molecules
            undesire (list): ratio of undesired molecules
            thresholds (list): score thresholds for the calculated scores in "scores"

        Returns:
            rewards (list): rewards based on
        """

        pass

class Environment(ModelEvaluator):
    """
    Definition of the generic environment class for DrugEx. Reference implementation is `DrugExEnvironment`.
    """

    def __init__(self, scorers, thresholds=None, reward_scheme=None):
        """
        Initialize environment with objective functions and their desirability thresholds. Molecules scoring above thresholds are desirable.

        Args:
            scorers: scoring functions to calculate the objective functions for molecules
            thresholds: desirability thresholds for each scoring function (passed to the reward scheme as well), should be the same length as "scorers". If `None`, they will be set to 0.99.
            reward_scheme: a `RewardScheme`, predefined schemes are available in `drugex.training.rewards`
        """

        self.scorers = scorers
        self.thresholds = thresholds if thresholds is not None else [0.99] * len(scorers)
        assert len(self.scorers) == len(self.thresholds)
        self.rewardScheme = reward_scheme

    def __call__(self, smiles, frags=None):
        return self.getScores(smiles, frags)

    @abstractmethod
    def getScores(self, smiles, frags=None):
        """
        Calculate the scores of all objectives for all of samples
        Args:
            smiles (list): the list of generated molecules
            frags (list): the list of input fragments

        Returns:
            scores (DataFrame): The scores of all objectives for all of samples which also includes validity
                and desirability for each SMILES.
        """

        pass

    def getRewards(self, smiles, frags=None):
        """
        Calculate the single value as the reward for each molecule used for reinforcement learning
        Args:
            smiles (List):  a list of SMILES-based molecules

        Returns:
            rewards (np.ndarray): n-d array in which the element is the reward for each molecule, and
                n is the number of array which equals to the size of smiles.
        """
        scores = self.getScores(smiles, frags=frags)
        valid = scores.VALID.values
        desire = scores.DESIRE.sum()
        undesire = len(scores) - desire
        scores = scores[self.getScorerKeys()].values

        rewards = self.rewardScheme(smiles, scores, valid, desire, undesire, self.thresholds)
        rewards[valid == 0] = 0
        return rewards

    def getScorerKeys(self):
        return [x.getKey() for x in self.scorers]

class ModelProvider(ABC):
    """
    Any instance that contains a DrugEx `Model` or its serialized form (i.e a state dictionary).
    """

    @abstractmethod
    def getModel(self):
        """
        Return the current model as a `Model` instance or in serialized form.

        Returns:
            model (`Model` or serialized states)
        """

        pass

class Model(nn.Module, ModelProvider, ABC):
    """
    Generic base class for all PyTorch models in DrugEx. Manages the GPU or CPU gpus available to the model.
    """

    def __init__(self, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS):
        super().__init__()
        self.device = None
        self.gpus = None
        self.updateDevices(device, use_gpus)

    def updateDevices(self, device, gpus):
        if device.type == 'cpu':
            self.device = torch.device('cpu')
            self.gpus = (-1,)
        elif device.type == 'cuda':
            self.device = torch.device(f'cuda:{gpus[0]}')
            self.attachToGPUs(gpus)
        else:
            raise ValueError(f"Unknown device: {device}")

    @abstractmethod
    def attachToGPUs(self, gpus):
        """
        Use this method to handle a request to change the used GPUs. This method is automatically called when the class is instantiated, but may need to be called again in subclasses to move all data to the required devices.

        Subclasses should also make sure to set "self.device" to the currently used device and "self.gpus" to GPU ids of the currently used GPUs

        Args:
            gpus: a `tuple` of new GPU IDs

        Returns:
            `None`
        """

        pass

    @abstractmethod
    def fit(self, train_loader, valid_loader, epochs=1000, monitor=None):
        """
        Train and validate the model with a given training and validation loader (see `DataSet` and its implementations docs to learn how to generate them).

        Args:
            train_loader: PyTorch `DataLoader` with training data.
            valid_loader: PyTorch `DataLoader` with validation data.
            epochs: number of epochs for training the model
            monitor: a `TrainingMonitor`

        Returns:
            `None`
        """

        pass

    def loadStatesFromFile(self, path):
        """
        Load the model states from a file.

        Args:
            path: path to file
        """

        self.loadStates(torch.load(path, map_location=self.device))


    def loadStates(self, state_dict, strict=True):
        self.load_state_dict(state_dict, strict=strict)

class Generator(Model, ABC):
    """
    The base generator class for fitting and evaluating a DrugEx generator.
    """

    @abstractmethod
    def fit(self, train_loader, valid_loader, epochs=1000, evaluator=None, monitor=None):
        """
        Start training.

        Args:
            train_loader: training data loader (see `DataSet`)
            valid_loader: testing data loader (see `DataSet`)
            epochs: maximum number of epochs for which to train
            evaluator: a `ModelEvaluator`
            monitor:

        Returns:

        """

        pass

    @abstractmethod
    def evaluate(self, n_samples, method=None, drop_duplicates=True):
        pass

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass

class Explorer(Model, ABC):
    """
    Implements the DrugEx exploration strategy for DrugEx models under the reinforcement learning framework.
    """

    def __init__(self, agent, env, mutate=None, crover=None, batch_size=128, epsilon=0.1, sigma=0.0, n_samples=-1,
                 repeat=1, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS):
        super().__init__(device=device, use_gpus=use_gpus)
        self.agent = agent
        self.mutate = mutate
        self.crover = crover
        self.batchSize = batch_size
        self.epsilon = epsilon
        self.sigma = sigma
        self.repeat = repeat
        self.env = env
        self.nSamples = n_samples
        self.bestState = None
        self.best_value = 0
        self.last_save = -1
        self.last_iter = -1


    def attachToGPUs(self, gpus):
        if hasattr(self, 'agent'):
            self.agent.attachToGPUs(gpus)
        if hasattr(self, 'mutate') and self.mutate:
            self.mutate.attachToGPUs(gpus)
        if hasattr(self, 'crover') and self.crover:
            self.crover.attachToGPUs(gpus)
        self.gpus = gpus

    def getNovelMoleculeMetrics(self, scores):

        """ Get metrics for novel molecules

        Metrics:
            valid_ratio (float): ratio of valid molecules
            unique_ratio (float): ratio of valid and unique molecules
            desired_ratio (float): ratio of valid, unique and desired molecules
            aMeanScore (float): arithmetic mean score of valid and unique molecules
            gMeanScore (float): geometric mean score of valid and unique molecules
        
        Args:
            scores (pd.DataFrame): scores for each molecule
        Returns:
            dict: metrics
        """
        
        dct = {}
        ntot = len(scores) 
        # Valid compounds
        valid = scores[scores.VALID == 1]
        dct['valid_ratio'] = len(valid) / ntot
        # Unique compounds
        unique = valid.drop_duplicates(subset='Smiles')
        dct['unique_ratio'] = len(unique) / ntot
        # Desired compounds
        dct['desired_ratio'] = unique.DESIRE.sum() / ntot
        # Average artithmetic and geometric mean score 
        dct['aMeanScore'] = unique[self.env.getScorerKeys()].values.mean()
        dct['gMeanScore'] = unique[self.env.getScorerKeys()].apply(gmean, axis=1).mean()
        
        return dct

    def getCriteriaValue(self, scores, criteria):

        """ Get value of selection criteria

        Args:
            scores (pd.DataFrame): scores for each molecule
            criteria (str or function): selection criteria
        Returns:
            value (float): value of selection criteria
        """
        
        unique = scores[(scores.VALID == 1)].drop_duplicates(subset='Smiles')
        try:
            if criteria == 'desired_ratio': return unique.DESIRE.sum() / len(scores)
            elif criteria == 'amean_score': return unique[self.env.getScorerKeys()].values.mean()
            elif criteria == 'gmean_score': return unique[self.env.getScorerKeys()].apply(gmean, axis=1).mean()
            else: return criteria(scores)
        except:
            raise ValueError(f"Invalid criteria: {criteria}. Valid criteria are: 'desired_ratio', 'amean_score', gmean_score' or custom function with signature (scores: pd.DataFrame) -> float")

    def saveBestState(self, scores, criteria, epoch, it):

        """ Save best state based on selection criteria

        Args:
            scores (pd.DataFrame): scores for each molecule
            criteria (str or function): selection criteria
            epoch (int): current epoch
            it (int): current iteration
        Returns:
            value (float): value of selection criteria
        """
        
        value = self.getCriteriaValue(scores, criteria)
        if value > self.best_value:
            self.monitor.saveModel(self.agent)
            self.bestState = deepcopy(self.agent.state_dict())
            self.best_value = value
            self.last_save = epoch
            self.last_iter = it
            logger.info(f"Model saved at epoch {epoch}")

    def logPerformanceAndCompounds(self, epoch, epochs, scores):

        """ Log performance of model

        Args:
            scores (pd.DataFrame): scores for each molecule
            criteria (str): selection criteria
        """
        
        smiles_scores = list(scores.itertuples(index=False, name=None))
        smiles_scores_key = scores.columns.tolist()
        metrics = self.getNovelMoleculeMetrics(scores)
        self.monitor.savePerformanceInfo(None, epoch, None, smiles_scores=smiles_scores, smiles_scores_key=smiles_scores_key, **metrics)
        self.monitor.saveProgress(None, epoch, None, epochs)
        self.monitor.endStep(None, epoch)

    @abstractmethod
    def fit(self, train_loader, valid_loader=None, epochs=1000, monitor=None):
        pass

    def getModel(self):
        """
        Returns the current state of the agent

        Returns:

        """
        return deepcopy(self.agent.state_dict())

class TrainingMonitor(ModelProvider, ABC):
    """
    Interface used to monitor model training.
    """

    @abstractmethod
    def saveModel(self, model):
        """
        Save the state dictionary of the `Model` instance currently being trained or serialize the model any other way.

        Args:
            model: a DrugEx `Model`
        """
        pass

    @abstractmethod
    def savePerformanceInfo(self, current_step=None, current_epoch=None, loss=None, *args, **kwargs):
        """
        Save performance data.

        Args:
            current_step: Current training step (batch).
            current_epoch: Current epoch.
            loss: current value of the training loss
            *args: other arguments depending on the model type
            **kwargs: other keyword arguments depending on the model type
        """

        pass

    @abstractmethod
    def saveProgress(self, current_step=None, current_epoch=None, total_steps=None, total_epochs=None, *args, **kwargs):
        """
        Notifies the monitor of the current progress of the training.

        Args:
            current_step: Current training step (batch).
            current_epoch: Current epoch.
            total_steps: Total number of training steps (batches).
            total_epochs: Total number of epochs.
            *args: other arguments depending on the model type
            **kwargs: other keyword arguments depending on the model type
        """

        pass

    @abstractmethod
    def endStep(self, step, epoch):
        """
        Notify the monitor that a step of the training has finished.

        Args:
            step: Current training step (batch).
            epoch: Current epoch.
        """

        pass

    @abstractmethod
    def close(self):
        """
        Close this monitor. Training has finished.
        """

        pass
