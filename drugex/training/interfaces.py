"""
interfaces

Created by: Martin Sicho
On: 01.06.22, 11:29
"""
import os
from abc import ABC, abstractmethod

import torch
from torch import nn

from drugex import DEFAULT_DEVICE_ID, DEFAULT_DEVICE
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


class Model(nn.Module, ABC):
    """
    Generic base class for all PyTorch models in DrugEx. Manages the GPU or CPU devices available to the model.
    """

    def __init__(self):
        super().__init__()
        self.device = None
        self.devices = None
        self.attachToDevice(DEFAULT_DEVICE)
        self.attachToDevices([DEFAULT_DEVICE_ID])

    def attachToDevice(self, device):
        """
        Attach this model to the given device.

        Args:
            device: result of "torch.device('cpu')" or "torch.device('gpu')"

        Returns:
            `None`
        """

        self.device = device

    def attachToDevices(self, device_ids):
        """
        Attach this model to multiple devices by giving their device ids.

        Args:
            device_ids: a `list` of devices to use for calculations

        Returns:

        """

        self.devices = device_ids

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

        self.load_state_dict(torch.load(path, map_location=self.device), strict=False)

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

class Explorer(Model, ABC):
    """
    Implements the DrugEx exploration strategy for DrugEx models under the reinforcement learning framework.
    """

    def __init__(self, agent, env, mutate=None, crover=None, batch_size=128, epsilon=0.1, sigma=0.0, n_samples=-1, repeat=1):
        super().__init__()
        self.batchSize = batch_size
        self.epsilon = epsilon
        self.sigma = sigma
        self.repeat = repeat
        self.env = env
        self.agent = agent
        self.mutate = mutate
        self.crover = crover
        self.nSamples = n_samples

    @abstractmethod
    def fit(self, train_loader, valid_loader=None, epochs=1000, monitor=None):
        pass

class ModelProvider(ABC):
    """
    Any instance that contains a DrugEx `Model`.
    """

    @abstractmethod
    def getModel(self):
        """
        Return the current model.

        Returns:
            model (`Model`)
        """

        pass

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

class Trainer(ModelProvider, ABC):
    """
    A convenience class that unifies training of the DrugEx models. Mostly to hold information and settings about the CPU and GPU devices used.
    """

    def __init__(self, algorithm, gpus=(DEFAULT_DEVICE_ID,)):
        """
        Direct the training of a DrugEx `Model`.

        Args:
            algorithm: the initialized `Model` to train
            gpus: IDs of GPUs to use for training.
        """

        assert len(gpus) > 0
        self.availableGPUs = gpus
        # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in self.availableGPUs)
        self.device = None
        self.deviceID = None
        self.model = algorithm
        self.attachDevices()

    def loadModel(self, provider):
        """
        Load a model from provider.

        Args:
            provider: a `ModelProvider`
        """

        self.model = provider.getModel()

    def getModel(self):
        """
        Get the currently trained model.

        Returns:
            model (`Model`)
        """

        return self.model

    def getDevices(self):
        """
        Get the list of used GPUs.
        """

        return self.availableGPUs

    def attachDevices(self, device_id=DEFAULT_DEVICE_ID, device=DEFAULT_DEVICE):
        """
        Attach the specified devices to the underlying model.

        Args:
            device_id: ID of the device
            device: either "torch.device('cpu')" or "torch.device('gpu')"

        Returns:
            device: currently set device
        """

        if device_id and (device_id not in self.availableGPUs):
            raise RuntimeError(f"Unavailable device: {device_id}")
        if not device_id:
            device_id = self.availableGPUs[0]
        torch.cuda.set_device(device_id)
        self.device = torch.device(device)
        self.deviceID = device_id
        self.model.attachToDevice(self.device)
        self.model.attachToDevices(self.availableGPUs)
        return self.device

    @abstractmethod
    def fit(self, train_loader, valid_loader=None, training_monitor=None, epochs=None, args=None, kwargs=None):
        """
        Custom fit method.

        Args:
            train_loader: loader with training data (see `DataSet`)
            valid_loader: loader with validation data (see `DataSet`)
            training_monitor: `TrainingMonitor` to use
            epochs: maximum number of epochs to train
            args: custom positional arguments for the `Model.fit()` method
            kwargs: custom keyword arguments for the `Model.fit()` method

        Returns:
            output: the output of `Model.fit()` of the algorithm in "self.model"
        """

        pass

    def loadStatesFromFile(self, path):
        """
        Load model states from the given path.

        Args:
            path: file path
        """

        self.model.loadStatesFromFile(path)
