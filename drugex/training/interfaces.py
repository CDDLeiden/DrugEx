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

    @abstractmethod
    def __call__(self, mols, frags=None):
        pass

class ScoreModifier:
    """
    Interface for score modifiers.
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

    def __init__(self, modifier=None):
        self.modifier = modifier

    @abstractmethod
    def getScores(self, mols, frags=None):
        """
        Returns scores for input data.

        Parameters
        ----------
        data
            Data used by model to make a prediction.

        Returns
        -------

        """

        pass

    def __call__(self, mols, frags=None):
        return self.getModifiedScores(self.getScores(mols, frags))

    def getModifiedScores(self, scores):
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

    def __init__(self, device=DEFAULT_DEVICE):
        self.device = device

    def getParetoFronts(self, scores):
        if self.device == torch.device('cuda'):
            swarm = torch.Tensor(scores).to(self.device)
            return gpu_non_dominated_sort(swarm)
        else:
            return cpu_non_dominated_sort(scores)

    @abstractmethod
    def __call__(self, smiles, scores):
        pass


class RewardScheme(ABC):

    class RewardException(Exception):
        pass

    def __init__(self, ranking=None):
        self.ranking = ranking

    @abstractmethod
    def __call__(self, smiles, scores, valid, desire, undesire, thresholds):
        pass

class Environment(ABC):

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

    def __call__(self, smiles, is_modified=True, frags=None):
        return self.getScores(smiles, is_modified, frags)

    @abstractmethod
    def getScores(self, smiles, is_modified=True, frags=None):
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

    def __init__(self):
        super().__init__()
        self.device = None
        self.devices = None
        self.attachToDevice(DEFAULT_DEVICE)
        self.attachToDevices([DEFAULT_DEVICE_ID])

    def attachToDevice(self, device):
        self.device = device

    def attachToDevices(self, devices):
        self.devices = devices

    @abstractmethod
    def fit(self, train_loader, valid_loader, epochs=1000, out=None):
        pass

    def loadStatesFromFile(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device), strict=False)

class Generator(Model, ABC):

    @abstractmethod
    def fit(self, train_loader, valid_loader, epochs=1000, evaluator=None, monitor=None):
        pass

    @abstractmethod
    def evaluate(self, n_samples, method=None, drop_duplicates=True):
        pass

class Explorer(Model, ABC):

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

    @abstractmethod
    def getModel(self):
        pass

class TrainingMonitor(ModelProvider, ABC):

    @abstractmethod
    def saveModel(self, model):
        pass

    @abstractmethod
    def savePerformanceInfo(self, current_step, current_epoch, loss, *args, **kwargs):
        pass

    @abstractmethod
    def saveProgress(self, current_step, current_epoch, total_steps, total_epochs, *args, **kwargs):
        pass

    @abstractmethod
    def endStep(self, step, epoch):
        pass

    @abstractmethod
    def close(self):
        pass

class Trainer(ModelProvider, ABC):

    def __init__(self, algorithm, gpus=(DEFAULT_DEVICE_ID,)):
        assert len(gpus) > 0
        self.availableGPUs = gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in self.availableGPUs)
        self.device = None
        self.deviceID = None
        self.model = algorithm
        self.attachDevices()

    def loadModel(self, provider):
        self.model = provider.getModel()

    def getModel(self):
        return self.model

    def getDevices(self):
        return self.availableGPUs

    def attachDevices(self, device_id=DEFAULT_DEVICE_ID, device=DEFAULT_DEVICE):
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
        pass

    def loadStatesFromFile(self, path):
        self.model.loadStatesFromFile(path)
