"""
interfaces

Created by: Martin Sicho
On: 01.06.22, 11:29
"""
import os
from abc import ABC, abstractmethod

import torch
from torch import nn

class ModelEvaluator(ABC):

    @abstractmethod
    def __call__(self, mols, frags=None):
        pass

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

class Environment(ABC):

    def __init__(self, scorers, thresholds=None):
        self.scorers = scorers
        self.thresholds = thresholds if thresholds is not None else [0.99] * len(scorers)
        assert len(self.scorers) == len(self.thresholds)

    def __call__(self, smiles, is_modified=True, frags=None):
        return self.getScores(smiles, is_modified, frags)

    @abstractmethod
    def getScores(self, smiles, is_modified=True, frags=None):
        pass

    @abstractmethod
    def getRewards(self, smiles, scheme='WS', frags=None):
        pass

    def getScorerKeys(self):
        return [x.getKey() for x in self.scorers]


class Model(nn.Module, ABC):

    def __init__(self):
        super().__init__()
        self.device = None
        self.devices = None
        self.attachToDevice(0)
        self.attachToDevices([0])

    def attachToDevice(self, device):
        self.device = device

    def attachToDevices(self, devices):
        self.devices = devices

    @abstractmethod
    def fit(self, train_loader, valid_loader=None, epochs=1000, out=None):
        pass

class Generator(Model, ABC):

    @abstractmethod
    def fit(self, train_loader, valid_loader=None, epochs=1000, method=None, out=None):
        pass

    @abstractmethod
    def evaluate(self, n_samples, method=None, drop_duplicates=True):
        pass

class Explorer(Model, ABC):

    def __init__(self, agent, env, mutate=None, crover=None, batch_size=128, epsilon=0.1, sigma=0.0, scheme='PR', repeat=1):
        super().__init__()
        self.batchSize = batch_size
        self.epsilon = epsilon
        self.sigma = sigma
        self.scheme = scheme
        self.repeat = repeat
        self.env = env
        self.agent = agent
        self.mutate = mutate
        self.crover = crover

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

    def __init__(self, algorithm, gpus=(0,)):
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

    def attachDevices(self, device_id=None):
        if device_id and (device_id not in self.availableGPUs):
            raise RuntimeError(f"Unavailable device: {device_id}")
        if not device_id:
            device_id = self.availableGPUs[0]
        torch.cuda.set_device(device_id)
        self.device = torch.device('cuda')
        self.deviceID = device_id
        self.model.attachToDevice(self.device)
        self.model.attachToDevices(self.availableGPUs)
        return self.device

    @abstractmethod
    def fit(self, train_loader, valid_loader=None, training_monitor=None, epochs=None, args=None, kwargs=None):
        pass




