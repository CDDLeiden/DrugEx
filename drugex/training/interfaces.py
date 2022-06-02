"""
interfaces

Created by: Martin Sicho
On: 01.06.22, 11:29
"""
import os
from abc import ABC, abstractmethod

import torch
from torch import nn

class Scorer(ABC):

    @abstractmethod
    def __call__(self, data):
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

    @abstractmethod
    def getKey(self):
        pass

    @abstractmethod
    def setModifier(self, score_modifier):
        pass

    @abstractmethod
    def setThreshold(self):
        pass

class Environment(ABC):

    def __init__(self, scorers):
        self.scorers = scorers

    @abstractmethod
    def __call__(self, smiles, is_modified=True, frags=None):
        pass


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

class Explorer(Model, ABC):

    def __init__(self, agent, prior, env, batch_size=128, epsilon=0.1, sigma=0.0, scheme='PR', repeat=1):
        super().__init__()
        self.batchSize = batch_size
        self.epsilon = epsilon
        self.sigma = sigma
        self.scheme = scheme
        self.repeat = repeat
        self.env = env
        self.agent = agent
        self.prior = prior

    @abstractmethod
    def fit(self, train_loader, valid_loader=None, epochs=1000, out=None):
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
    def saveProgress(self, current_step, current_epoch, total_steps, total_epochs, *args, **kwargs):
        pass

class Trainer(ModelProvider, ABC):

    def __init__(self, algorithm, gpus=(0,)):
        assert len(gpus) > 0
        self.availableGPUs = gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(self.availableGPUs)
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
        torch.cuda.set_device(device_id)
        self.device = torch.device('cuda')
        self.deviceID = device_id
        self.model.attachToDevice(self.device)
        self.model.attachToDevices(self.availableGPUs)
        return self.device

    @abstractmethod
    def fit(self, train_loader, valid_loader=None, training_monitor=None, epochs=None, args=None, kwargs=None):
        pass





