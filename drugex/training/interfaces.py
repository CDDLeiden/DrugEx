from abc import ABC, abstractmethod

import numpy as np
from typing import Literal

import torch
from torch import nn

from drugex import DEFAULT_GPUS, DEFAULT_DEVICE
from drugex.logs import logger


class ModelEvaluator(ABC):
    """
    A simple function to score a model based on the generated molecules and input fragments if applicable.
    """

    @abstractmethod
    def __call__(self, mols, frags=None):
        """
        Score molecules.

        Parameters
        ----------
        mols : list
            List of SMILES strings of the molecules to score.
        frags : list, optional
            List of SMILES strings of the fragments used to generate the molecules.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the scores for the molecules.
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



    @abstractmethod
    def __call__(self, smiles, scores, thresholds):
        """
        Calculate the rewards for generated molecules and rank them according to teh given `RankingStrategy`.

        Parameters
        ----------
        smiles : list
            List of SMILES strings of the molecules to rank.
        scores : np.ndarray
            Matrix of scores for the multiple objectives
        thresholds : list
            List of thresholds for the objectives.

        Returns
        -------
        np.ndarray
            Array of rewards for the molecules.
        """
        pass

class Environment(ModelEvaluator):
    """
    Definition of the generic environment class for DrugEx. Reference implementation is `DrugExEnvironment`.
    """

    def __init__(self, scorers, thresholds=None, reward_scheme=None):
        """
        Initialize environment with objective functions and their desirability thresholds. Molecules scoring above thresholds are desirable.

        Parameters
        ----------
        scorers : list
            List of objective functions to use for scoring molecules.
        thresholds : list, optional
            List of desirability thresholds for the objective functions.
            If `None`, all thresholds are set to 0.99.
        reward_scheme : RewardScheme, optional
            The reward scheme to use for ranking solutions. If `None`, the `DefaultRewardScheme` is used.
        
        Raises
        ------
        AssertionError
            If the number of scorers and thresholds does not match. In case of,
            multi-task scorer, a threshold should be provided for each task.
        
        Notes
        -----
        The `scorers` and `thresholds` are passed to the `RewardScheme` as well.
        """

        self.scorers = scorers
        num_scorers = len(self.getScorerKeys()) if isinstance(self.scorers, list) else 1
        self.thresholds = thresholds if thresholds is not None else [0.99] * num_scorers
        assert num_scorers == len(self.thresholds), f"Number of scorers ({num_scorers}) and thresholds ({len(self.thresholds)}) do not match."
        self.rewardScheme = reward_scheme

    def __call__(self, smiles, frags=None):
        """
        Score molecules.
        
        Parameters
        ----------
        smiles : list
            List of SMILES strings of the molecules to score.
        frags : list, optional
            List of SMILES strings of the fragments used to generate the molecules.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame with the scores for the molecules."""
        return self.getScores(smiles, frags)

    @abstractmethod
    def getScores(self, smiles, frags=None):
        """
        Calculate the scores of all objectives per molecule and qualify generated molecules (valid, accurate, desired).

        Parameters
        ----------
        smiles : list
            List of SMILES strings of the molecules to score.
        frags : list, optional
            List of SMILES strings of the fragments used to generate the molecules.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame with the scores and qualifications for the molecules.
        """
        pass

    @abstractmethod
    def getUnmodifiedScores(self, smiles):
        """
        Calculate the scores without applying modifiers of all objectives per molecule.

        Parameters
        ----------
        smiles : list
            List of SMILES strings of the molecules to score.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame with the scores for the molecules.
        """
        pass

    def getRewards(self, smiles, frags=None):
        """
        Calculate the single value as the reward for each molecule used for reinforcement learning.

        Parameters
        ----------
        smiles : list
            List of SMILES strings of the molecules to score.
        frags : list, optional
            List of SMILES strings of the fragments used to generate the molecules.
        
        Returns
        -------
        np.ndarray
            Array of rewards for the molecules.
        """

        # Get scores
        scores = self.getScores(smiles, frags=frags)
        scores['SMILES'] = smiles  

        # Initialize rewards to 0
        rewards = np.zeros((len(smiles),1))
        valid_idx = scores[scores.Valid == 1].index.tolist()
        # Compute rewards for valid molecules
        if len(valid_idx) > 0:
            smiles_valid = scores[scores.Valid == 1].SMILES.tolist()
            scores_valid = scores.loc[scores.Valid == 1, self.getScorerKeys()].values
            rewards_valid = self.rewardScheme(smiles_valid, scores_valid, self.thresholds)
            rewards[valid_idx] = rewards_valid
        else:
            logger.warning("No valid molecules generated. All rewards are 0.")

        return rewards

    def getScorerKeys(self):
        """
        Get the keys of the scorers.

        Returns
        -------
        list
            List of keys of the scorers.
        """
        keys = []
        for scorer in self.scorers:
            scorer_keys = scorer.getKey()
            if isinstance(scorer_keys, list):
                keys.extend(scorer_keys)
            else:
                keys.append(scorer_keys)
        return keys

class ModelProvider(ABC):
    """
    Any instance that contains a DrugEx `Model` or its serialized form (i.e a state dictionary).
    """

    @abstractmethod
    def getModel(self):
        """
        Return the current model as a `Model` instance or in serialized form.

        Returns
        -------
        Model or dict
            The current model or its serialized form.
        """

        pass

class Model(nn.Module, ModelProvider, ABC):
    """
    Generic base class for all PyTorch models in DrugEx. Manages the GPU or CPU gpus available to the model.
    """

    def __init__(self, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS):
        super().__init__()
        """
        Initialize the model with the given device and GPUs.

        Parameters
        ----------
        device : torch.device, optional
            The device to use for the model. If `None`, the default device is used.
        use_gpus : list, optional
            List of GPUs to use for the model. If `None`, the default GPUs are used.
        """

        self.device = None
        self.gpus = None
        self.updateDevices(device, use_gpus)

    def updateDevices(self, device, gpus):
        """
        Update the device and GPUs used by the model.

        Parameters
        ----------
        device : torch.device
            The device to use for the model.
        gpus : list
            List of GPUs to use for the model.
        """

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

        Parameters
        ----------
        gpus : tuple
            Tuple of GPU ids to use.
        """
        pass

    @abstractmethod
    def fit(self, train_loader, valid_loader, epochs=1000, monitor=None, **kwargs):
        """
        Train and validate the model with a given training and validation loader (see `DataSet` and its implementations docs to learn how to generate them).

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            The training data loader.
        valid_loader : torch.utils.data.DataLoader
            The validation data loader.
        epochs : int, optional
            The number of epochs to train the model for.
        monitor : TrainingMonitor, optional
            A `TrainingMonitor` instance to monitor the training process.     
        **kwargs
            Additional keyword arguments to pass to the training loop.  
        """
        pass

    def loadStatesFromFile(self, path):
        """
        Load the model states from a file.

        Parameters
        ----------
        path : str
            The path to the file containing the model states.
        """

        self.loadStates(torch.load(path, map_location=self.device))


    def loadStates(self, state_dict, strict=True):
        """
        Load the model states from a dictionary.

        Parameters
        ----------
        state_dict : dict
            The dictionary containing the model states.
        strict : bool, optional
            Whether to raise an error if the dictionary contains keys that do not match the model.
        """
        self.load_state_dict(state_dict, strict=strict)

class TrainingMonitor(ModelProvider, ABC):
    """
    Interface used to monitor model training.
    """

    @abstractmethod
    def saveModel(self, model, identifier=None):
        """
        Save the state dictionary of the `Model` instance currently being trained or serialize the model any other way.

        Parameters
        ----------
        model : Model
            The model to save.
        identifier : str
            Suffix added to saved model.
        """
        pass

    @abstractmethod
    def savePerformanceInfo(self, performance_dict, df_smiles=None):
        """
        Save the performance data for the current epoch.

        Parameters
        ----------
        performance_dict : dict
            A dictionary with the performance data.
        df_smiles : pd.DataFrame
            A DataFrame with the SMILES of the molecules generated in the current epoch.
        """
        pass

    @abstractmethod
    def saveProgress(self, model: Model, current_step=None, current_epoch=None, total_steps=None, total_epochs=None, *args, **kwargs):
        """
        Notifies the monitor of the current progress of the training.

        Parameters
        ----------
        model : Model
            The model being trained.
        current_step : int, optional
            The current training step (i.e. batch).
        current_epoch : int, optional
            The current epoch.
        total_steps : int, optional
            The total number of training steps.
        total_epochs : int, optional
            The total number of epochs.
        *args
            Additional arguments depending on the model type.
        **kwargs
            Additional keyword arguments depending on the model type.
        """
        pass

    @abstractmethod
    def endStep(self, step, epoch):
        """
        Notify the monitor that a step of the training has finished.

        Parameters
        ----------
        step : int
            The current training step (i.e. batch).
        epoch : int
            The current epoch.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close this monitor. Training has finished.
        """
        pass

    @abstractmethod
    def getSaveModelOption(self) -> Literal['best', 'all', 'improvement']:
        """
        Return the scheme implemented by the monitor to save model snapshots.

        Returns
        -------
        Literal['best', 'all', 'improvement']
            The scheme implemented by the monitor to save model snapshots.
        """
        pass