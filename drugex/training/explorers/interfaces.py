"""
interfaces

Created by: Martin Sicho
On: 01.06.22, 11:29
"""
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from scipy.stats import gmean
from typing import List, Tuple, Union, Dict, Optional

import torch
from torch import nn
from tqdm import tqdm

from drugex import DEFAULT_GPUS, DEFAULT_DEVICE
from drugex.logs import logger
from drugex.training.interfaces import Model
from torch.utils.data import DataLoader
#from drugex.training.monitors import NullMonitor

class Explorer(Model, ABC):
    """
    Implements the DrugEx exploration strategy for DrugEx models under the reinforcement learning framework.
    """

    def __init__(self, agent, env, mutate=None, crover=None, batch_size=128, epsilon=0.1, beta=0.0, n_samples=-1,
                 repeat=1, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS):
        super().__init__(device=device, use_gpus=use_gpus)
        self.agent = agent
        self.mutate = mutate
        self.crover = crover
        self.batchSize = batch_size
        self.epsilon = epsilon
        self.beta = beta
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

    @abstractmethod
    def policy_gradient(self, *args, **kwargs):
        """
        Policy gradient training.
        """
        pass

    def getModel(self):
        """
        Returns the current state of the agent

        Returns:

        """
        return deepcopy(self.agent.state_dict())

class FragExplorer(Explorer):
    """
    Implements the DrugEx exploration strategy for DrugEx models under 
    the reinforcement learning framework for fragment-based generators
    """

    @abstractmethod
    def getBatchOutputs(self, src, net):
        """
        Outputs (frags, smiles) and loss of the agent for a batch of fragments-molecule pairs.
        """
        pass

    def policy_gradient(self, loader, no_multifrag_smiles=True):
        """
        Policy gradient training.
 
        Novel molecules are generated by the agent and are scored by the environment.
        The policy gradient is calculated using the REINFORCE algorithm and the agent is updated.
        
        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            Data loader for training data
        no_multifrag_smiles : bool
            If True, multi-fragment SMILES are not considered valid and reward is set to 0
        """

        net = nn.DataParallel(self.agent, device_ids=self.gpus)
        total_steps = len(loader)
        
        for step_idx, src in enumerate(tqdm(loader, desc='Iterating over validation batches', leave=False)):

            # Decode fragments and smiles, and get loss
            frags, smiles, loss = self.getBatchOutputs(net, src)
            
            # Get rewards
            reward = self.env.getRewards(smiles, frags=frags)

            # Filter out molecules with multiple fragments by setting reward to 0
            if self.no_multifrag_smiles:
                reward = [r if s.count('.') == 0 else [0] for s,r in zip(smiles, reward)]
            reward = torch.Tensor(reward).to(self.device)
            
            # Train model with policy gradient
            self.optim.zero_grad()
            loss = loss * ( reward - self.beta )
            loss = -loss.mean()
            loss.backward()
            self.optim.step()

            self.monitor.saveProgress(step_idx, None, total_steps, None)
            self.monitor.savePerformanceInfo(step_idx, None, loss.item())
            del loss

    @abstractmethod
    def sampleEncodedPairsToLoader(self, net, loader):
        """
        Sample new fragments-molecule pairs from a data loader.
        """
        pass

    @abstractmethod
    def sample_input(self, loader, is_test=False):
        
        """
        Sample a batch of fragments-molecule pairs from the training data loader.
        """
        pass

    def fit(self, train_loader, valid_loader=None, epochs=1000, patience=50, criteria='desired_ratio', min_epochs=100, monitor=None):
        
        """
        Fit the graph explorer to the training data.
        
        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Data loader for training data
        valid_loader : torch.utils.data.DataLoader
            Data loader for validation data
        epochs : int
            Number of epochs to train for
        patience : int
            Number of epochs to wait for improvement before early stopping
        criteria : str
            Criteria to use for early stopping
        min_epochs : int
            Minimum number of epochs to train for
        monitor : Monitor
            Monitor to use for logging and saving model
            
        Returns
        -------
        None
        """
        
        self.monitor = monitor if monitor else NullMonitor()
        self.bestState = deepcopy(self.agent.state_dict())
        self.monitor.saveModel(self.agent)

        n_iters = 1 if self.crover is None else 10
        net = nn.DataParallel(self, device_ids=self.gpus)
        logger.info(' ')
        
        for it in range(n_iters):
            if n_iters > 1:
                logger.info('\n----------\nITERATION %d/%d\n----------' % (it, n_iters))
            for epoch in tqdm(range(epochs), desc='Fitting graph explorer'):
                epoch += 1

                # If nSamples is set, sample a subset of the training data at each epoch              
                if self.nSamples > 0:
                    if epoch == 1:
                        train_loader_original = train_loader
                        valid_loader_original = valid_loader
                    train_loader = self.sample_input(train_loader_original)
                    valid_loader = self.sample_input(valid_loader_original, is_test=True)

                
                # Sample encoded molecules from the network
                loader = self.sampleEncodedPairsToLoader(net, train_loader)
                
                # Train the agent with policy gradient
                self.policy_gradient(loader)

                # Evaluate model
                smiles, frags = self.agent.sample(valid_loader, self.repeat)
                scores = self.agent.evaluate(smiles, frags, evaluator=self.env, no_multifrag_smiles=self.no_multifrag_smiles)
                scores['Smiles'], scores['Frags'] = smiles, frags             

                # Save evaluate criteria and save best model
                self.saveBestState(scores, criteria, epoch, it)

                # Log performance and genearated compounds
                self.logPerformanceAndCompounds(epoch, epochs, scores)
        
                # Early stopping
                if (epoch >= min_epochs) and  (epoch - self.last_save > patience) : break

            if self.crover is not None:
                self.agent.load_state_dict(self.bestState)
                self.crover.load_state_dict(self.bestState)
            if it - self.last_iter > 1: break

        torch.cuda.empty_cache()
        self.monitor.close()
