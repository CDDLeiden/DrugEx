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
from tqdm import tqdm

from drugex import DEFAULT_GPUS, DEFAULT_DEVICE
from drugex.logs import logger
from drugex.training.interfaces import Model
from drugex.training.monitors import NullMonitor

class Explorer(Model, ABC):
    """
    Implements the DrugEx exploration strategy for DrugEx models under the reinforcement learning framework.
    """

    def __init__(self, agent, env, mutate=None, crover=None, no_multifrag_smiles=True,
        batch_size=128, epsilon=0.1, beta=0.0, n_samples=-1, 
        device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS):

        """
        Initialize the explorer.

        Parameters
        ----------
        agent : Model
            The agent model optimized by the explorer.
        env : Environment
            The environment in which the agent operates.
        mutate : Generator
            The pre-trained network which increases the exploration of the chemical space.
        crover : Generator
            The iteratively updated network which increases the exploitation of the chemical space.
        no_multifrag_smiles : bool
            If True, only single-fragment SMILES are considered valid.
        batch_size : int
            The batch size used for training.
        epsilon : float
            The probability of using the mutate network to generate new molecules.
        beta : float
            The baseline for the reward function.
        n_samples : int
            The number of samples to generate in each iteration. If -1, the whole dataset is used per epoch.
        device : str
            The device on which the model is trained.
        use_gpus : tuple   
            The GPUs to use for training.
        """
        
        super().__init__(device=device, use_gpus=use_gpus)
        self.agent = agent
        self.mutate = mutate
        self.crover = crover
        self.batchSize = batch_size
        self.epsilon = epsilon
        self.beta = beta
        self.env = env
        self.nSamples = n_samples
        self.no_multifrag_smiles = no_multifrag_smiles
        self.bestState = None
        self.best_value = 0
        self.last_save = -1
        self.last_iter = -1


    def attachToGPUs(self, gpus):

        """ 
        Attach the model to GPUs

        Parameters
        ----------
        gpus : tuple
            The GPUs to use for training.
        """

        if hasattr(self, 'agent'):
            self.agent.attachToGPUs(gpus)
        if hasattr(self, 'mutate') and self.mutate:
            self.mutate.attachToGPUs(gpus)
        if hasattr(self, 'crover') and self.crover:
            self.crover.attachToGPUs(gpus)
        self.gpus = gpus

    def getNovelMoleculeMetrics(self, scores):

        """ Get metrics for novel molecules
        
        Parameters
        ----------
        scores : pd.DataFrame
            The scores for each molecule.

        Returns
        -------
        dict
            The metrics:
                - valid_ratio (float): ratio of valid molecules
                - unique_ratio (float): ratio of valid and unique molecules
                - desired_ratio (float): ratio of valid, unique and desired molecules
                - avg_amean (float): average arithmetic mean score of valid and unique molecules
                - avg_gmean (float): average geometric mean score of valid and unique molecules
        """
        
        dct = {}
        ntot = len(scores) 
        
        # Valid compounds
        valid = scores[scores.Valid == 1]
        dct['valid_ratio'] = len(valid) / ntot
        
        # Accurate compounds
        if 'Accurate' in scores.columns:
            accurate = valid[valid.Accurate == 1]
            dct['accurate_ratio'] = len(accurate) / ntot
        else:
            accurate = valid

        # Unique compounds
        unique = accurate.drop_duplicates(subset='SMILES')
        dct['unique_ratio'] = len(unique) / ntot
        
        # Desired compounds
        dct['desired_ratio'] = unique.Desired.sum() / ntot
        
        # Average artithmetic and geometric mean score 
        dct['avg_amean'] = unique[self.env.getScorerKeys()].values.mean()
        dct['avg_gmean'] = unique[self.env.getScorerKeys()].apply(gmean, axis=1).mean()
        
        return dct
    
    def saveBestState(self, value, epoch, it):

        """
        Save best state based on selection criteria

        Parameters
        ----------
        value : float
            The value of the selection criteria.
        epoch : int
            The current epoch.
        it : int
        """
        
        self.monitor.setModel(self)
        self.bestState = self.getModel()
        self.best_value = value
        self.last_save = epoch
        self.last_iter = it

    def logPerformanceAndCompounds(self, epoch, metrics, scores):

        """
        Log performance of model

        Parameters
        ----------
        epoch : int
            The current epoch.
        metrics : dict
            The metrics.
        scores : pd.DataFrame
            The scores for each molecule.
        """
        
        # Add epoch and best epoch to metrics, and order columns
        metrics['Epoch'] = epoch
        metrics = {k : metrics[k] for k in ['Epoch'] + [k for k in metrics.keys() if k != 'Epoch']}
        metrics['best_epoch'] = self.last_save
        
        # Add epoch to scores, and order columns
        scores['Epoch'] = epoch
        if 'Accurate' in scores.columns:
            scores = scores[['Epoch', 'SMILES', 'Frags', 'Valid', 'Accurate', 'Desired'] + self.env.getScorerKeys()]
        else:
            scores = scores[['Epoch', 'SMILES', 'Valid', 'Desired'] + self.env.getScorerKeys()]

        self.monitor.savePerformanceInfo(metrics, df_smiles=scores)
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

        Returns
        -------
        torch.nn.Module
            The current state of the agent
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

    def policy_gradient(self, loader):
        """
        Policy gradient training.
 
        Novel molecules are generated by the agent and are scored by the environment.
        The policy gradient is calculated using the REINFORCE algorithm and the agent is updated.
        
        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            Data loader for training data

        Returns
        -------
        float
            The average loss of the agent
        """

        net = nn.DataParallel(self.agent, device_ids=self.gpus)
        total_steps = len(loader)
        
        for step_idx, src in enumerate(tqdm(loader, desc='Calculating policy gradient...', leave=False)):

            # Decode fragments and smiles, and get loss
            frags, smiles, loss = self.getBatchOutputs(net, src)
            
            # Get rewards
            reward = self.env.getRewards(smiles, frags=frags)

            # Filter out molecules with multiple fragments by setting reward to 0
            if self.no_multifrag_smiles:
                reward = np.array([r if s.count('.') == 0 else [0] for s,r in zip(smiles, reward)])
            reward = torch.Tensor(reward).to(self.device)
            
            # Train model with policy gradient
            self.optim.zero_grad()
            loss = loss * ( reward - self.beta )
            loss = -loss.mean()
            loss.backward()
            self.optim.step()

            self.monitor.saveProgress(self, step_idx, None, total_steps, None, loss=loss.item())
            
        return loss.item()

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
        """
        
        self.monitor = monitor if monitor else NullMonitor()
        self.monitor.setModel(self)
        self.monitor.saveModel(self)
        self.bestState = self.getModel()

        n_iters = 1 if self.crover is None else 10
        net = nn.DataParallel(self, device_ids=self.gpus)
        logger.info(' ')
        
        for it in range(n_iters):
            if n_iters > 1:
                logger.info('\n----------\nITERATION %d/%d\n----------' % (it, n_iters))
            for epoch in tqdm(range(epochs), desc='Fitting graph explorer'):
                epoch += 1
                is_best = False

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
                train_loss = self.policy_gradient(loader)

                # Evaluate model on validation set
                smiles, frags = self.agent.sample(valid_loader)
                scores = self.agent.evaluate(smiles, frags, evaluator=self.env, no_multifrag_smiles=self.no_multifrag_smiles)
                scores['SMILES'], scores['Frags'] = smiles, frags    

                # Compute metrics
                metrics = self.getNovelMoleculeMetrics(scores)    
                metrics['loss_train'] = train_loss     

                # Save evaluate criteria and set best model
                if metrics[criteria] > self.best_value:
                    is_best = True
                    self.saveBestState(metrics[criteria], epoch, it)

                # Save (intermediate) models
                save_model_option = monitor.getSaveModelOption()
                if save_model_option == 'all' or is_best == True:
                    monitor.saveModel(self, epoch if save_model_option in ('all', 'improvement') else None)
                    logger.info(f"Model saved at epoch {epoch}")

                # Log performance and generated compounds
                self.logPerformanceAndCompounds(epoch, metrics, scores)
        
                # Early stopping
                if (epoch >= min_epochs) and  (epoch - self.last_save > patience) : break

            if self.crover is not None:
                logger.warning('Behavior of using crover with a Transformer is not tested yet. Use at your own risk.')
                self.agent.load_state_dict(self.bestState)
                self.crover.load_state_dict(self.bestState)
            if it - self.last_iter > 1: break

        torch.cuda.empty_cache()
        self.monitor.close()
