"""
interfaces

Created by: Martin Sicho
On: 01.06.22, 11:29
"""
from abc import ABC, abstractmethod
from copy import deepcopy

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from typing import List
from rdkit import Chem

from drugex.logs import logger
from drugex.training.scorers.smiles import SmilesChecker
from drugex.training.interfaces import Model   
from drugex.training.monitors import NullMonitor

class Generator(Model, ABC):
    """
    The base generator class for fitting and evaluating a DrugEx generator.
    """
    
    @abstractmethod
    def sample(self, *args, **kwargs):
        """
        Samples molcules from the generator.

        Returns
        -------
        smiles : List
            List of SMILES strings
        frags : List, optional
            List of fragments used to generate the molecules
        """
        pass

    @abstractmethod
    def trainNet(self, loader, epoch, epochs):
        """
        Train the generator for a single epoch.

        Parameters
        ----------
        loader : DataLoader
            a `DataLoader` instance to use for training
        epoch : int
            the current epoch
        epochs : int
            the total number of epochs
        """
        pass

    @abstractmethod
    def validateNet(self, loader=None, evaluator=None, no_multifrag_smiles=True, n_samples=None):
        """
        Validate the performance of the generator.

        Parameters
        ----------
        loader : DataLoader
            a `DataLoader` instance to use for validation.
        evaluator : ModelEvaluator
            a `ModelEvaluator` instance to use for validation
        no_multifrag_smiles : bool
            if `True`, only single-fragment SMILES are considered valid
        n_samples : int
            the number of samples to use for validation. Not used by transformers.

        Returns
        -------
        valid_metrics : dict
            a dictionary with the validation metrics
        smiles_scores : DataFrame
            a `DataFrame` with the scores for each molecule
        """
        pass

    @abstractmethod
    def generate(self, *args, **kwargs):

        """
        Generate molecules from the generator.

        Returns
        -------
        df_smiles : DataFrame
            a `DataFrame` with the generated molecules (and their scores)
        """
        pass

    @abstractmethod
    def filterNewMolecules(self, smiles, new_smiles, new_frags=None, 
        drup_duplicates=True, drop_undesired=True, 
        evaluator=None, no_multifrag_smiles=True):
        """
        Filter out molecules that are already in the training set.

        Parameters
        ----------
        smiles : List
            a list of previously generated molecules
        new_smiles : List
            a list of newly generated molecules
        new_frags : List    
            a list of fragments for the newly generated molecules
        drup_duplicates : bool
            if `True`, duplicate molecules are dropped
        drop_undesired : bool
            if `True`, molecules that are not desired are dropped
        evaluator : ModelEvaluator
            a `ModelEvaluator` instance used to evaluate the molecule desirability
        no_multifrag_smiles : bool
            if `True`, only single-fragment SMILES are considered valid
        
        Returns
        -------
        new_smiles : List
            a list of newly generated molecules after filtering
        new_frags : List
            a list of fragments for the newly generated molecules after filtering        
        """
        pass 

    def fit(self, train_loader, valid_loader, epochs=100, patience=50, evaluator=None, monitor=None, no_multifrag_smiles=True):
        """
        Fit the generator.

        Parameters
        ----------
        train_loader : DataLoader
            a `DataLoader` instance to use for training
        valid_loader : DataLoader
            a `DataLoader` instance to use for validation
        epochs : int
            the number of epochs to train for
        patience : int
            the number of epochs to wait for improvement before early stopping
        evaluator : ModelEvaluator
            a `ModelEvaluator` instance to use for validation
            TODO: maybe the evaluator should be hard coded to None here as during PT/FT training we don't need it
        monitor : Monitor
            a `Monitor` instance to use for saving the model and performance info
        no_multifrag_smiles : bool
            if `True`, only single-fragment SMILES are considered valid
        """
        self.monitor = monitor if monitor else NullMonitor()
        best = float('inf')
        last_save = -1
         
        for epoch in tqdm(range(epochs), desc='Fitting model'):
            epoch += 1
            
            # Train model
            loss_train = self.trainNet(train_loader, epoch, epochs)

            # Validate model
            valid_metrics, smiles_scores = self.validateNet(loader=valid_loader, evaluator=evaluator, no_multifrag_smiles=no_multifrag_smiles, n_samples=train_loader.batch_size*2)

            # Save model based on validation loss or valid ratio
            if 'loss_valid' in valid_metrics.keys(): value = valid_metrics['loss_valid']
            else : value = 1 - valid_metrics['valid_ratio']

            if value < best:
                monitor.saveModel(self)    
                best = value
                last_save = epoch
                logger.info(f"Model was saved at epoch {epoch}")   
            
            # Save performance info and generate smiles
            valid_metrics['loss_train'] = loss_train
            valid_metrics['best_value'] = best
            # monitor.savePerformanceInfo(None, epoch, None, loss_valid=loss_valid, valid_ratio=valid, desire_ratio=frags_desire, best_loss=best, smiles_scores=smiles_scores, smiles_scores_key=('SMILES', 'Valid', 'Desire', 'Frags'))
            monitor.savePerformanceInfo(None, epoch, None, smiles_scores=smiles_scores, smiles_scores_key=smiles_scores.keys(), **valid_metrics)

            del loss_train, valid_metrics, smiles_scores
            monitor.endStep(None, epoch)
                
            # Early stopping
            if epoch - last_save > patience : break
        
        torch.cuda.empty_cache()
        monitor.close()



    def evaluate(self, smiles : List[str], frags : List[str]=None, evaluator=None, no_multifrag_smiles : bool=True, unmodified_scores : bool=False):
        """
        Evaluate molecules by using the given evaluator or checking for validity.

        Parameters:
        ----------
        smiles: List
            List of SMILES to evaluate
        frags: List
            List of fragments used to generate the SMILES
        evaluator: ModelEvaluator
            A `ModelEvaluator` instance used to evaluate the molecules
        no_multifrag_smiles: bool
            If `True`, only single-fragment SMILES are considered valid
        unmodified_scores: bool
            If `True`, the scores are not modified by the evaluator

        Returns
        -------
        scores: DataFrame
            A `DataFrame` with the scores for each molecule
        """

        if evaluator is None:
            scores = SmilesChecker.checkSmiles(smiles, frags=frags, no_multifrag_smiles=no_multifrag_smiles)
        else:
            if unmodified_scores:
                scores = evaluator.getUnmodifiedScores(smiles)
            else:
                scores = evaluator.getScores(smiles, frags=frags, no_multifrag_smiles=no_multifrag_smiles)
        
        return scores

    def getModel(self):
        """
        Return a copy of this model as a state dictionary.

        Returns
        -------
        model: dict
            A serializable copy of this model as a state dictionary
        """
        return deepcopy(self.state_dict())   

class FragGenerator(Generator):

    """
    A generator for fragment-based molecules.
    """

    def init_states(self):
        """
        Initialize model parameters
        
        Notes:
        -----
        Xavier initialization for all parameters except for the embedding layer
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.attachToGPUs(self.gpus)

    def attachToGPUs(self, gpus):
        """
        Attach model to GPUs

        Parameters:
        ----------
        gpus: `tuple`
            A tuple of GPU ids to use
        """
        self.gpus = gpus
        self.to(self.device)

    @abstractmethod
    def loaderFromFrags(self, frags, batch_size=32, n_proc=1):
        """
        Encode the input fragments and create a dataloader object
        
        Parameters:
        ----------
        frags: `list`
            A list of input fragments (in SMILES format)
        batch_size: `int`
            Batch size for the dataloader
        n_proc: `int`
            Number of processes to use for encoding the fragments
        
        Returns:
        -------
        loader: `torch.utils.data.DataLoader`
            A dataloader object to iterate over the input fragments 
        """
        pass      

    def filterNewMolecules(self, smiles, new_smiles, new_frags, drop_duplicates=True, drop_undesired=True, evaluator=None, no_multifrag_smiles=True):
        """
        Filter the generated SMILES
        
        Parameters:
        ----------
        smiles: `list`
            A list of previous SMILES
        new_smiles: `list`
            A list of additional generated SMILES
        frags: `list`  
            A list of additional input fragments
        drop_duplicates: `bool`
            If `True`, duplicate SMILES are dropped
        drop_undesired: `bool`
            If `True`, SMILES that do not fulfill the desired objectives
        evaluator: `Evaluator`
            An evaluator object to evaluate the generated SMILES
        no_multifrag_smiles: `bool`
            If `True`, only single-fragment SMILES are considered valid
        
        Returns:
        -------
        new_smiles: `list`
            A list of filtered SMILES
        new_frags: `list`
            A list of filtered input fragments
        """
        
        # Make sure both valid molecules and include input fragments
        scores = SmilesChecker.checkSmiles(new_smiles, frags=new_frags, no_multifrag_smiles=no_multifrag_smiles)
        new_smiles = np.array(new_smiles)[scores.Accurate == 1].tolist()
        new_frags = np.array(new_frags)[scores.Accurate == 1].tolist()
        
        # Canonalize SMILES
        new_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in new_smiles]    

        # drop duplicates
        if drop_duplicates:
            new_smiles = np.array(new_smiles)
            new_frags = np.array(new_frags)[np.logical_not(np.isin(new_smiles, smiles))].tolist()
            new_smiles = new_smiles[np.logical_not(np.isin(new_smiles, smiles))].tolist()

        # drop undesired molecules
        if drop_undesired:
            if evaluator is None:
                raise ValueError('Evaluator must be provided to filter molecules by desirability')
            # Compute desirability scores
            scores = self.evaluate(new_smiles, new_frags, evaluator=evaluator, no_multifrag_smiles=no_multifrag_smiles)
            # Filter out undesired molecules
            new_smiles = np.array(new_smiles)[scores.Desired == 1].tolist()
            new_frags = np.array(new_frags)[scores.Desired == 1].tolist()

        return new_smiles, new_frags