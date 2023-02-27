"""
interfaces

Created by: Martin Sicho
On: 01.06.22, 11:29
"""
from abc import ABC, abstractmethod
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
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

    def filterNewMolecules(self, df_old, df_new, with_frags = True, drop_duplicates=True, drop_undesired=True, evaluator=None, no_multifrag_smiles=True):
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
        
        # Make sure both valid molecules and include input fragments if needed
        scores = SmilesChecker.checkSmiles(df_new.SMILES.tolist(), frags=df_new.Frags.tolist() if with_frags else None,
                                           no_multifrag_smiles=no_multifrag_smiles)
        df_new = pd.concat([df_new, scores], axis=1)
        
        if with_frags:
            df_new = df_new[df_new.Accurate == 1]
        else:
            df_new = df_new[df_new.Valid == 1]
        
        # Canonalize SMILES
        df_new['SMILES'] = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in df_new.SMILES]    

        # drop duplicates
        if drop_duplicates:
            df_new = df_new.drop_duplicates(subset=['SMILES'])
            df_new = df_new[df_new.SMILES.isin(df_old.SMILES) == False]

        # drop undesired molecules
        if drop_undesired:
            if evaluator is None:
                raise ValueError('Evaluator must be provided to filter molecules by desirability')
            # Compute desirability scores
            scores = self.evaluate(df_new.SMILES.tolist(), frags=df_new.SMILES.tolist(),
                                   evaluator=evaluator, no_multifrag_smiles=no_multifrag_smiles)
            df_new['Desired'] = scores['Desired']
            # Filter out undesired molecules
            df_new = df_new[df_new.Desired == 1]
        
        return df_new
    
    def logPerformanceAndCompounds(self, epoch, metrics, scores):
        """ 
        Log performance and compounds
        
        Parameters:
        ----------
        epoch: `int`
            The current epoch
        metrics: `dict`
            A dictionary with the performance metrics
        scores: `DataFrame`
            A `DataFrame` with generated molecules and their scores
        """

        # Add epoch to metrics and order columns
        metrics['Epoch'] = epoch
        metrics = {k: metrics[k] for k in ['Epoch', 'loss_train', 'loss_valid', 'valid_ratio', 'accurate_ratio', 'best_epoch'] if k in metrics.keys()}

        # Add epoch to scores and order columns
        scores['Epoch'] = epoch
        if 'Frags' in scores.columns:
            firts_cols = ['Epoch', 'SMILES', 'Frags', 'Valid', 'Accurate']
        else:
            firts_cols = ['Epoch', 'SMILES', 'Valid']
        scores = pd.concat([scores[firts_cols], scores.drop(firts_cols, axis=1)], axis=1)

        # Save performance info and generate smiles
        self.monitor.savePerformanceInfo(metrics, df_smiles = scores)
        self.monitor.endStep(None, epoch)


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
            valid_metrics['loss_train'] = loss_train

            if value < best:
                monitor.saveModel(self)    
                best = value
                last_save = epoch
                logger.info(f"Model was saved at epoch {epoch}")               
            valid_metrics['best_epoch'] = last_save

            # Log performance and generated compounds
            self.logPerformanceAndCompounds(epoch, valid_metrics, smiles_scores)

            del loss_train, valid_metrics, smiles_scores
                
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
        evaluator: Environement
            An `Environement` instance used to evaluate the molecules
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