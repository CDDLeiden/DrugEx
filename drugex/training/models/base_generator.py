from abc import ABC, abstractmethod
from copy import deepcopy

import torch
import time

from tqdm import tqdm
from tqdm.auto import tqdm
from torch import nn

from drugex.logs import logger
from drugex.training.interfaces import Generator
from drugex.training.scorers.smiles import SmilesChecker
from drugex.training.monitors import NullMonitor


class BaseGenerator(Generator, ABC):

    @abstractmethod
    def trainNet(self, loader, monitor=None):
        pass

    @abstractmethod
    def validate(self, loader, evaluator=None):
        pass

    def attachToGPUs(self, gpus):
        self.gpus = gpus
        self.to(self.device)

    def getModel(self):
        return deepcopy(self.state_dict())

    def fit(self, train_loader, valid_loader, epochs=100, patience=50, evaluator=None, monitor=None, no_multifrag_smiles=True):
        monitor = monitor if monitor else NullMonitor()
        best = float('inf')
        last_save = -1
         
        for epoch in tqdm(range(epochs), desc='Fitting model'):
            epoch += 1
            t0 = time.time()
            self.trainNet(train_loader, monitor)
            valid, frags_desire, loss_valid, smiles_scores = self.validate(valid_loader, evaluator=evaluator, no_multifrag_smiles=no_multifrag_smiles)
            t1 = time.time()
            
            logger.info(f"Epoch: {epoch} Validation loss: {loss_valid:.3f} Valid: {valid:.3f} FragsDesire: {frags_desire:.3f} Time: {int(t1-t0)}s")
            monitor.saveProgress(None, epoch, None, epochs)

            if loss_valid < best:
                monitor.saveModel(self)    
                best = loss_valid
                last_save = epoch
                logger.info(f"Model was saved at epoch {epoch}")     

            monitor.savePerformanceInfo(None, epoch, None, loss_valid=loss_valid, valid_ratio=valid, desire_ratio=frags_desire, best_loss=best, smiles_scores=smiles_scores, smiles_scores_key=('SMILES', 'Valid', 'Desire', 'Frags'))

            del loss_valid
            monitor.endStep(None, epoch)
                
            if epoch - last_save > patience : break
        
        torch.cuda.empty_cache()
        monitor.close()

    def evaluate(self, loader, repeat=1, method=None, no_multifrag_smiles=True):
        smiles, frags = self.sample(loader, repeat)

        if method is None:
            scores = SmilesChecker.checkSmiles(smiles, frags=frags, no_multifrag_smiles=no_multifrag_smiles)
        else:
            scores = method.getScores(smiles, frags=frags, no_multifrag_smiles=no_multifrag_smiles)
        return frags, smiles, scores

    def init_states(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.attachToGPUs(self.gpus)