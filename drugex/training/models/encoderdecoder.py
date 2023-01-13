from abc import ABC, abstractmethod
from copy import deepcopy

import torch
import time

from tqdm import tqdm
from tqdm.auto import tqdm
from torch import nn

from drugex import utils, DEFAULT_DEVICE, DEFAULT_GPUS
from drugex.logs import logger
from .attention import DecoderAttn
from drugex.training.interfaces import Generator
from drugex.training.scorers.smiles import SmilesChecker
from drugex.training.monitors import NullMonitor
from drugex.logs.utils import callwarning


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
        
class SmilesFragsGeneratorBase(BaseGenerator):
        
    def trainNet(self, loader, monitor=None):
        monitor = monitor if monitor else NullMonitor()
        net = nn.DataParallel(self, device_ids=self.gpus)
        total_steps = len(loader)
        current_step = 0
        for src, trg in tqdm(loader, desc='Iterating over training batches', leave=False):
            src, trg = src.to(self.device), trg.to(self.device)
            self.optim.zero_grad()
            loss = net(src, trg)
            loss = -loss.mean()     
            loss.backward()
            self.optim.step()
            current_step += 1
            monitor.saveProgress(current_step, None, total_steps, None)
            monitor.savePerformanceInfo(current_step, None, loss.item())
            
    def validate(self, loader, evaluator=None, no_multifrag_smiles=True):
        
        net = nn.DataParallel(self, device_ids=self.gpus)

        pbar = tqdm(loader, desc='Iterating over validation batches', leave=False)
        frags, smiles, scores = self.evaluate(pbar, method=evaluator, no_multifrag_smiles=no_multifrag_smiles)
        valid = scores.VALID.mean() 
        desired = scores.DESIRE.mean()
                
        with torch.no_grad():
            loss_valid = sum( [ sum([-l.mean().item() for l in net(src, trg)]) for src, trg in loader ] )
                
        smiles_scores = []
        for idx, smile in enumerate(smiles):
            logger.debug(f"{scores.VALID[idx]}\t{frags[idx]}\t{smile}")
            smiles_scores.append((smile, scores.VALID[idx], scores.DESIRE[idx], frags[idx]))
                
        return valid, desired, loss_valid, smiles_scores
    
    def sample(self, loader, repeat=1):
        net = nn.DataParallel(self, device_ids=self.gpus)
        frags, smiles = [], []
        with torch.no_grad():
            for _ in range(repeat):                
                for src, _ in loader:
                    trg = net(src.to(self.device))
                    smiles += [self.voc_trg.decode(s, is_tk=False) for s in trg]
                    frags += [self.voc_trg.decode(s, is_tk=False) for s in src]

        return smiles, frags
