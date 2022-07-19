"""
monitors

Created by: Martin Sicho
On: 02.06.22, 13:59
"""
import os.path
from abc import ABC

import numpy as np
import pandas as pd
import torch

from drugex.training.interfaces import TrainingMonitor

class DictMonitor(TrainingMonitor, ABC):

    def __init__(self, verbose=False):
        self.dict = {1 : []}
        self.currentEpoch = 1
        self.currentStep = 1
        self.totalSteps = None
        self.totalEpochs = None
        self.bestState = None
        self.verbose = verbose

    def getDict(self):
        return self.dict

    def asDataFrame(self):
        evaluation = {
            'epoch': [],
            'valid_ratio': [],
            'unique_ratio': [],
            'desire_ratio': [],
            'mean_train_loss': [],
        }
        for epoch in self.dict:
            evaluation['epoch'].append(epoch)
            losses = []
            for item in self.dict[epoch]:
                if item['loss_train'] is not None:
                    losses.append(item['loss_train'])
                else:
                    evaluation['valid_ratio'].append(item['kwargs']['valid_ratio'])
                    evaluation['desire_ratio'].append(item['kwargs']['desire_ratio'])
                    evaluation['unique_ratio'].append(item['kwargs']['unique_ratio'])
            evaluation['mean_train_loss'].append(np.mean(losses))
        return pd.DataFrame(evaluation)

    def saveModel(self, model):
        self.bestState = model.getModel()

    def changeEpoch(self, epoch):
        current_epoch = max([epoch, max(self.dict.keys())]) if epoch else max([self.currentEpoch, max(self.dict.keys())])
        self.currentEpoch = current_epoch

    def changeStep(self, step):
        if step:
            self.currentStep = step

    def savePerformanceInfo(self, current_step=None, current_epoch=None, loss=None, *args, **kwargs):
        if not self.verbose:
            for key in ('smiles_scores', 'smiles', 'frags', 'smiles_scores_key'):
                if key in kwargs:
                    del kwargs[key]
        self.changeEpoch(current_epoch)
        self.changeStep(current_step)
        if not self.currentEpoch in self.dict:
            self.dict[self.currentEpoch] = []
        self.dict[self.currentEpoch].append({
            'step' : self.currentStep,
            'loss_train' : loss,
            'args' : args,
            'kwargs' : kwargs
        })

    def saveProgress(self, current_step=None, current_epoch=None, total_steps=None, total_epochs=None, *args, **kwargs):
        if not self.totalSteps and total_steps:
            self.totalSteps = total_steps
        if not self.totalEpochs and total_epochs:
            self.totalEpochs = total_epochs
        self.changeEpoch(current_epoch)
        self.changeStep(current_step)

    def endStep(self, step, epoch):
        if not step and epoch:
            self.currentEpoch = epoch + 1

    def getModel(self):
        return self.bestState


class NullMonitor(TrainingMonitor):

    def saveModel(self, model):
        pass

    def savePerformanceInfo(self, current_step=None, current_epoch=None, loss=None, *args, **kwargs):
        pass

    def saveProgress(self, current_step=None, current_epoch=None, total_steps=None, total_epochs=None, *args, **kwargs):
        pass

    def endStep(self, step, epoch):
        pass

    def close(self):
        pass

    def getModel(self):
        pass


class FileMonitor(DictMonitor):
    """
    A simple `TrainingMonitor` implementation with file outputs.

    """

    def __init__(self, path, verbose=False):
        super().__init__(verbose)
        self.path = path
        self.directory = os.path.dirname(path)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.out = open(path + '_fit.log', 'w', encoding='utf-8')
        self.outSmiles = open(path + '_smiles.tsv', 'w', encoding='utf-8')
        self.outSmilesHeaderDone = False
        self.bestState = None

    def savePerformanceInfo(self, current_step=None, current_epoch=None, loss=None, *args, **kwargs):
        super().savePerformanceInfo(current_step, current_epoch, loss, *args, **kwargs)
        self.out.write(f"Current training loss: {loss} \n")
        if 'smiles_scores' in kwargs:
            smiles_scores = kwargs['smiles_scores']
            smiles_scores_key = kwargs['smiles_scores_key'] if 'smiles_scores_key' in kwargs else [str(x+1) for x in range(len(smiles_scores[0]))]
            if not self.outSmilesHeaderDone:
                header = '\t'.join(['Epoch'] + list(smiles_scores_key)) + '\n'
                self.outSmiles.write(header)
                self.outSmilesHeaderDone = True
            for item in smiles_scores:
                self.outSmiles.write('\t'.join([f'{self.currentEpoch}'] + [str(x) for x in item]) + '\n')
            del kwargs['smiles_scores']
            if 'smiles_scores_key' in kwargs:
                del kwargs['smiles_scores_key']
        self.out.write(f"Other data: \n\t args=\n{args} \n\t kwargs=\n{kwargs} \n")
        self.out.flush()

    def saveModel(self, model):
        super().saveModel(model)
        torch.save(self.bestState, self.path + '.pkg')

    def saveProgress(self, current_step=None, current_epoch=None, total_steps=None, total_epochs=None, *args, **kwargs):
        super().saveProgress(current_step, current_epoch, total_steps, total_epochs, *args, **kwargs)
        self.out.write(f"Epoch {self.currentEpoch}/{self.totalEpochs}, Step {self.currentStep}/{self.totalSteps}\n")
        self.out.flush()

    def endStep(self, step, epoch):
        super().endStep(step, epoch)
        self.out.flush()

    def close(self):
        super().close()
        self.out.close()
