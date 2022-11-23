"""
monitors

Created by: Martin Sicho
On: 02.06.22, 13:59
"""
import os.path
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch

from drugex.training.interfaces import TrainingMonitor

class DictMonitor(TrainingMonitor, ABC):

    def __init__(self, verbose=False, clean_after_epoch=False):
        self.dict = {1 : []}
        self.currentEpoch = 1
        self.currentStep = 1
        self.totalSteps = None
        self.totalEpochs = None
        self.bestState = None
        self.verbose = verbose
        self.cleanAfterEpoch = clean_after_epoch

    def getDict(self):
        return self.dict

    def getEpochSummary(self):
        evaluation = {
            'epoch': [],
            'loss_valid': [],
            'valid_ratio': [],
            'unique_ratio': [],
            'desire_ratio': [],
            'mean_train_loss': [],
        }
        for epoch in self.dict:
            evaluation['epoch'].append(epoch)
            losses = []
            for item in self.dict[epoch]:
                if type(item['step']) == int:
                    losses.append(item['loss_train'])
                elif item['step'] == 'final':
                    kwargs = item['kwargs']
                    evaluation['loss_valid'].append(kwargs['loss_valid'] if 'loss_valid' in kwargs else None)
                    evaluation['valid_ratio'].append(kwargs['valid_ratio'])
                    evaluation['desire_ratio'].append(kwargs['desire_ratio'] if 'desire_ratio' in kwargs else None)
                    evaluation['unique_ratio'].append(kwargs['unique_ratio'] if 'unique_ratio' in kwargs else None)
            evaluation['mean_train_loss'].append(np.mean(losses) if losses else None)
        return pd.DataFrame(evaluation)

    def getMoleculeSummary(self):
        df = None
        for epoch in self.dict:
            data = [ x for x in self.dict[epoch] if 'final_smiles' == x['step']]
            if not data:
                return None
            data = data[0]
            data = pd.DataFrame(data['scores'], columns=data['keys'])
            data['Epoch'] = [epoch] * len(data)
            if not df:
                df = data
            else:
                df.append(data)
        return df


    def saveModel(self, model):
        self.bestState = model.getModel()

    def changeEpoch(self, epoch):
        current_epoch = max([epoch, max(self.dict.keys())]) if epoch else max([self.currentEpoch, max(self.dict.keys())])
        self.currentEpoch = current_epoch

    def changeStep(self, step):
        if step:
            self.currentStep = step

    def savePerformanceInfo(self, current_step=None, current_epoch=None, loss=None, *args, **kwargs):
        is_final_step = False
        if 'smiles_scores' in kwargs:
            is_final_step = True

        if not self.verbose:
            for key in ('smiles_scores', 'smiles', 'frags', 'smiles_scores_key'):
                if key in kwargs:
                    del kwargs[key]
        self.changeEpoch(current_epoch)
        self.changeStep(current_step)
        if not self.currentEpoch in self.dict:
            self.dict[self.currentEpoch] = []
        self.dict[self.currentEpoch].append({
            'step' : self.currentStep if not is_final_step else 'final',
            'loss_train' : loss,
            'args' : args,
            'kwargs' : kwargs
        })

        if is_final_step and 'smiles_scores' in kwargs:
            smiles_scores = kwargs['smiles_scores']
            smiles_scores_key = kwargs['smiles_scores_key'] if 'smiles_scores_key' in kwargs else [str(x+1) for x in range(len(smiles_scores[0]))]
            self.dict[self.currentEpoch].append(
                {
                    'step' : 'final_smiles',
                    'epoch' : self.currentEpoch,
                    'scores' : smiles_scores,
                    'keys' : list(smiles_scores_key)
                }
            )

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

        if epoch:
            self.saveEpochData(self.getEpochSummary())
            mols = self.getMoleculeSummary()
            if mols is not None:
                self.saveMolecules(mols)
            if self.cleanAfterEpoch:
                self.dict = {epoch + 1 : []}

    def getModel(self):
        return self.bestState

    @abstractmethod
    def saveEpochData(self, df):
        pass

    @abstractmethod
    def saveMolecules(self, df):
        pass


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

    def __init__(self, path, verbose=False, clean_after_epoch=True):
        super().__init__(verbose, clean_after_epoch)
        self.path = path
        self.directory = os.path.dirname(path)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.out = open(path + '_fit.log', 'w', encoding='utf-8')
        self.outDF = path + '_fit.tsv'
        self.outSmiles = path + '_smiles.tsv' if verbose else None
        self.outSmilesHeaderDone = False
        self.bestState = None

    def savePerformanceInfo(self, current_step=None, current_epoch=None, loss=None, *args, **kwargs):
        super().savePerformanceInfo(current_step, current_epoch, loss, *args, **kwargs)
        if loss:
            self.out.write(f"Current training loss: {loss} \n")
        self.out.flush()

    def saveModel(self, model):
        super().saveModel(model)
        torch.save(self.bestState, self.path + '.pkg')

    def saveProgress(self, current_step=None, current_epoch=None, total_steps=None, total_epochs=None, *args, **kwargs):
        super().saveProgress(current_step, current_epoch, total_steps, total_epochs, *args, **kwargs)
        self.out.write(f"Epoch {self.currentEpoch}/{self.totalEpochs if self.totalEpochs else '--'}, Step {self.currentStep}/{self.totalSteps}\n")
        self.out.flush()

    def endStep(self, step, epoch):
        super().endStep(step, epoch)
        self.out.flush()

    def close(self):
        super().close()
        # df_out = pd.read_csv(self.outDF, sep='\t', header=0)
        # df_smiles = pd.read_csv(self.outSmiles, sep='\t', header=0)
        self.out.close()

    def saveEpochData(self, df):
        self.appendTableToFile(df, self.outDF)

    def saveMolecules(self, df):
        if self.outSmiles:
            self.appendTableToFile(df, self.outSmiles)

    @staticmethod
    def appendTableToFile(df, outfile):
        header_written = os.path.isfile(outfile)
        open_mode = 'a' if header_written else 'w'
        df.to_csv(
            outfile,
            sep='\t',
            index=False,
            header=not header_written,
            mode=open_mode,
            encoding='utf-8',
            na_rep='NA'
        )