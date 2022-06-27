"""
monitors

Created by: Martin Sicho
On: 02.06.22, 13:59
"""
import os.path
from copy import deepcopy

import torch

from drugex.training.interfaces import TrainingMonitor


class FileMonitor(TrainingMonitor):

    def __init__(self, path, verbose=False):
        self.path = path
        self.directory = os.path.dirname(path)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.out = open(path + '.log', 'w', encoding='utf-8')
        self.bestState = None
        self.verbose = verbose

    def savePerformanceInfo(self, current_step, current_epoch, loss, *args, **kwargs):
        self.out.write(f"Current loss: {loss} \n")
        if not self.verbose:
            for key in ('smiles_scores', 'smiles', 'frags'):
                if key in kwargs:
                    del kwargs[key]
        if 'smiles_scores' in kwargs:
            smiles_scores = kwargs['smiles_scores']
            for item in smiles_scores:
                self.out.write('\t'.join([str(x) for x in item]) + '\n')
        self.out.write(f"Other data: \n\t args=\n{args} \n\t kwargs=\n{kwargs} \n")
        self.out.flush()

    def saveModel(self, model):
        self.bestState = deepcopy(model.state_dict())
        torch.save(self.bestState, self.path + '.pkg')

    def saveProgress(self, current_step, current_epoch, total_steps, total_epochs, *args, **kwargs):
        self.out.write(f"Epoch {current_epoch+1 if current_epoch else '--'}/{total_epochs if total_epochs else '--'}, Step {current_step+1 if current_step else '--'}/{total_steps if total_steps else '--'}\n")
        self.out.flush()

    def getModel(self):
        return self.bestState

    def endStep(self, step, epoch):
        self.out.flush()

    def close(self):
        self.out.close()

