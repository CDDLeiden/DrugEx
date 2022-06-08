"""
monitors

Created by: Martin Sicho
On: 02.06.22, 13:59
"""
import torch

from drugex.training.interfaces import TrainingMonitor


class FileMonitor(TrainingMonitor):

    def __init__(self, path):
        self.path = path
        self.out = open(path + '.log', 'w', encoding='utf-8')
        self.model = None

    def savePerformanceInfo(self, current_step, current_epoch, loss, *args, **kwargs):
        smiles_scores = kwargs['smiles_scores']
        for item in smiles_scores:
            self.out.write('%d\t%.3f\t%s\t%s\n' % (item[1], item[2], item[3], item[0]))

    def saveModel(self, model):
        self.model = model
        torch.save(self.model.state_dict(), self.path + '.pkg')

    def saveProgress(self, current_step, current_epoch, total_steps, total_epochs, *args, **kwargs):
        pass

    def getModel(self):
        return self.model

    def endStep(self, step, epoch):
        self.out.flush()

    def close(self):
        self.out.close()

