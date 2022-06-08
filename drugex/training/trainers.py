"""
trainers

Created by: Martin Sicho
On: 01.06.22, 11:44
"""
from drugex.training.interfaces import Trainer



class Pretrainer(Trainer):

    def fit(self, train_loader, valid_loader=None, monitor=None, epochs=1000, evaluator=None, args=tuple(), kwargs=dict()):
        self.model.fit(train_loader, valid_loader, *args, monitor=monitor, epochs=epochs, **kwargs)


class FineTuner(Pretrainer):
    pass


class Reinforcer(Trainer):


    def __init__(self, explorer, gpus=(0,)):
        super().__init__(explorer, gpus)


    def fit(self, train_loader, valid_loader=None, monitor=None, epochs=1000, args=tuple(), kwargs=dict()):

        self.model.fit(train_loader, valid_loader, *args, epochs=epochs, monitor=monitor, **kwargs)








