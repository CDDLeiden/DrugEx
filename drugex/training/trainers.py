"""
trainers

Created by: Martin Sicho
On: 01.06.22, 11:44
"""
from drugex.training.interfaces import Trainer



class Pretrainer(Trainer):

    def __init__(self, algorithm, gpus=(0,)):
        super().__init__(algorithm, gpus)

    def fit(self, train_loader, valid_loader=None, monitor=None, epochs=1000, evaluator=None, args=tuple(), kwargs=dict()):
        self.model.fit(train_loader, valid_loader, *args, monitor=monitor, epochs=epochs, **kwargs)


class FineTuner(Pretrainer):
    pass


class Reinforcer(Trainer):


    def fit(self, train_loader, valid_loader=None, training_monitor=None, epochs=1000, args=tuple(), kwargs=dict()):
        self.model.attachToDevice(self.device)
        self.model.attachToDevices(self.device)
        self.model.fit(train_loader, valid_loader, *args, epochs=epochs, out=training_monitor, **kwargs)








