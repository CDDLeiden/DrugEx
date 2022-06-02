"""
trainers

Created by: Martin Sicho
On: 01.06.22, 11:44
"""
from drugex.training.interfaces import Trainer



class Pretrainer(Trainer):

    def __init__(self, algorithm, gpus=(0,)):
        super().__init__(algorithm, gpus)
        self.initAlgorithm()

    def initAlgorithm(self):
        self.model.to(self.device)

    def fit(self, train_loader, valid_loader=None, training_monitor=None, epochs=None, args=tuple(), kwargs=dict()):
        self.model.fit(train_loader, valid_loader, *args, out=training_monitor, epochs=epochs, **kwargs)


class FineTuner(Pretrainer):

    def __init__(self, previous, gpus=(0,)):
        super().__init__(previous.getModel(), gpus)


class Reinforcer(Trainer):


    def fit(self, train_loader, valid_loader=None, training_monitor=None, epochs=1000):
        self.model.attachToDevice(self.device)
        self.model.attachToDevices(self.device)
        self.model.fit(train_loader=train_loader, test_loader=valid_loader, epochs=epochs, out=training_monitor)








