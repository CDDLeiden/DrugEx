"""
trainers

Created by: Martin Sicho
On: 01.06.22, 11:44
"""
from drugex import DEFAULT_DEVICE_ID, DEFAULT_DEVICE
from drugex.training.interfaces import Trainer



class Pretrainer(Trainer):

    def fit(self, train_loader, valid_loader=None, monitor=None, epochs=1000, evaluator=None, args=tuple(), kwargs=dict()):
        self.model.fit(train_loader, valid_loader, *args, monitor=monitor, epochs=epochs, **kwargs)


class FineTuner(Pretrainer):
    pass


class Reinforcer(Trainer):


    def __init__(self, explorer, gpus=(0,)):
        super().__init__(explorer, gpus)

    def attachDevices(self, device_id=DEFAULT_DEVICE_ID, device=DEFAULT_DEVICE):
        super().attachDevices(device_id=device_id, device=device)
        if self.model.mutate:
            self.model.mutate.attachToDevice(self.device)
            self.model.mutate.attachToDevices(self.getDevices())
        if self.model.crover:
            self.model.crover.attachToDevice(self.device)
            self.model.crover.attachToDevices(self.getDevices())


    def fit(self, train_loader, valid_loader=None, monitor=None, epochs=1000, args=tuple(), kwargs=dict()):

        self.model.fit(train_loader, valid_loader, *args, epochs=epochs, monitor=monitor, **kwargs)








