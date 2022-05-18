"""
splitting

Created by: Martin Sicho
On: 07.05.22, 15:54
"""
from abc import ABC, abstractmethod


class DataCollector(ABC):

    @abstractmethod
    def __call__(self, data):
        pass

class TrainTestSplitter(ABC):

    def __init__(self, train_collector=None, test_collector=None):
        self.trainCollect = train_collector
        self.testCollect = test_collector

    @abstractmethod
    def __call__(self, data):
        pass

class FragmentPairEncoder(ABC):

    @abstractmethod
    def encodeMol(self, mol):
        pass

    def encodeFrag(self, frag):
        pass
