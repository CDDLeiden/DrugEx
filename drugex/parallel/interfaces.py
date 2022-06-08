"""
interfaces

Created by: Martin Sicho
On: 29.05.22, 18:09
"""
import multiprocessing
from abc import ABC, abstractmethod

import numpy as np


class DataSplitter(ABC):

    def __init__(self, chunks):
        self.chunks = chunks

    @abstractmethod
    def __call__(self, data_source):
        pass


class ArraySplitter(DataSplitter):

    @staticmethod
    def isIter(chunk):
        try:
            iter(chunk)
            return True
        except TypeError:
            return False

    @staticmethod
    def makeIter(chunk):
        return np.nditer(chunk)

    def __call__(self, data_source):
        return [chunk if self.isIter(chunk) else self.makeIter(chunk) for chunk in np.array_split(data_source, self.chunks)]


class ParallelException(Exception):
    pass


class ResultCollector(ABC):

    @abstractmethod
    def __call__(self, result):
        pass


class MoleculeProcessor(ABC):

    def __init__(self, n_proc=None, chunk_size=None):
        self.nProc = n_proc if n_proc else multiprocessing.cpu_count()
        self.chunkSize = chunk_size

    def getChunks(self, data):
        if self.chunkSize:
            return (len(data) // self.chunkSize) + (len(data) % self.chunkSize)

    def getApplierArgs(self, data, collector):
        return {
            "chunks" : self.getChunks(data),
            "n_proc" : self.nProc,
            "result_collector" : collector
        }

    @abstractmethod
    def applyTo(self, data):
        pass