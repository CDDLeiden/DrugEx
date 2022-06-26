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


class ParallelProcessor(ABC):
    """
    Simple interface to define parameters for parallel processing of data.
    """

    def __init__(self, n_proc=None, chunk_size=None, chunks=None):
        """
        Initialize parameters.

        Args:
            n_proc: Number of processes to initialize. Defaults to all available CPUs.
            chunk_size: Maximum size of a chunk to process by a single CPU (can help bring down memory usage, but more processing overhead). If `None`, it is set to "len(data) / n_proc".
            chunks: Number of chunks to divide the input data into. Defaults to 'n_proc'. You can also provide a `DataSplitter` that produces the chunks of data to be processed itself. If "chunks" is present, "chunkSize" is ignored.
        """
        self.nProc = n_proc if n_proc else multiprocessing.cpu_count()
        self.chunkSize = chunk_size
        self.chunks = chunks

    def getChunks(self, data):
        method = None

        if self.chunks:
            if type(self.chunks) == int:
                method = ArraySplitter(self.chunks)
            else:
                method = self.chunks
        elif self.chunkSize:
            n_chunks = (len(data) // self.chunkSize) + (1 if len(data) % self.chunkSize != 0 else 0)
            method = ArraySplitter(n_chunks)
        else:
            method = ArraySplitter(self.nProc)

        return method(data)

    @abstractmethod
    def apply(self, data):
        pass