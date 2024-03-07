import multiprocessing
from abc import ABC, abstractmethod


class ParallelException(Exception):
    pass


class ResultCollector(ABC):

    @abstractmethod
    def __call__(self, result):
        pass

class ListCollector(ResultCollector, ABC):

    @abstractmethod
    def getList(self):
        pass


class ParallelProcessor(ABC):
    """
    Simple interface to define parameters for parallel processing of data.
    """

    def __init__(self, n_proc=None, chunk_size=1000, chunks=None):
        """

        Args:
            n_proc: Number of processes to initialize. Defaults to all available CPUs.
            chunk_size: Maximum size of a chunk to process by a single CPU (can help bring down memory usage, but more processing overhead). If `None`, it is set to "len(data) / n_proc" by `getChunkSize`.
            chunks: Number of chunks to divide the input data into. Defaults to 'n_proc'. If both "chunks" and "chunk_size" are specified, "chunk_size" takes precedence (see `getChunkSize`).
        """
        self.nProc = n_proc if n_proc else multiprocessing.cpu_count()
        self.chunks = chunks if chunks else self.nProc
        self.chunkSize = chunk_size

    def getChunkSize(self, data):
        """
        Determine the chunk size from data.

        Args:
            data: input data (needs to have `len`)

        Returns:
            `int` representing the size of one chunk sent to the parallel process
        """

        if self.chunkSize:
            return self.chunkSize
        else:
            return len(data) // self.chunks

    @abstractmethod
    def apply(self, data, collector):
        """
        Apply the processor to the given data.

        Args:
            data: input data (format depends on the implementation)
            collector: a `ResultCollector` that is used to collect data produced from each process.

        Returns:
            `None`.
        """

        pass