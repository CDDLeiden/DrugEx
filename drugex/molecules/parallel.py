"""
parallel

Created by: Martin Sicho
On: 02.05.22, 17:30
"""
import logging
import multiprocessing
from abc import ABC, abstractmethod

import numpy as np

from drugex.logs import logger


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

    @abstractmethod
    def get(self):
        pass

class ListCollector(ResultCollector):

    def __init__(self, mode='extend'):
        self.result = []
        self.mode = mode
        if self.mode not in ('append', 'extend'):
            raise ParallelException("Unknown list collector method:" + mode)

    def __call__(self, result):
        if self.mode == 'extend':
            self.result.extend(result)
        else:
            self.result.append(result)

    def get(self):
        return self.result

class ParallelSupplierEvaluator:

    def __init__(self, supplier_class, n_proc=multiprocessing.cpu_count(), chunks=None, return_unique=True, return_suppliers=False, result_collector=None, args=None, kwargs=None):
        self.nProc = n_proc
        self.makeUnique = return_unique
        self.includeSuppliers = return_suppliers
        self.chunks = self.nProc if not chunks else chunks
        if type(chunks) != DataSplitter:
            self.chunks = ArraySplitter(self.chunks)
        self.args = [] if not args else args
        self.kwargs = dict() if not kwargs else kwargs
        self.supplier = supplier_class
        self.result = result_collector if result_collector else (ListCollector('append') if self.includeSuppliers else ListCollector())
        self.errors = []

    def initSupplier(self, supplier_class, chunk):
        return supplier_class(chunk, *self.args, **self.kwargs)

    def run(self, chunk, current_chunk, total_chunks):
        sup = self.initSupplier(self.supplier, chunk)
        ret = sup.get()
        logger.info(f"Finished {current_chunk}/{total_chunks} chunks for supplier: {sup}")
        if self.includeSuppliers:
            return ret, sup
        else:
            return ret

    def callback(self, data):
        self.result(data)

    def error(self, data):
        self.errors.append(data)
        logging.error(repr(data))
        raise ParallelException(data)

    def get(self, data):
        pool = multiprocessing.Pool(self.nProc)

        tasks = []
        chunks = self.chunks(data)
        for idx, chunk in enumerate(chunks):
            tasks.append(pool.apply_async(
                self.run, args=(chunk, idx+1, len(chunks)),
                callback=self.callback,
                error_callback=self.error
            ))
        for task in tasks:
            task.wait()
        pool.close()
        pool.join()

        result = self.result.get()
        return result if not self.makeUnique or self.includeSuppliers else list(set(result))

