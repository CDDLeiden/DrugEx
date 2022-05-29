"""
evaluator

Created by: Martin Sicho
On: 29.05.22, 18:12
"""
import multiprocessing

from drugex.logs import logger
from drugex.parallel.interfaces import DataSplitter, ArraySplitter, ParallelException


class ParallelSupplierEvaluator:

    def __init__(self, supplier_class, n_proc=None, chunks=None, return_unique=True, return_suppliers=False, result_collector=None, always_return=False, args=None, kwargs=None):
        self.nProc = n_proc if n_proc else multiprocessing.cpu_count()
        self.makeUnique = return_unique
        self.includeSuppliers = return_suppliers
        self.chunks = self.nProc if not chunks else chunks
        if type(chunks) != DataSplitter:
            self.chunks = ArraySplitter(self.chunks)
        self.args = [] if not args else args
        self.kwargs = dict() if not kwargs else kwargs
        self.supplier = supplier_class
        self.collector = result_collector
        self.alwaysReturn = always_return
        self.result = []
        self.errors = []

    def initSupplier(self, supplier_class, chunk):
        return supplier_class(chunk, *self.args, **self.kwargs)

    def run(self, chunk, current_chunk, total_chunks):
        sup = self.initSupplier(self.supplier, chunk)
        ret = sup.toList()
        logger.info(f"Finished {current_chunk}/{total_chunks} chunks for supplier: {sup}")
        if self.includeSuppliers:
            return ret, sup
        else:
            return ret

    def collectResult(self, data):
        if self.includeSuppliers:
            self.result.append(data)
        else:
            self.result.extend(data)

    def callback(self, data):
        if self.collector:
            self.collector(data)
            if self.alwaysReturn:
                self.collectResult(data)
        else:
            self.collectResult(data)

    def error(self, data):
        self.errors.append(data)
        logger.exception(data)
        raise ParallelException(data)

    def apply(self, data):
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

        ret = None
        if self.result:
            if not self.makeUnique or self.includeSuppliers:
                ret = self.result
            else:
                ret = list(set(self.result))
        return ret