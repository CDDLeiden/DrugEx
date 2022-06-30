"""
evaluator

Created by: Martin Sicho
On: 29.05.22, 18:12
"""
import multiprocessing

from drugex.logs import logger
from drugex.parallel.interfaces import ParallelException, ParallelProcessor


class ParallelSupplierEvaluator(ParallelProcessor):
    """
    Class implementing parallel evaluation of `MolSupplier` instances on input data (see `ParallelSupplierEvaluator.apply`).

    """

    def __init__(self, supplier_class, n_proc=None, chunk_size=None, chunks=None, return_unique=True, return_suppliers=False, result_collector=None, always_return=False, args=None, kwargs=None):
        """
        Initialize this instance with a `MolSupplier` and other parameters. Note that the supplier is passed as a class and not an instance. This helps to avoid some issues with serialization between processes and, thus, `ParallelSupplierEvaluator` serves only as a template for execution. Also note that the `ParallelSupplierEvaluator` assumes that the first argument of the `MolSupplier` constructor accepts the data to be processed.

        Results of the calculation invoked by `ParallelSupplierEvaluator.apply` are concatenated and returned as a `list` unless 'return_suppliers' is specified.

        Args:
            supplier_class: Class of the `MolSupplier` to use for evaluation.
            return_unique: Attempt to only extract unique results from the result set. This is achieved by converting the results to a `set` first.
            return_suppliers: Return the created suppliers along with the results. This implies 'return_unique=False' and cancels concatenation of results. In this case the return value is a list of tuples where the first item is the list returned from the process and the second is the associated `MolSupplier` instance. Note that some `MolSupplier` implementations might not allow serialization between processes so this may fail in some cases.
            result_collector: A `callable` that will handle processing of results from each process. If it is set, the `ParallelSupplierEvaluator.apply` method returns `None`. This can be reverted by setting 'always_return=False'
            always_return: `ParallelSupplierEvaluator.apply` always returns the concatenated result set.
            args:
            kwargs:
        """

        super().__init__(n_proc, chunk_size, chunks)
        self.makeUnique = return_unique
        self.includeSuppliers = return_suppliers
        self.args = [] if not args else args
        self.kwargs = dict() if not kwargs else kwargs
        self.supplier = supplier_class
        self.collector = result_collector
        self.alwaysReturn = always_return
        self.result = []
        self.errors = []

    def initSupplier(self, supplier_class, chunk):
        """
        Initialize a `MolSupplier` instance on the given chung of data.
        Args:
            supplier_class: `MolSupplier` to initialize.
            chunk: Data chunk.

        Returns:
            initialized `MolSupplier`
        """

        return supplier_class(chunk, *self.args, **self.kwargs)

    def run(self, chunk, chunk_id, total_chunks):
        """
        Initialize and start evaluation of the `MolSupplier` instance on the given chunk of data.

        Args:
            chunk: Current chunk of data.
            chunk_id: ID of the current chunk.
            total_chunks: Total number of chunkSize to be processed by this `ParallelSupplierEvaluator`.

        Returns:
            result of the `MolSupplier.toList()` method that is used for evaluation.
        """

        sup = self.initSupplier(self.supplier, chunk)
        ret = sup.toList()
        logger.info(f"Finished {chunk_id}/{total_chunks} chunkSize for supplier: {sup}")
        if self.includeSuppliers:
            return ret, sup
        else:
            return ret

    def collectResult(self, data):
        """
        Collect and concatenate the current result set (self.result) with the newly generated data from the process.

        Args:
            data: Data to be collected.
        """

        if self.includeSuppliers:
            self.result.append(data)
        else:
            self.result.extend(data)

    def callback(self, data):
        """
        Called whenever a process finishes and `ParallelSupplierEvaluator.run()` returns new data. If a 'result_collector' was specified, the data is passed to it. Otherwise, the `ParallelSupplierEvaluator.collectResult()` method is invoked.

        Args:
            data: New data from the external process.
        """

        if self.collector:
            self.collector(data)
            if self.alwaysReturn:
                self.collectResult(data)
        else:
            self.collectResult(data)

    def error(self, data):
        """
        Catch and log an error occurring in the parallel process.

        Raises:
            `ParallelException`

        Args:
            data: error data
        """

        self.errors.append(data)
        logger.exception(data)
        raise ParallelException(data)

    def apply(self, data, collector=None):
        """
        Apply the `ParallelSupplierEvaluator.run()` across a `Pool` of workers.

        Args:
            data:

        Returns:
            Concatenated list of values generated by the specified `MolSupplier` in each thread. Can return None or a list of tuples if the `ParallelSupplierEvaluator` was initialized with 'result_collector' or 'return_suppliers' (see `ParallelSupplierEvaluator.__init__()`).
        """

        pool = multiprocessing.Pool(self.nProc)

        tasks = []
        chunks = self.getChunks(data)
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