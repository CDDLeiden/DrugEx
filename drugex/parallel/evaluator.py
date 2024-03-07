import concurrent
from tqdm.auto import tqdm

from drugex.parallel.interfaces import ParallelException, ParallelProcessor

class ParallelSupplierEvaluator(ParallelProcessor):
    """
    Class implementing parallel evaluation of `MolSupplier` instances on input data (see `ParallelSupplierEvaluator.apply`).

    """

    def __init__(self, supplier_class, n_proc=None, chunk_size=1000, chunks=None, args=None, kwargs=None):
        """
        Initialize this instance with a `MolSupplier` and other parameters. Note that the supplier is passed as a class and not an instance. This helps to avoid some issues with serialization between processes and, thus, `ParallelSupplierEvaluator` serves only as a template for execution. Also note that the `ParallelSupplierEvaluator` assumes that the first argument of the `MolSupplier` constructor accepts the data to be processed.

        Results of the calculation invoked by `ParallelSupplierEvaluator.apply` are concatenated and returned as a `list` unless 'return_suppliers' is specified.

        Args:
            supplier_class: Class of the `MolSupplier` to use for evaluation.
            args:
            kwargs:
        """

        super().__init__(n_proc, chunk_size, chunks)
        self.args = [] if not args else args
        self.kwargs = dict() if not kwargs else kwargs
        self.supplier = supplier_class
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

    def run(self, chunk, error):
        """
        Initialize and start evaluation of the `MolSupplier` instance on the given chunk of data.

        Args:
            chunk: Current chunk of data.
        Returns:
            result of the `MolSupplier.toList()` method for the given chunk.
        """

        try:
            sup = self.initSupplier(self.supplier, chunk)
            return sup.toList(), sup
        except Exception as exp:
            error(exp)

    def error(self, data):
        """
        Catch and log an error occurring in the parallel process.

        Raises:
            `ParallelException`

        Args:
            data: error data
        """
        self.errors.append(data)
        raise ParallelException(data)

    def apply(self, data, collector, error=None, desc_string=None):
        """
        Apply the `ParallelSupplierEvaluator.run()` across a `Pool` of workers.

        Args:
            data: input data to divide into chunks and process in parallel
            collector: a `ResultCollector` that receives results of parallel processes
            error: a callable to handle errors during evaluation of each parallel supplier
            desc_string: progress bar description string
        Returns:
            `None`
        """

        error = error if error else self.error
        chunk_size = self.getChunkSize(data)
        data = [data[i: i+chunk_size] for i in range(0, len(data), chunk_size)]
        batch_size = self.nProc
        results = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.nProc) as executor:
            batches = [data[i: i+batch_size] for i in range(0, len(data), batch_size)]
            for batch in tqdm(batches, desc=f"{self.supplier.__name__ if not desc_string else desc_string} (batch processing)"):
                for result in executor.map(self.run, batch, len(batch) * [error]):
                    results.append(collector(result))

        return results