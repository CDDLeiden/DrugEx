"""
processing

Created by: Martin Sicho
On: 27.05.22, 10:16
"""
import math

import numpy as np
from sklearn.model_selection import train_test_split

from drugex.data.interfaces import DataSplitter
from drugex.logs import logger
from drugex.parallel.evaluator import ParallelSupplierEvaluator
from drugex.parallel.interfaces import ParallelProcessor
from drugex.molecules.converters.standardizers import DefaultStandardizer
from drugex.molecules.suppliers import StandardizedSupplier


class Standardization(ParallelProcessor):
    """
    Processor to standardize molecules in parallel.
    """

    def __init__(self, standardizer=DefaultStandardizer(), **kwargs):
        """
        Initialize the standardization processor.

        Args:
            standardizer: The standardizer to use for conversion of input molecules.
        """

        super().__init__(**kwargs)
        self.standardizer = standardizer

    def apply(self, mols, collector=None):
        """
        Apply defined standardization to an iterable of molecules.

        This method just automates initialization of a `ParallelSupplierEvaluator` on the given molecules. Molecules can be given
        as a generator or a `MolSupplier`, but note that they will be evaluated before processing, which may add overhead. In such
        a case consider evaluating the list with a `ParallelSupplierEvaluator` separately prior to processing.

        Args:
            mols: an iterable containing molecules to transform
            collector: a callable to collect the results, passed as the 'result_collector' to `ParallelSupplierEvaluator`

        Returns:
            Standardized list of molecules. If 'collector' is specified, the result is None.
        """

        standardizer = ParallelSupplierEvaluator(
            StandardizedSupplier,
            kwargs={
                "standardizer": self.standardizer
            },
            chunk_size=self.chunkSize,
            chunks=self.chunks,
            result_collector=collector
        )
        return standardizer.apply(np.asarray(list(mols)))

class CorpusEncoder(ParallelProcessor):

    def __init__(self, corpus_class, corpus_options, n_proc=None, chunk_size=None):
        super().__init__(n_proc, chunk_size)
        self.corpus = corpus_class
        self.options = corpus_options

    def apply(self, mols, collector=None):
        evaluator = ParallelSupplierEvaluator(
            self.corpus,
            kwargs=self.options,
            return_suppliers=True,
            chunk_size=self.chunkSize,
            chunks=self.chunks,
            result_collector=collector
        )
        results = evaluator.apply(mols)
        if results:
            data = []
            voc = None
            for result in results:
                data.extend(result[0])
                if not voc:
                    voc = result[1].getVoc()
                else:
                    voc += result[1].getVoc()
            return data, voc


class RandomTrainTestSplitter(DataSplitter):

    def __init__(self, test_size, max_test_size=1e4, shuffle=True):
        self.testSize = test_size
        self.maxSize = max_test_size
        self.shuffle = shuffle

    def __call__(self, data):
        test_size = min(int(math.ceil(len(data) * self.testSize)), int(self.maxSize))
        if len(data) * self.testSize > int(self.maxSize):
            logger.warning(f'To speed up the training, the test set is reduced to a random sample of {self.maxSize} from the original test!')
        # data = np.asarray(data)
        return train_test_split(data, test_size=test_size, shuffle=self.shuffle)