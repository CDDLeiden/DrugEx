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
    """
    This processor translates input molecules to representations that can be used directly as input to both sequence- and graph-based models. It works by evaluating a `Corpus` in parallel on the input data.
    """

    def __init__(self, corpus_class, corpus_options, n_proc=None, chunk_size=None):
        """
        Initialize from a `Corpus` class and its options.

        Args:
            corpus_class: a `Corpus` implementation used in the evaluation.
            corpus_options: constructor options for the "corpus_class" except for the first positional argument representing the input data.
            n_proc: number of processes to use for the evaluation.
            chunk_size: maximum chunk of data to use per process (can help save memory).
        """

        super().__init__(n_proc, chunk_size)
        self.corpus = corpus_class
        self.options = corpus_options

    def apply(self, mols, collector=None):
        """
        Apply the encoder to given molecules.

        Args:
            mols: `list` or similar data structure with molecules (representation of each molecule depends on the `Corpus` implementation used).
            collector: custom `ResultCollector` to use as a callback to customize how results are collected. If it is specified, this method returns None. A `tuple` with two items is passed to the collector: the encoded data and the associated `Corpus` instance used to calculate it.

        Returns:
            a `tuple`, first item is the concatenated encoded data and second is the associated vocabulary from the `Corpus` instances used for calculations.
        """

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
    """
    Simple splitter to facilitate a random split into training and test set with the option to fix the maximum size of the test set.
    """

    def __init__(self, test_size, max_test_size=1e4, shuffle=True):
        """

        Args:
            test_size (`float`): Size of the test set as a ratio of the original data (i.e. 0.1 for 10%).
            max_test_size: maximum number of samples in the test data. If the "test_size" implies a test set larger than "max_test_size", the test set size is capped at this value.
            shuffle: Choose to shuffle the data before splitting or not (default: `True`).
        """

        self.testSize = test_size
        self.maxSize = max_test_size
        self.shuffle = shuffle

    def __call__(self, data):
        """
        Apply a split to the data.

        Args:
            data: data to split.

        Returns:
            a `tuple`, first item is the training data and second the test set
        """

        test_size = min(int(math.ceil(len(data) * self.testSize)), int(self.maxSize))
        if len(data) * self.testSize > int(self.maxSize):
            logger.warning(f'To speed up the training, the test set is reduced to a random sample of {self.maxSize} from the original test!')
        # data = np.asarray(data)
        return train_test_split(data, test_size=test_size, shuffle=self.shuffle)