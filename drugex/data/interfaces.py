"""
splitting

Created by: Martin Sicho
On: 07.05.22, 15:54
"""
import os
from abc import ABC, abstractmethod

import numpy as np

from drugex.logs import logger
from drugex.parallel.interfaces import ResultCollector

class DataSplitter(ABC):
    """
    Splits input data into multiple parts.
    """

    @abstractmethod
    def __call__(self, data):
        """

        Args:
            data: input data to split

        Returns:
            a tuple of splits

        """

        pass

class DataToLoader(ABC):
    """
    Responsible for the conversion of raw input data into data loaders used by the DrugEx models for training.
    """

    @abstractmethod
    def __call__(self, data, batch_size, vocabulary):
        pass

class DataSet(ResultCollector, ABC):
    """
    Data sets represent encoded input data for the various DrugEx models. Each `DataSet` is associated with a file and also acts as a `ResultCollector` to append data from parallel operations (see `ParallelProcessor`). The `DataSet` is also coupled with the `Vocabulary` used to encode the data in it. However, `Vocabulary` is usually saved in a separate file(s) and needs to be loaded explicitly with `DataSet.readVocs()`.
    """

    def __init__(self, path, autoload=False):
        """
        Initialize this `DataSet`. A path to the associated file must be given. Data is saved to this file upon calling `DataSet.save()`.

        If the associated file already exists, the data is loaded automatically upon initialization.

        Args:
            path: path to the output file.
            no_file_init: if `True`, do not initialize data from an existing file.
        """

        self.outpath = path
        self.data = []
        if autoload and os.path.exists(self.outpath):
            try:
                self.fromFile(self.outpath)
                logger.info(f"Reading data set from an existing file: {self.outpath}. If it is not desired, disable it with: no_file_init=True")
            except Exception as exp:
                logger.warning(f"{self.outpath} -- File already exists, but failed to initialize due to error: {exp}.\n Are you sure you have the right file?\n Initializing an empty data set instead...")


    @abstractmethod
    def getDataFrame(self):
        """
        Get this `DataSet` as a pandas `DataFrame`.

        Returns:
            pandas `DataFrame` representing this instance.
        """

        pass

    @abstractmethod
    def save(self):
        """
        Save the data set to its associated file.

        Returns:
            `None`
        """

        pass

    @abstractmethod
    def getVoc(self):
        """
        Return the `Vocabulary` associated with this data set (should comprise all tokens within it). The vocabulary can be generated from the results collected from `CorpusEncoder` or `FragmentCorpusEncoder` on which this class acts as a collector. Or it can be loaded from files with `DataSet.readVocs()`.

        Returns:
            the associated `Vocabulary` instance.
        """

        pass

    @abstractmethod
    def fromFile(self, path, vocs=tuple(), voc_class=None):
        """
        Initialize this `DataSet` from file and load the associated vocabulary.

        Args:
            path: Path to the encoded data.
            vocs: Paths to the file(s) containing the vocabulary
            voc_class: The `Vocabulary` implementation to initialize.

        Returns:
            `None`
        """

        pass

    def asDataLoader(self, batch_size, splitter=None, split_converter=None, n_samples=-1, n_samples_ratio=None):
        """
        Convert the data in this `DataSet` to a compatible PyTorch `DataLoader`.

        Args:
            batch_size: the desired batch size
            splitter: If a split of the data is required (i.e. training/validation set) a custom `ChunkSplitter` can be supplied. Otherwise, only a single `DataLoader` is created.
            split_converter: a custom `DataToLoader` implementation can be supplied to convert each split to a `DataLoader`. By default, the `DataSet.dataToLoader()` method is used instead.
            n_samples: Number of desired samples in the supplied data before splitting. If "n_samples > 0" and "len(data) < n_samples", the data of the `DataSet` is oversampled to match "len(data) == n_samples"
            n_samples_ratio: If supplied only "n_samples*n_samples_ratio" samples are generated from this `DataSet` before splitting.

        Returns:
            a `tuple` of PyTorch `DataLoader` instances matching the number of splits as defined by the current "splitter". If only one `DataLoader` split data set is created, it returns its `DataLoader` directly.
        """

        split_converter = split_converter if split_converter else self.dataToLoader

        data = self.getData()
        if len(data) == 0:
            raise ValueError("DataSet is not initialized. Cannot convert to data loader.")

        if n_samples_ratio:
            n_samples = int(n_samples*n_samples_ratio)

        if n_samples > 0 and n_samples > len(data):
            logger.info('Replicating original {} samples of data to have set of {} samples.'.format(len(data), n_samples))
            data = np.asarray(data)
            m = int(n_samples/data.shape[0])
            data = data.repeat(m, axis=0)

        results = []
        for split in self.createLoaders(data, batch_size, splitter=splitter, converter=split_converter):
            results.append(split)

        if len(results) == 1:
            return results[0]
        else:
            return results

    @staticmethod
    @abstractmethod
    def dataToLoader(data, batch_size, vocabulary):
        """
        The default method to use to convert data (as returned from `DataSet.getData()`) to a PyTorch `DataLoader`. Basically, mirrors the `DataToLoader` interface.

        Args:
            data: data from `DataSet.getData()`
            batch_size: specified batch size for the `DataLoader`
            vocabulary: a `Vocabulary` instance (in this case should be the same as returned by `DataSet.getVoc()`)

        Returns:
            typically an instance of PyTorch `DataLoader` generated from "data", but depends on the implementation
        """

        pass

    @abstractmethod
    def getData(self):
        """
        Gets the data from this `DataSet` that should be converted to a PyToch `DataLoader`.

        Returns:
            data convertible by the appropriate `DataToLoader` or `DataSet.dataToLoader()`
        """

        pass

    def createLoaders(self, data, batch_size, splitter=None, converter=None):
        """
        Facilitates splitting and conversion of data to `DataLoader`s.

        Args:
            data: data to convert
            batch_size: batch size
            splitter: the `ChunkSplitter` to use
            converter: the `DataToLoader` instance to convert with

        Returns:
            a `list` of created data loaders (same length as the "splitter" return value)
        """

        splits = []
        if splitter:
            splits = splitter(data)
        else:
            splits.append(data)
        return [converter(split, batch_size, self.getVoc()) if converter else split for split in splits]

    def readVocs(self, paths, voc_class, *args, **kwargs):
        """
        Read vocabularies from files and add them together to form the full vocabulary for this `DataSet`.

        Args:
            paths: file paths to vocabulary files
            voc_class: `Vocabulary` implementation to initialize from the files
            *args: any positional arguments passed to the `Vocabulary` constructor besides "words"
            **kwargs: any keyword arguments passed to the `Vocabulary` constructor

        Returns:
            `None`
        """
        if not paths:
            raise ValueError(f'Invalid paths: {paths}.')

        vocs = [voc_class.fromFile(path, *args, **kwargs) for path in paths]
        if len(vocs) > 1:
            voc = sum(vocs[1:], start=vocs[0])
        else:
            voc = vocs[0]

        return self.setVoc(voc)

    @abstractmethod
    def setVoc(self, voc):
        """
        Explicitly set the vocabulary for this `DataSet`.

        Args:
            voc: the new vocabulary

        Returns:
            `None`
        """

        pass

class FragmentPairEncoder(ABC):
    """
    Encode fragments and the associated molecules for the fragment-based DrugEx models.
    """

    @abstractmethod
    def encodeMol(self, mol):
        """
        Encode molecule.

        Args:
            mol: molecule as SMILES

        Returns:
            the encoded representation of this molecule
        """
        pass

    @abstractmethod
    def encodeFrag(self, mol, frag):
        """
        Encode fragment.

        Args:
            mol: the parent molecule of this fragment
            frag: the fragment to encode

        Returns:
            the encoded representation of the fragment
        """
        pass

    @abstractmethod
    def getVoc(self):
        """
        The vocabulary used for encoding.

        Returns:
            a `Vocabulary` instance

        """

        pass
