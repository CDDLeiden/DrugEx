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

    def __init__(self, path):
        self.outpath = path
        self.data = []
        if os.path.exists(self.outpath):
            try:
                self.fromFile(self.outpath)
                logger.info(f"Reading existing file: {self.outpath}")
            except Exception as exp:
                logger.warning(f"{self.outpath} -- File already exists, but failed to initialize due to error: {exp}.\n Are you sure you have the right file? Initializing an empty data set instead...")


    @abstractmethod
    def getDataFrame(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def getVoc(self):
        pass

    @abstractmethod
    def fromFile(self, path, vocs=tuple(), voc_class=None):
        pass

    def asDataLoader(self, batch_size, splitter=None, split_converter=None, n_samples=-1, n_samples_ratio=None):
        split_converter = split_converter if split_converter else self.dataToLoader

        data = self.getData()

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
        pass

    @abstractmethod
    def getData(self):
        pass

    def createLoaders(self, data, batch_size, splitter=None, converter=None):
        splits = []
        if splitter:
            splits = splitter(data)
        else:
            splits.append(data)
        return [converter(split, batch_size, self.getVoc()) if converter else split for split in splits]

    def readVocs(self, paths, voc_class, *args, **kwargs):
        vocs = [voc_class.fromFile(path, *args, **kwargs) for path in paths]
        if len(vocs) > 1:
            voc = sum(vocs[1:], start=vocs[0])
        else:
            voc = vocs[0]

        self.setVoc(voc)

    @abstractmethod
    def setVoc(self, voc):
        pass

class FragmentPairEncoder(ABC):

    @abstractmethod
    def encodeMol(self, mol):
        pass

    @abstractmethod
    def encodeFrag(self, mol, frag):
        pass
