"""
splitting

Created by: Martin Sicho
On: 07.05.22, 15:54
"""
from abc import ABC, abstractmethod

import numpy as np

from drugex.logs import logger
from drugex.parallel.interfaces import ResultCollector


class EncodingCollector(ResultCollector, ABC):

    def __init__(self, outpath):
        self.outpath = outpath

    @abstractmethod
    def getDataFrame(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def getVoc(self):
        pass

class DataSplitter(ABC):

    @abstractmethod
    def __call__(self, data):
        pass

class DataConverter(ABC):

    @abstractmethod
    def __call__(self, data):
        pass

class DataLoaderCreator(DataConverter, ABC):

    def __init__(self, batch_size,vocabulary):
        self.voc = vocabulary
        self.batchSize = batch_size

class DataSet(EncodingCollector, ABC):

    @abstractmethod
    def fromFile(self, path, vocs=tuple(), voc_class=None):
        pass

    def asDataLoader(self, batch_size=None, splitter=None, split_converter=None, n_samples=-1, n_samples_ratio=None):
        if not (split_converter or batch_size):
            raise RuntimeError('You must specify either the batch size or provide a custom split converter.')
        split_converter = split_converter if split_converter else self.getDefaultSplitConverter(batch_size, self.getVoc())

        data = self.getData()

        if n_samples_ratio:
            n_samples = int(n_samples*n_samples_ratio)

        if n_samples > 0 and n_samples > len(data):
            logger.info('Replicating original {} fragment-molecule pairs to have set of {} pairs.'.format(len(data), n_samples))
            data = np.asarray(data)
            m = int(n_samples/data.shape[0])
            data = data.repeat(m, axis=0)

        results = []
        for split in self.generateSplits(data, splitter, split_converter):
            results.append(split)

        if len(results) == 1:
            return results[0]
        else:
            return results

    @abstractmethod
    def getDefaultSplitConverter(self, batch_size, vocabulary):
        pass

    @abstractmethod
    def getData(self):
        pass

    @staticmethod
    def generateSplits(data, splitter=None, split_converter=None):
        splits = []
        if splitter:
            splits = splitter(data)
        else:
            splits.append(data)

        return [split_converter(split) if split_converter else split for split in splits]

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

    @abstractmethod
    def getVoc(self):
        pass

class FragmentPairEncoder(ABC):

    @abstractmethod
    def encodeMol(self, mol):
        pass

    @abstractmethod
    def encodeFrag(self, mol, frag):
        pass
