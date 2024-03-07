import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

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

    def __init__(self, path, rewrite=False, save_voc=True, voc_file=None):
        """
        Initialize this `DataSet`. A path to the associated file must be given. Data is saved to this file upon calling `DataSet.save()`.

        If the associated file already exists, the data is loaded automatically upon initialization.

        Parameters
        ----------
        path : str
            Path to the file to use for this `DataSet`.
        rewrite : bool
            If `True`, the associated file is deleted and a new one is created. If `False`, the data is loaded from the file if it exists.
        save_voc : bool
            If `True`, the vocabulary is saved to a separate file. If `False`, the vocabulary is not saved.
        voc_file : str
            Path to the file to use for the vocabulary. If `None`, the vocabulary is saved to a file with the same name as the data set file but with the `.vocab` extension.
        """

        self.outpath = path
        self.save_voc = save_voc
        self.voc_file = voc_file

        if not os.path.exists(os.path.dirname(self.outpath)):
            os.makedirs(os.path.dirname(self.outpath))
        self.voc = None
        try:
            self.fromFile(self.outpath)
            if rewrite:
                self.reset()
        except FileNotFoundError:
            logger.warning(f"Initialized empty dataset. The data set file does not exist (yet): {self.outpath}. You can add data by calling this instance with the appropriate parameters.")

    def reset(self):
        logger.info(f"Initializing new {self.__class__.__name__} at {self.outpath}...")
        if os.path.exists(self.outpath):
            os.remove(self.outpath)
            logger.info(f"Removed: {self.outpath}")
        voc_path = self.getVocPath()
        if os.path.exists(voc_path):
            os.remove(voc_path)
            logger.info(f"Removed: {voc_path}")

        logger.info(f"{self} initialized.")

    def getVocPath(self):
        if self.voc_file:
            return self.voc_file
        else:
            return f'{self.outpath}.vocab'

    def sendDataToFile(self, data, columns=None):
        header_written = os.path.isfile(self.outpath)
        open_mode = 'a' if header_written else 'w'
        pd.DataFrame(data, columns=columns if columns else [f'Col{x+1}' for x in range(len(data[0]))]).to_csv(
            self.outpath,
            sep='\t',
            index=False,
            header=not header_written,
            mode=open_mode,
            encoding='utf-8'
        )

    def getData(self, chunk_size=None):
        """
        Get this `DataSet` as a pandas `DataFrame`.

        Args:
            chunk_size: the size of the chunk to load at a time

        Returns:
            pandas `DataFrame` representing this instance. If "chunks" is specified an iterator is returned that supplies the chunks.
        """
        kwargs = dict()
        if chunk_size:
            kwargs['chunksize'] = chunk_size

        return pd.read_csv(self.outpath, sep='\t', header=0, **kwargs).to_numpy()

    def updateVoc(self, voc):
        """
        Accept a `Vocabulary` instance and add it to the existing one.

        Args:
            voc: vocabulary to add

        Returns:
            `None`
        """

        if not self.voc:
            self.voc = voc
        else:
            self.voc += voc

        if self.save_voc:
            self.voc.toFile(self.getVocPath())   

    def getVoc(self):
        """
        Return the `Vocabulary` associated with this data set (should comprise all tokens within it). The vocabulary can be generated from the results collected from `CorpusEncoder` or `FragmentCorpusEncoder` on which this class acts as a collector. Or it can be loaded from files with `DataSet.readVocs()`.

        Returns:
            the associated `Vocabulary` instance.
        """

        return self.voc

    def setVoc(self, voc):
        self.voc = voc

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

        self.outpath = path
        if os.path.exists(self.outpath):
            if vocs:
                self.readVocs(vocs, voc_class)
        else:
            raise FileNotFoundError(f"The specified data file does not exist: {self.outpath}")

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
            a `tuple` of the molecule tokens (as determined by the specified vocabulary) and the encoded representation
        """
        pass

    @abstractmethod
    def encodeFrag(self, mol, mol_tokens, frag):
        """
        Encode fragment.

        Args:
            mol: the parent molecule of this fragment
            mol_tokens: the encoded representation of the parent molecule
            frag: the fragment to encode

        Returns:
            the encoded representation of the fragment-molecule pair (i.e. the generated tokens corresponding to both the fragment and the parent molecule)
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
