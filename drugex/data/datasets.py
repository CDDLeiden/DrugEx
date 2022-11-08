"""
defaultdatasets

Created by: Martin Sicho
On: 25.06.22, 19:42
"""

from itertools import chain
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

from drugex.data.corpus.vocabulary import VocSmiles, VocGraph
from drugex.data.interfaces import DataSet, DataToLoader


class SmilesDataSet(DataSet):
    """
    `DataSet` that holds the encoded SMILES representations of molecules for the single-network sequence-based DrugEx model (`RNN`).
    """

    columns=('Smiles', 'Token') # column names to use for the data frame

    def __init__(self, path, voc=None, rewrite=False):
        super().__init__(path, rewrite=rewrite)
        self.setVoc(voc if voc else VocSmiles(False))

    @staticmethod
    def dataToLoader(data, batch_size, vocabulary):
        dataset = torch.from_numpy(data).long().view(len(data), vocabulary.max_len)
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)
        return loader

    def __call__(self, result):
        """
        Collect results from `SequenceCorpus`.

        Args:
            result: A list of items generated from `SequenceCorpus` to be added to the current `DataSet`.

        Returns:
            `None`
        """

        self.updateVoc(result[1].getVoc())
        self.sendDataToFile(result[0], columns=self.getColumns())

    def getColumns(self):
        return ['C%d' % d for d in range(self.getVoc().max_len)]

    def readVocs(self, paths, voc_class, *args, **kwargs):
        super().readVocs(paths, voc_class=voc_class, *args, encode_frags=False, **kwargs)


class SmilesFragDataSet(DataSet):
    """
    `DataSet` that holds the encoded SMILES representations of fragment-molecule pairs for the sequence-based encoder-decoder type of DrugEx models.
    """

    columns=('Input', 'Output')

    class TargetCreator(DataToLoader):
        """
        Old creator for test data that currently is no longer being used. Saved here for future reference.
        """

        class TgtData(Dataset):
            def __init__(self, seqs, ix, max_len=100):
                self.max_len = max_len
                self.index = np.array(ix)
                self.map = {idx: i for i, idx in enumerate(self.index)}
                self.seq = seqs

            def __getitem__(self, i):
                seq = self.seq[i]
                return i, seq

            def __len__(self):
                return len(self.seq)

            def collate_fn(self, arr):
                collated_ix = np.zeros(len(arr), dtype=int)
                collated_seq = torch.zeros(len(arr), self.max_len).long()
                for i, (ix, tgt) in enumerate(arr):
                    collated_ix[i] = ix
                    collated_seq[i, :] = tgt
                return collated_ix, collated_seq

        def __call__(self, data, batch_size, vocabulary):
            dataset = data[:,0]
            dataset = pd.Series(dataset).drop_duplicates()
            dataset = [seq.split(' ') for seq in dataset]
            dataset = vocabulary.encode(dataset)
            dataset = self.TgtData(dataset, ix=[vocabulary.decode(seq, is_tk=False) for seq in dataset])
            dataset = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
            return dataset

    def __init__(self, path, voc=None, rewrite=False):
        super().__init__(path, rewrite=rewrite)
        self.voc = voc if voc else VocSmiles(True)

    def __call__(self, result):
        """
        Collect encoded data from the given results. Designated to be used as the collector of encodings for `FragmentCorpusEncoder`.

        Args:
            result: `tuple` with two items -- a `list` of `tuple`s as supplied by: `FragmentPairsEncodedSupplier` and the `FragmentPairsEncodedSupplier` itself

        Returns:
            `None`
        """

        self.updateVoc(result[1].encoder.getVoc())
        self.sendDataToFile([list(chain.from_iterable(x)) for x in result[0]], columns=self.getColumns())

    def createLoaders(self, data, batch_size, splitter=None, converter=None):
        splits = []
        if splitter:
            splits = splitter(data)
        else:
            splits.append(data)
        return [converter(split, batch_size, self.getVoc()) if converter else split for split in splits]

    @staticmethod
    def dataToLoader(data, batch_size, vocabulary):
        # Split into molecule and fragment embedding
        dataset = TensorDataset(torch.from_numpy(data[:, :vocabulary.max_len]).long().view(len(data), vocabulary.max_len),
                                torch.from_numpy(data[:, vocabulary.max_len:]).long().view(len(data), vocabulary.max_len))
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)
        return loader

    def getColumns(self):
        return ['C%d' % d for d in range(self.getVoc().max_len * 2)]

    def readVocs(self, paths, voc_class, *args, **kwargs):
        super().readVocs(paths, voc_class=voc_class, *args, encode_frags=True, **kwargs)


class GraphFragDataSet(DataSet):
    """
    `DataSet` to manage the fragment-molecule pair encodings for the graph-based model (`GraphModel`).
    """

    def __init__(self, path, voc=None, rewrite=False):
        super().__init__(path, rewrite=rewrite)
        self.voc = voc if voc else VocGraph()

    def __call__(self, result):
        """
        Collect encoded data from the given results. Designated to be used as the collector of encodings for `FragmentCorpusEncoder`.

        Args:
            result: `tuple` with two items -- a `list` of `tuple`s as supplied by: `FragmentPairsEncodedSupplier` and the `FragmentPairsEncodedSupplier` itself

        Returns:
            `None`
        """

        self.updateVoc(result[1].encoder.getVoc())
        data = [x[0] for x in result[0]]
        self.sendDataToFile(data, columns=self.getColumns())

    def getColumns(self):
        return ['C%d' % d for d in range(self.getVoc().max_len * 5)]

    @staticmethod
    def dataToLoader(data, batch_size, vocabulary):
        dataset = torch.from_numpy(data).long().view(len(data), vocabulary.max_len, -1)
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)
        return loader