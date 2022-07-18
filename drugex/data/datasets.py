"""
defaultdatasets

Created by: Martin Sicho
On: 25.06.22, 19:42
"""

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

    def __init__(self, path, voc=None):
        super().__init__(path)
        self.voc = self.setVoc(voc if voc else VocSmiles())

    @staticmethod
    def dataToLoader(data, batch_size, vocabulary):
        split = np.asarray(data)[:,1]
        tensor = torch.LongTensor(vocabulary.encode([seq.split(' ') for seq in split]))
        loader = DataLoader(tensor, batch_size=batch_size, shuffle=True)
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
        self.sendDataToFile([(x['seq'], x['token']) for x in result[0]], columns=self.columns)

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
            dataset = np.asarray(data)[:,0]
            dataset = pd.Series(dataset).drop_duplicates()
            dataset = [seq.split(' ') for seq in dataset]
            dataset = vocabulary.encode(dataset)
            dataset = self.TgtData(dataset, ix=[vocabulary.decode(seq, is_tk=False) for seq in dataset])
            dataset = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
            return dataset

    def __init__(self, path):
        super().__init__(path)
        self.voc = VocSmiles()

    def __call__(self, result):
        """
        Collect encoded data from the given results. Designated to be used as the collector of encodings for `FragmentCorpusEncoder`.

        Args:
            result: `tuple` with two items -- a `list` of `tuple`s as supplied by: `FragmentPairsEncodedSupplier` and the `FragmentPairsEncodedSupplier` itself

        Returns:
            `None`
        """

        self.updateVoc(result[1].encoder.getVoc())
        self.sendDataToFile(
                [
                    (
                        " ".join(x[1]),
                        " ".join(x[0])
                    )
                    for x in result[0] if x[0] and x[1]
                ],
                columns=self.columns
            )

    @staticmethod
    def dataToLoader(data, batch_size, vocabulary):
        arr = np.asarray(data)
        _in = vocabulary.encode([seq.split(' ') for seq in arr[:,0]])
        _out = vocabulary.encode([seq.split(' ') for seq in arr[:,1]])
        del arr
        dataset = TensorDataset(_in, _out)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class SmilesScaffoldDataSet(SmilesFragDataSet):

    def __call__(self, result):
        if result[0]:
            self.updateVoc(result[1].getVoc())
            self.sendDataToFile(
                [
                    (
                        " ".join(x['frag']),
                        " ".join(x['mol'])
                    )
                    for x in result[0] if x['mol'] and x['frag']
                ],
                columns=self.columns
            )


class GraphFragDataSet(DataSet):
    """
    `DataSet` to manage the fragment-molecule pair encodings for the graph-based model (`GraphModel`).
    """

    def __init__(self, path):
        super().__init__(path)
        self.voc = VocGraph()

    def __call__(self, result):
        """
        Collect encoded data from the given results. Designated to be used as the collector of encodings for `FragmentCorpusEncoder`.

        Args:
            result: `tuple` with two items -- a `list` of `tuple`s as supplied by: `FragmentPairsEncodedSupplier` and the `FragmentPairsEncodedSupplier` itself

        Returns:
            `None`
        """

        self.updateVoc(result[1].encoder.getVoc())
        self.sendDataToFile([x[1] for x in result[0]], columns=self.getColumns())

    def getColumns(self):
        return ['C%d' % d for d in range(self.getVoc().max_len * 5)]

    @staticmethod
    def dataToLoader(data, batch_size, vocabulary):
        dataset = np.asarray(data)
        dataset = torch.from_numpy(dataset).long().view(len(dataset), vocabulary.max_len, -1)
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)
        return loader


class GraphScaffoldDataSet(GraphFragDataSet):

    def __call__(self, result):
        self.sendDataToFile(result[0], columns=self.getColumns())