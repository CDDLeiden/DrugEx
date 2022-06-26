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

    columns=('Smiles', 'Token')

    def __init__(self, path, voc=VocSmiles()):
        super().__init__(path)
        self.voc = voc

    def getDataFrame(self):
        return pd.DataFrame(self.data, columns=self.columns)

    def save(self):
        self.getDataFrame().to_csv(self.outpath, sep='\t', index=False)

    def getVoc(self):
        return self.voc

    def getData(self):
        return self.data

    def setVoc(self, voc):
        self.voc = voc

    @staticmethod
    def dataToLoader(data, batch_size, vocabulary):
        split = np.asarray(data)[:,1]
        tensor = torch.LongTensor(vocabulary.encode([seq.split(' ') for seq in split]))
        loader = DataLoader(tensor, batch_size=batch_size, shuffle=True)
        return loader

    def __call__(self, result):
        self.data.extend([(x['seq'], x['token']) for x in result[0]])

        voc = result[1].getVoc()
        if not self.voc:
            self.voc = voc
        else:
            self.voc += voc

    def fromFile(self, path, vocs=tuple(), voc_class=None, smiles_col='Smiles', token_col='Token'):
        self.data = pd.read_csv(path, header=0, sep='\t', usecols=[smiles_col, token_col]).values.tolist()

        if vocs and voc_class:
            self.voc = self.readVocs(vocs, voc_class)


class SmilesFragDataSet(DataSet):

    class TargetCreator(DataToLoader):
        """
        Old creator that currently is not being used. Saved here just for reference.
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

    columns=('Input', 'Output')

    def __init__(self, path):
        super().__init__(path)
        self.voc = VocSmiles()

    def __call__(self, result):
        self.data.extend(
                [
                    (
                        " ".join(x[1]),
                        " ".join(x[0])
                    )
                    for x in result[0] if x[0] and x[1]
                ]
            )
        voc = result[1].encoder.getVoc()
        if not self.voc:
            self.voc = voc
        else:
            self.voc += voc

    def getDataFrame(self):
        return pd.DataFrame(self.data, columns=self.columns)

    def save(self):
        self.getDataFrame().to_csv(self.outpath, sep='\t', index=False)

    def getData(self):
        return self.data

    def getVoc(self):
       return self.voc

    @staticmethod
    def dataToLoader(data, batch_size, vocabulary):
        arr = np.asarray(data)
        _in = vocabulary.encode([seq.split(' ') for seq in arr[:,0]])
        _out = vocabulary.encode([seq.split(' ') for seq in arr[:,1]])
        del arr
        dataset = TensorDataset(_in, _out)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def setVoc(self, voc):
        self.voc = voc

    def fromFile(self, path, vocs=tuple(), voc_class=None):
        self.data = pd.read_csv(path, header=0, sep='\t', usecols=self.columns).values.tolist()

        if vocs and voc_class:
            self.voc = self.readVocs(vocs, voc_class)


class SmilesScaffoldDataSet(SmilesFragDataSet):

    def __call__(self, result):
        if result[0]:
            self.data.extend(
                [
                    (
                        " ".join(x['frag']),
                        " ".join(x['mol'])
                    )
                    for x in result[0] if x['mol'] and x['frag']
                ]
            )

            voc = result[1].getVoc()
            if not self.voc:
                self.voc = voc
            else:
                self.voc += voc


class GraphFragDataSet(DataSet):

    def __init__(self, path):
        super().__init__(path)
        self.voc = VocGraph()

    def __call__(self, result):
        self.data.extend(x[1] for x in result[0])

    def addVoc(self, voc):
        if not self.voc:
            self.voc = voc
        else:
            self.voc += voc

    def getDataFrame(self):
        columns = ['C%d' % d for d in range(self.voc.max_len * 5)]
        return pd.DataFrame(self.data, columns=columns)

    def save(self):
        self.getDataFrame().to_csv(self.outpath, sep='\t', index=False)

    def getData(self):
        return self.data

    @staticmethod
    def dataToLoader(data, batch_size, vocabulary):
        dataset = np.asarray(data)
        dataset = torch.from_numpy(dataset).long().view(len(dataset), vocabulary.max_len, -1)
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)
        return loader

    def getVoc(self):
       return self.voc

    def setVoc(self, voc):
        self.voc = voc

    def fromFile(self, path, vocs=tuple(), voc_class=None):
        self.data = pd.read_csv(path, header=0, sep='\t').values.tolist()

        if vocs and voc_class:
            self.voc = self.readVocs(vocs, voc_class)


class GraphScaffoldDataSet(GraphFragDataSet):

    def __call__(self, result):
        self.data.extend(result[0])