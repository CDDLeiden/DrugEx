"""
defaultdatasets

Created by: Martin Sicho
On: 25.06.22, 19:42
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from drugex.data.corpus.vocabulary import VocSmiles, VocGraph
from drugex.data.interfaces import DataSet, DataLoaderCreator


class SmilesDataSet(DataSet):

    class SplitConverter(DataLoaderCreator):

        def __call__(self, split):
            split = np.asarray(split)[:,1]
            tensor = torch.LongTensor(self.voc.encode([seq.split(' ') for seq in split]))
            loader = DataLoader(tensor, batch_size=self.batchSize, shuffle=True)
            return loader

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

    def getDefaultSplitConverter(self, batch_size, vocabulary):
        return self.SplitConverter(batch_size, vocabulary)

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

    class InOutSplitConverter(DataLoaderCreator):

        def __call__(self, split):
            split = np.asarray(split)
            split_in = self.voc.encode([seq.split(' ') for seq in split[:,0]])
            split_out = self.voc.encode([seq.split(' ') for seq in split[:,1]])
            split_set = TensorDataset(split_in, split_out)
            split_loader = DataLoader(split_set, batch_size=self.batchSize, shuffle=True)
            return split_loader

    class TargetSplitConverter(DataLoaderCreator):

        class TgtData:
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

        def __call__(self, split):
            split = np.asarray(split)
            split = pd.Series(split[:,0]).drop_duplicates()
            split = self.voc.encode([seq.split(' ') for seq in split])
            split = self.TgtData(split, ix=[self.voc.decode(seq, is_tk=False) for seq in split])
            split = DataLoader(split, batch_size=self.batchSize, collate_fn=split.collate_fn)
            return split

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

    def getDefaultSplitConverter(self, batch_size, vocabulary):
        return self.InOutSplitConverter(batch_size, vocabulary)

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

    class SplitConverter(DataLoaderCreator):

        def __call__(self, split):
            split = np.asarray(split)
            split = torch.from_numpy(split).long().view(len(split), self.voc.max_len, -1)
            loader = DataLoader(split, batch_size=self.batchSize, drop_last=False, shuffle=True)
            return loader

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

    def getDefaultSplitConverter(self, batch_size, vocabulary):
        return self.SplitConverter(batch_size, vocabulary)

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