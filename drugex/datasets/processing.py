"""
processing

Created by: Martin Sicho
On: 27.05.22, 10:16
"""
import os.path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from drugex.corpus.vocabulary import VocSmiles, VocGraph
from drugex.datasets.fragments import FragmentPairsEncodedSupplier, FragmentPairsSupplier, FragmentPairsSplitterBase
from drugex.datasets.interfaces import EncodingCollector, DataSet, DataConverter, DataLoaderCreator
from drugex.logs import logger
from drugex.parallel.evaluator import ParallelSupplierEvaluator
from drugex.parallel.interfaces import MoleculeProcessor
from drugex.molecules.converters.standardizers import DrExStandardizer
from drugex.molecules.suppliers import StandardizedSupplier


class Standardization(MoleculeProcessor):

    def __init__(self, standardizer=DrExStandardizer(input='SMILES', output='SMILES'), n_proc=None, chunk_size=None):
        super().__init__(n_proc, chunk_size)
        self.standardizer = standardizer

    def applyTo(self, data, collector=None):
        standardizer = ParallelSupplierEvaluator(
            StandardizedSupplier,
            kwargs={
                "standardizer": self.standardizer
            },
            **self.getApplierArgs(data, collector)
        )
        return standardizer.apply(np.asarray(list(data)))

class MoleculeEncoder(MoleculeProcessor):

    def __init__(self, corpus_class, corpus_options, n_proc=None, chunk_size=None):
        super().__init__(n_proc, chunk_size)
        self.corpus = corpus_class
        self.options = corpus_options

    def applyTo(self, mols, collector=None):
        evaluator = ParallelSupplierEvaluator(
            self.corpus,
            kwargs=self.options,
            return_suppliers=True,
            **self.getApplierArgs(mols, collector)
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

class FragmentEncoder(MoleculeProcessor):

    def __init__(self, fragmenter, encoder, pairs_splitter=None, n_proc=None, chunk_size=None):
        super().__init__(n_proc, chunk_size)
        self.fragmenter = fragmenter
        self.encoder = encoder
        self.pairsSplitter = pairs_splitter if pairs_splitter else FragmentPairsSplitterBase()

    def getFragmentPairs(self, mols, collector):
        evaluator = ParallelSupplierEvaluator(
            FragmentPairsSupplier,
            kwargs={
                "fragmenter" : self.fragmenter
            },
            return_unique=False,
            always_return=True,
            **self.getApplierArgs(mols, collector)
        )
        results = []
        for result in evaluator.apply(mols):
            results.extend(result)
        return results

    def splitFragmentPairs(self, pairs):
        return self.pairsSplitter(pairs)

    def encodeFragments(self, pairs, collector):
        evaluator = ParallelSupplierEvaluator(
            FragmentPairsEncodedSupplier,
            kwargs={
                'encoder': self.encoder,
                'mol_col' : self.pairsSplitter.molCol,
                'frags_col': self.pairsSplitter.fragsCol
            },
            return_unique=False,
            return_suppliers=True,
            **self.getApplierArgs(pairs, collector)
        )
        results = evaluator.apply(pairs)
        if results:
            voc = None
            data = []
            for result in results:
                data.extend(result[0])
                if not voc:
                    voc = result[1].encoder.getVoc()
                else:
                    voc += result[1].encoder.getVoc()

            return data, voc


    def applyTo(self, mols, fragmentCollector=None, encodingCollectors=None):
        pairs = self.getFragmentPairs(mols, fragmentCollector)
        ret = []
        ret_voc = None
        splits = self.splitFragmentPairs(pairs)
        if encodingCollectors and len(encodingCollectors) != len(splits):
            raise RuntimeError(f'The number of encoding collectors must match the number of splits: {len(encodingCollectors)} != {len(splits)}')
        for split_idx, split in enumerate(splits):
            result = self.encodeFragments(split, encodingCollectors[split_idx] if encodingCollectors else None)
            if result:
                result, voc = result
                ret.append(result)
                if not ret_voc:
                    ret_voc = voc
                else:
                    ret_voc += voc
        if ret or ret_voc:
            return ret, ret_voc

class SmilesDataSet(DataSet):

    class SplitConverter(DataLoaderCreator):

        def __call__(self, split):
            split = np.asarray(split)[:,1]
            tensor = torch.LongTensor(self.voc.encode([seq.split(' ') for seq in split]))
            loader = DataLoader(tensor, batch_size=self.batchSize, shuffle=True)
            return loader

    def __init__(self, outpath):
        super().__init__(outpath)
        self.voc = VocSmiles()
        if os.path.exists(outpath):
            self.fromFile(outpath)
        else:
            self.data = []

    def getDataFrame(self, columns=('Smiles', 'Token')):
        return pd.DataFrame(self.data, columns=columns)

    def save(self, columns=('Smiles', 'Token')):
        self.getDataFrame(columns).to_csv(self.outpath, sep='\t', index=False)

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


    def __init__(self, outpath, columns=('Input', 'Output')):
        super().__init__(outpath)
        self.voc = VocSmiles()
        self.columns = columns
        if os.path.exists(outpath):
            try:
                self.fromFile(outpath)
            except Exception as exp:
                logger.warning(f"{outpath} -- File already exists, but failed to initialize due to error: {exp}.\n Are you sure you have the right file? Initializing an empty data set instead...")
        else:
            self.codes = []

    def __call__(self, result):
        self.codes.extend(
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
        return pd.DataFrame(self.codes, columns=self.columns)

    def save(self,):
        self.getDataFrame().to_csv(self.outpath, sep='\t', index=False)

    def getData(self):
        return self.codes

    def getVoc(self):
       return self.voc

    def getDefaultSplitConverter(self, batch_size, vocabulary):
        return self.InOutSplitConverter(batch_size, vocabulary)

    def setVoc(self, voc):
        self.voc = voc

    def fromFile(self, path, vocs=tuple(), voc_class=None):
        self.codes = pd.read_csv(path, header=0, sep='\t', usecols=self.columns).values.tolist()

        if vocs and voc_class:
            self.voc = self.readVocs(vocs, voc_class)

class SmilesScaffoldDataSet(SmilesFragDataSet):

    def __call__(self, result):
        if result[0]:
            self.codes.extend(
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

class GraphDataSet(DataSet):

    class SplitConverter(DataLoaderCreator):

        def __call__(self, split):
            split = np.asarray(split)
            split = torch.from_numpy(split).long().view(len(split), self.voc.max_len, -1)
            loader = DataLoader(split, batch_size=self.batchSize, drop_last=False, shuffle=True)
            return loader

    def __init__(self, outpath):
        super().__init__(outpath)
        if os.path.exists(outpath):
            self.fromFile(outpath)
        else:
            self.codes = []
        self.voc = VocGraph()

    def __call__(self, result):
        self.codes.extend(result[0])
        # voc = result[1].getVoc()
        # self.addVoc(voc)

    def addVoc(self, voc):
        if not self.voc:
            self.voc = voc
        else:
            self.voc += voc

    def getDataFrame(self):
        columns = ['C%d' % d for d in range(self.voc.max_len * 5)]
        return pd.DataFrame(self.codes, columns=columns)

    def save(self):
        self.getDataFrame().to_csv(self.outpath, sep='\t', index=False)

    def getData(self):
        return self.codes

    def getDefaultSplitConverter(self, batch_size, vocabulary):
        return self.SplitConverter(batch_size, vocabulary)

    def getVoc(self):
       return self.voc

    def setVoc(self, voc):
        self.voc = voc

    def fromFile(self, path, vocs=tuple(), voc_class=None):
        self.codes = pd.read_csv(path, header=0, sep='\t').values.tolist()

        if vocs and voc_class:
            self.voc = self.readVocs(vocs, voc_class)

class GraphFragDataSet(GraphDataSet):

    def __init__(self, outpath):
        super().__init__(outpath)

    def __call__(self, result):
        self.codes.extend(x[1] for x in result[0])
        # voc = result[1].encoder.getVoc()
        # self.addVoc(voc)
