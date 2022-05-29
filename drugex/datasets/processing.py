"""
processing

Created by: Martin Sicho
On: 27.05.22, 10:16
"""

import numpy as np
import pandas as pd

from drugex.datasets.fragments import FragmentPairsEncodedSupplier, FragmentPairsSupplier
from drugex.datasets.interfaces import EncodingCollector
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
        self.pairsSplitter = pairs_splitter if pairs_splitter else lambda x : (pd.DataFrame(x, columns=['Frags', 'Smiles']),)

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
                'mol_col' : self.pairsSplitter.smilesCol,
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

class SmilesDataCollector(EncodingCollector):

    def __init__(self, outpath):
        super().__init__(outpath)
        self.voc = None
        self.data = []

    def getDataFrame(self, columns=('Smiles', 'Token')):
        return pd.DataFrame(self.data, columns=columns)

    def save(self, columns=('Smiles', 'Token')):
        self.getDataFrame(columns).to_csv(self.outpath, sep='\t', index=False)

    def getVoc(self):
        return self.voc

    def __call__(self, result):
        self.data.extend([(x['seq'], x['token']) for x in result[0]])

        voc = result[1].getVoc()
        if not self.voc:
            self.voc = voc
        else:
            self.voc += voc

class SmilesFragDataCollector(EncodingCollector):

    def __init__(self, outpath):
        super().__init__(outpath)
        self.codes = []
        self.voc = None

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

    def getDataFrame(self, columns=('Input', 'Output')):
        return pd.DataFrame(self.codes, columns=columns)

    def save(self, columns=('Input', 'Output')):
        self.getDataFrame(columns).to_csv(self.outpath, sep='\t', index=False)

    def getVoc(self):
       return self.voc

class GraphFragDataCollector(EncodingCollector):

    def __init__(self, outpath):
        super().__init__(outpath)
        self.voc = None
        self.codes = []

    def __call__(self, result):
        self.codes.extend(x[1] for x in result[0])
        voc = result[1].encoder.getVoc()
        if not self.voc:
            self.voc = voc
        else:
            self.voc += voc

    def getDataFrame(self):
        columns = ['C%d' % d for d in range(self.voc.max_len * 5)]
        return pd.DataFrame(self.codes, columns=columns)

    def save(self):
        self.getDataFrame().to_csv(self.outpath, sep='\t', index=False)

    def getVoc(self):
       return self.voc