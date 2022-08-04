"""
tests

Created by: Martin Sicho
On: 18.05.22, 11:49
"""
import tempfile
from unittest import TestCase

import pandas as pd

from drugex.data.corpus.corpus import SequenceCorpus
from drugex.data.corpus.vocabulary import VocSmiles, VocGraph
from drugex.data.fragments import FragmentPairsEncodedSupplier, SequenceFragmentEncoder, GraphFragmentEncoder, \
    FragmentPairsSplitter, FragmentPairsSupplier, FragmentCorpusEncoder
from drugex.data.processing import Standardization, CorpusEncoder
from drugex.data.datasets import SmilesDataSet, SmilesFragDataSet, GraphFragDataSet
from drugex.molecules.converters.fragmenters import Fragmenter
from drugex.molecules.converters.standardizers import DefaultStandardizer
from drugex.parallel.evaluator import ParallelSupplierEvaluator
from drugex.molecules.suppliers import StandardizedSupplier


class FragmentPairs(TestCase):

    def getPairs(self):
        smiles = StandardizedSupplier(
            ['CN1[C@H]2CC[C@@H]1[C@H]([C@H](C2)OC(=O)C3=CC=CC=C3)C(=O)OC', 'N[C@@H]1C[C@@H]1C(=O)NCCN1CCNCC1', 'BrCCO'],
            standardizer=DefaultStandardizer()
        )

        return FragmentPairsSupplier(smiles, fragmenter=Fragmenter(4, 4, 'brics')).toList()

    def test_pair_encode_smiles(self):
        pairs = self.getPairs()
        encoder = SequenceFragmentEncoder()
        encoded_pairs = FragmentPairsEncodedSupplier(
            pairs,
            encoder=encoder
        )

        voc = encoder.getVoc()
        self.assertTrue('Br' not in voc.words)
        for encoded in encoded_pairs:
            self.assertTrue(encoded[0][-1] == 'EOS')
            self.assertTrue(encoded[1][-1] == 'EOS')

    def test_pair_encode_smiles_parallel(self):
        pairs_df = pd.Series(self.getPairs()).sample(100, replace=True)
        evaluator = ParallelSupplierEvaluator(
            FragmentPairsEncodedSupplier,
            kwargs={'encoder': SequenceFragmentEncoder()}
        )

        def collect(result):
            data = result[0]
            for item in data:
                self.assertTrue(item[0][-1] == 'EOS')
                self.assertTrue(item[1][-1] == 'EOS')
            voc = result[1].encoder.getVoc()
            self.assertTrue('Br' not in voc.words)

        evaluator.apply(pairs_df, collect)

    def test_pair_encode_graph(self):
        pairs_df = pd.Series(self.getPairs()).sample(100, replace=True)
        evaluator = ParallelSupplierEvaluator(
            FragmentPairsEncodedSupplier,
            kwargs={'encoder': GraphFragmentEncoder()},
        )

        def collect(result):
            self.assertTrue(type(result[0]) == list)
            self.assertTrue(type(result[1]) == FragmentPairsEncodedSupplier)

        evaluator.apply(pairs_df, collect)

class ProcessingTests(TestCase):

    @staticmethod
    def getRandomFile():
        return tempfile.NamedTemporaryFile().name

    def getStandardizationMols(self):
        mols_orig = [
                "c1ccccc1CN(=O)",
                "CC(=O)[O-]",
                "CCCCn1cc[n+](c1)C.F[P-](F)(F)(F)(F)F"
        ]
        mols_expected = (
            "O=NCc1ccccc1", # canonical
            "CC(=O)O", # without charge
            "CCCCn1cc[n+](C)c1", # remove salt
        )

        return mols_orig, mols_expected

    def getTestMols(self):
        mols = ['CN1[C@H]2CC[C@@H]1[C@H]([C@H](C2)OC(=O)C3=CC=CC=C3)C(=O)OC', 'N[C@@H]1C[C@@H]1C(=O)NCCN1CCNCC1', 'CN1[C@H]2CC[C@@H]1[C@H]([C@H](C2)OC(=O)C3=CC=CC=C3)C(=O)OC', 'N[C@@H]1C[C@@H]1C(=O)NCCN1CCNCC1', 'CN1[C@H]2CC[C@@H]1[C@H]([C@H](C2)OC(=O)C3=CC=CC=C3)C(=O)OC', 'N[C@@H]1C[C@@H]1C(=O)NCCN1CCNCC1', 'CN1[C@H]2CC[C@@H]1[C@H]([C@H](C2)OC(=O)C3=CC=CC=C3)C(=O)OC', 'N[C@@H]1C[C@@H]1C(=O)NCCN1CCNCC1', 'CN1[C@H]2CC[C@@H]1[C@H]([C@H](C2)OC(=O)C3=CC=CC=C3)C(=O)OC', 'N[C@@H]1C[C@@H]1C(=O)NCCN1CCNCC1', ]
        smiles = StandardizedSupplier(
            mols,
            standardizer=DefaultStandardizer()
        ).toList()
        return smiles

    def test_standardization(self):
        originals, expected = self.getStandardizationMols()
        standardizer = Standardization(n_proc=2, chunk_size=2)

        def collect(result):
            standardized = result[0]
            sup = result[1]
            self.assertTrue(type(sup) == StandardizedSupplier)
            for mol in standardized:
                self.assertTrue(mol in expected)

        standardizer.apply(originals, collect)

    def test_smiles_encoder(self):
        mols = self.getTestMols()
        mols.append('NC1CC1C(=O)NCCN1CCNCC1CCBr') # add smiles with new element
        encoder = CorpusEncoder(
            SequenceCorpus,
            {
                "vocabulary" : VocSmiles()
            },
            n_proc=2, chunk_size=2
        )

        # with collector
        collector = SmilesDataSet(self.getRandomFile())
        encoder.apply(mols, collector=collector)
        voc = collector.getVoc()
        self.assertTrue('R' in voc.words)
        df = collector.getData()
        self.assertTrue(df.shape == (11, 2))

    def test_smiles_frag_encoder(self):
        mols = self.getTestMols()
        mols.append('NC1CC1C(=O)NCCN1CCNCC1CCBr') # add smiles with new element
        encoder = FragmentCorpusEncoder(
            fragmenter=Fragmenter(4, 4, 'brics'),
            encoder=SequenceFragmentEncoder(
                VocSmiles()
            ),
            pairs_splitter=FragmentPairsSplitter(0.1, 1e4),
            n_proc=2,
            chunk_size=2
        )

        # with collectors
        collectors = [SmilesFragDataSet(x) for x in (self.getRandomFile(), self.getRandomFile())]
        encoder.apply(mols, encodingCollectors=collectors)
        for collector in collectors:
            df = collector.getData()
            self.assertTrue(df.Input[0].endswith('EOS') and df.Output[0].endswith('EOS'))

    def test_frag_suppliers(self):
        pairs = FragmentPairsSupplier(self.getTestMols(), Fragmenter(4, 4, 'brics')).toList()
        encoded = FragmentPairsEncodedSupplier(pairs, GraphFragmentEncoder(VocGraph(n_frags=4)))
        count = 0
        for item in encoded:
            self.assertTrue(len(item) == 2)
            self.assertTrue(type(item[0]) == str)
            self.assertTrue(type(item[1]) == list)
            self.assertTrue(type(item[1][0]) == int)
            count+=1
        self.assertTrue(count == len(pairs))

    def test_graph_frag_encoder(self):
        mols = self.getTestMols()
        encoder = FragmentCorpusEncoder(
                fragmenter=Fragmenter(4, 4, 'brics'),
                encoder=GraphFragmentEncoder(
                    VocGraph(n_frags=4)
                ),
                pairs_splitter=FragmentPairsSplitter(0.1, 1e4),
                n_proc=2,
                chunk_size=2
        )

        # with collectors
        collectors = [GraphFragDataSet(x) for x in (self.getRandomFile(), self.getRandomFile())]
        fragment_collector = FragmentCorpusEncoder.FragmentPairsCollector()
        encoder.apply(mols, fragmentPairsCollector=fragment_collector, encodingCollectors=collectors)
        self.assertTrue(len(fragment_collector.getList()) == (len(collectors[0].getData()) + len(collectors[1].getData())))
        for collector in collectors:
            df = collector.getData()
            self.assertTrue(df.columns[0][0] == 'C')