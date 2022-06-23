"""
tests

Created by: Martin Sicho
On: 18.05.22, 11:49
"""
from unittest import TestCase

import pandas as pd

from drugex.corpus.corpus import SequenceCorpus
from drugex.corpus.vocabulary import VocSmiles, VocGraph
from drugex.datasets.fragments import FragmentPairsEncodedSupplier, SequenceFragmentEncoder, GraphFragmentEncoder, \
    FragmentPairsSplitter, FragmentPairsSupplier
from drugex.datasets.processing import Standardization, MoleculeEncoder, FragmentEncoder, SmilesFragDataSet, \
    GraphFragDataSet, SmilesDataSet
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

        pairs = FragmentPairsSupplier(smiles, fragmenter=Fragmenter(4, 4, 'brics')).toList()
        pairs_flat = []
        for lst in pairs:
            pairs_flat.extend(lst)
        return pd.DataFrame(pairs_flat, columns=['Frags', 'Smiles'])

    def test_pair_encode_smiles(self):
        pairs_df = self.getPairs()
        encoder = SequenceFragmentEncoder()
        encoded_pairs = FragmentPairsEncodedSupplier(
            pairs_df,
            encoder=encoder
        )

        voc = encoder.getVoc()
        self.assertTrue('Br' not in voc.words)
        for encoded in encoded_pairs:
            self.assertTrue(encoded[0][-1] == 'EOS')
            self.assertTrue(encoded[1][-1] == 'EOS')

    def test_pair_encode_smiles_parallel(self):
        pairs_df = self.getPairs().sample(100, replace=True)
        evaluator = ParallelSupplierEvaluator(
            FragmentPairsEncodedSupplier,
            kwargs={'encoder': SequenceFragmentEncoder()},
            return_suppliers=True
        )

        for result in evaluator.apply(pairs_df):
            data = result[0]
            for item in data:
                self.assertTrue(item[0][-1] == 'EOS')
                self.assertTrue(item[1][-1] == 'EOS')
            voc = result[1].encoder.getVoc()
            self.assertTrue('Br' not in voc.words)

    def test_pair_encode_graph(self):
        pairs_df = self.getPairs().sample(100, replace=True)
        evaluator = ParallelSupplierEvaluator(
            FragmentPairsEncodedSupplier,
            kwargs={'encoder': GraphFragmentEncoder()},
            return_unique=False
        )

        for result in evaluator.apply(pairs_df):
            self.assertTrue(type(result[1]) == list)

class ProcessingTests(TestCase):

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
        standardized = standardizer.applyTo(originals)
        self.assertTrue(len(standardized) == len(originals))
        for mol in standardized:
            self.assertTrue(mol in expected)

    def test_smiles_encoder(self):
        mols = self.getTestMols()
        mols.append('NC1CC1C(=O)NCCN1CCNCC1CCBr') # add smiles with new element
        encoder = MoleculeEncoder(
            SequenceCorpus,
            {
                "vocabulary" : VocSmiles()
            },
            n_proc=2, chunk_size=2
        )

        results, voc = encoder.applyTo(mols)
        self.assertTrue('R' in voc.words)
        self.assertTrue(len(results) == len(mols))
        for result in results:
            self.assertTrue(result['token'].endswith("EOS"))
            self.assertTrue(result['seq'])

        # with collector
        collector = SmilesDataSet('a')
        encoder.applyTo(mols, collector=collector)
        voc = collector.getVoc()
        self.assertTrue('R' in voc.words)
        df = collector.getDataFrame()
        self.assertTrue(df.shape == (11, 2))

    def test_smiles_frag_encoder(self):
        mols = self.getTestMols()
        mols.append('NC1CC1C(=O)NCCN1CCNCC1CCBr') # add smiles with new element
        encoder = FragmentEncoder(
            fragmenter=Fragmenter(4, 4, 'brics'),
            encoder=SequenceFragmentEncoder(
                VocSmiles()
            ),
            pairs_splitter=FragmentPairsSplitter(0.1, 1e4),
            n_proc=2,
            chunk_size=2
        )

        # without collectors
        splits, voc = encoder.applyTo(mols)
        self.assertTrue('R' in voc.words)
        for split in splits:
            for result in split:
                self.assertTrue(result[0][-1] == result[1][-1] == 'EOS')

        # with collectors
        collectors = [SmilesFragDataSet(x) for x in ('a', 'b', 'c')]
        encoder.applyTo(mols, encodingCollectors=collectors)
        for collector in collectors:
            df = collector.getDataFrame()
            self.assertTrue(df.Input[0].endswith('EOS') and df.Output[0].endswith('EOS'))


    def test_graph_frag_encoder(self):
        mols = self.getTestMols()
        encoder = FragmentEncoder(
                fragmenter=Fragmenter(4, 4, 'brics'),
                encoder=GraphFragmentEncoder(
                    VocGraph(n_frags=4)
                ),
                pairs_splitter=FragmentPairsSplitter(0.1, 1e4),
                n_proc=2,
                chunk_size=2
        )

        # with collectors
        collectors = [GraphFragDataSet(x) for x in ('a', 'b', 'c')]
        encoder.applyTo(mols, encodingCollectors=collectors)
        for collector in collectors:
            df = collector.getDataFrame()
            self.assertTrue(df.columns[0][0] == 'C')

        # without collectors
        results_splits, voc = encoder.applyTo(mols)
        self.assertTrue(type(voc) == VocGraph)
        for split in results_splits:
            for result in split:
                self.assertTrue(type(result[0]) == str)
                self.assertTrue(len(result[1]) == 400)