"""
tests

Created by: Martin Sicho
On: 18.05.22, 11:49
"""
from unittest import TestCase

import pandas as pd

from drugex.datasets.fragments import FragmentPairsEncodedSupplier, SequenceFragmentEncoder, GraphFragmentEncoder
from drugex.molecules.converters.fragmenters import Fragmenter
from drugex.molecules.converters.standardizers import DrExStandardizer
from drugex.molecules.fragments import FragmentPairsSupplier
from drugex.molecules.parallel import ParallelSupplierEvaluator, ListCollector
from drugex.molecules.suppliers import TestSupplier, StandardizedSupplier


class FragmentPairs(TestCase):

    def getPairs(self):
        mols = TestSupplier(['CN1[C@H]2CC[C@@H]1[C@H]([C@H](C2)OC(=O)C3=CC=CC=C3)C(=O)OC', 'N[C@@H]1C[C@@H]1C(=O)NCCN1CCNCC1', 'BrCCO'])
        smiles = StandardizedSupplier(
            mols,
            standardizer=DrExStandardizer()
        )

        pairs = FragmentPairsSupplier(smiles, fragmenter=Fragmenter(4, 4, 'brics')).get()
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
            self.assertTrue(encoded['mol_encoded'][-1] == 'EOS')
            self.assertTrue(encoded['frag_encoded'][-1] == 'EOS')

    def test_pair_encode_smiles_parallel(self):
        pairs_df = self.getPairs().sample(100, replace=True)
        evaluator = ParallelSupplierEvaluator(
            FragmentPairsEncodedSupplier,
            kwargs={'encoder': SequenceFragmentEncoder()},
            return_suppliers=True
        )

        for result in evaluator.get(pairs_df):
            data = result[0]
            for item in data:
                self.assertTrue(item['mol_encoded'][-1] == 'EOS')
                self.assertTrue(item['frag_encoded'][-1] == 'EOS')
            voc = result[1].encoder.getVoc()
            self.assertTrue('Br' not in voc.words)

    def test_pair_encode_graph(self):
        pairs_df = self.getPairs().sample(100, replace=True)
        evaluator = ParallelSupplierEvaluator(
            FragmentPairsEncodedSupplier,
            kwargs={'encoder': GraphFragmentEncoder()},
            return_unique=False
        )

        for result in evaluator.get(pairs_df):
            self.assertTrue(result['mol'] == result['mol_encoded'])
            self.assertTrue(type(result['frag_encoded']) == list)