import os
from unittest import TestCase

import pandas as pd

from drugex.molecules.mol import DrExMol
from drugex.molecules.suppliers import ListSupplier
from drugex.parallel.evaluator import ParallelSupplierEvaluator


class TestFileParsers(TestCase):

    @staticmethod
    def getTestFile(name):
        return f'{os.path.dirname(__file__)}/test_files/{name}'

    def test_parallel(self):
        smiles = pd.read_csv(self.getTestFile('test.tsv'), sep="\t", header=0).CANONICAL_SMILES.tolist()
        para_supplier = ParallelSupplierEvaluator(
            ListSupplier,
            n_proc=2,
            chunk_size=int(len(smiles) / 2),
            kwargs={
                "hide_duplicates": True
            }
        )

        ret = []
        def collect(results):
            results = results[0]
            for mol in results:
                ret.append(mol)
                self.assertTrue(isinstance(mol, DrExMol))

        para_supplier.apply(smiles, collect)
        self.assertTrue(len(ret) == 10)

    def test_parallel_with_suppliers(self):
        smiles = pd.read_csv(self.getTestFile('test.tsv'), sep="\t", header=0).CANONICAL_SMILES.tolist()
        para_supplier = ParallelSupplierEvaluator(
            ListSupplier,
            n_proc=2,
            kwargs={
                "hide_duplicates": True
            }
        )

        def collect(ret):
            result = ret[0]
            supplier = ret[1]
            self.assertTrue(len(result) > 0)
            self.assertTrue(isinstance(supplier, ListSupplier))
            #
            for mol in result:
                self.assertTrue(isinstance(mol, DrExMol))

            return len(result)

        ret = para_supplier.apply(smiles, collect)
        self.assertTrue(sum(ret) == 10)
