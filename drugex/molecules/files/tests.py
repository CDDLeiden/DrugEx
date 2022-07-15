"""
tests

Created by: Martin Sicho
On: 12.04.22, 15:07
"""
import os
from unittest import TestCase

import pandas as pd

from drugex.molecules.files.suppliers import CSVSupplier, SDFSupplier
from drugex.molecules.files.interfaces import FileParseException
from drugex.molecules.mol import DrExMol
from drugex.parallel.evaluator import ParallelSupplierEvaluator
from drugex.molecules.suppliers import DataFrameSupplier


class TestFileParsers(TestCase):

    @staticmethod
    def getTestFile(name):
        return f'{os.path.dirname(__file__)}/test_files/{name}'

    def test_csv_correct(self):
        # check correct data
        input_file = self.getTestFile('test.tsv')
        csv_mols = CSVSupplier(
            input_file,
            mol_col='CANONICAL_SMILES',
            sep='\t'
        )
        mols = []
        for idx,mol in enumerate(csv_mols):
            self.assertTrue(isinstance(mol, DrExMol))
            mols.append(mol)
        self.assertTrue(len(mols) == 10)

    def test_csv_bad_row(self):
        # check data with bad row
        bad_input_row  = self.getTestFile('test_bad_row.tsv')
        mols_bad_row =  CSVSupplier(
            bad_input_row,
            mol_col='CANONICAL_SMILES',
            sep='\t'
        )
        mols = []
        for idx,mol in enumerate(mols_bad_row):
            self.assertTrue(isinstance(mol, DrExMol))
            mols.append(mol)
        self.assertTrue(len(mols) == 6) # four bad rows in the file

    def test_csv_bad_header(self):
        # check data with bad header
        bad_input_header = self.getTestFile('test_bad_header.tsv')
        mols_bad_header =  CSVSupplier(
            bad_input_header,
            mol_col='CANONICAL_SMILES',
            sep='\t'
        )
        self.assertRaises(FileParseException,  lambda : [x for x in mols_bad_header])

    def test_csv_duplicates(self):
        # check if duplicates are correctly removed
        duplicates = self.getTestFile('test_duplicates.tsv')
        mols_duplicates = CSVSupplier(
            duplicates,
            mol_col='CANONICAL_SMILES',
            sep='\t',
            hide_duplicates=True
        )
        mols = []
        for idx,mol in enumerate(mols_duplicates):
            self.assertTrue(isinstance(mol, DrExMol))
            mols.append(mol)
        self.assertTrue(len(mols) == 10) # we should only get 10 original molecules

    def test_csv_annotations(self):
        input_file = self.getTestFile('test.tsv')
        cols = ("LOGP", "MWT",)
        mols = CSVSupplier(
            input_file,
            mol_col='CANONICAL_SMILES',
            sep='\t',
            extra_cols=cols
        )
        for idx,mol in enumerate(mols):
            for col in cols:
                self.assertTrue(mol.getAnnotation(col))
                self.assertTrue(col in mol.getMetadata())


    def test_sdf(self):
        mols = SDFSupplier(self.getTestFile('basic.sdf'))
        for idx,mol in enumerate(mols):
            self.assertTrue(isinstance(mol, DrExMol))

    def test_df(self):
        df = pd.read_csv(self.getTestFile('test.tsv'), sep="\t", header=0)
        cols = ("LOGP",)
        df_supplier = DataFrameSupplier(df, mol_col="CANONICAL_SMILES", extra_cols=cols)

        for mol in df_supplier:
            self.assertTrue(isinstance(mol, DrExMol))
            for col in cols:
                self.assertTrue(mol.getAnnotation(col))
                self.assertTrue(col in mol.getMetadata())

    def test_parallel(self):
        df = pd.read_csv(self.getTestFile('test.tsv'), sep="\t", header=0)
        cols = ("LOGP", "MWT",)
        para_supplier = ParallelSupplierEvaluator(
            DataFrameSupplier,
            n_proc=2,
            chunk_size=int(df.shape[0] / 2),
            kwargs={"mol_col" : "CANONICAL_SMILES", "extra_cols": cols}
        )

        ret = []
        def collect(results):
            results = results[0]
            for mol in results:
                ret.append(mol)
                self.assertTrue(isinstance(mol, DrExMol))
                for col in cols:
                    self.assertTrue(mol.getAnnotation(col))
                    self.assertTrue(col in mol.getMetadata())

        para_supplier.apply(df, collect)
        self.assertTrue(len(ret) == 10)

    def test_parallel_with_suppliers(self):
        df = pd.read_csv(self.getTestFile('test.tsv'), sep="\t", header=0)
        para_supplier = ParallelSupplierEvaluator(
            DataFrameSupplier,
            n_proc=2,
            kwargs={"mol_col" : "CANONICAL_SMILES"}
        )

        def collect(ret):
            result = ret[0]
            supplier = ret[1]
            self.assertTrue(len(result) > 0)
            self.assertTrue(isinstance(supplier, DataFrameSupplier))
            self.assertTrue(len(result) == 5)
            for mol in result:
                self.assertTrue(isinstance(mol, DrExMol))

        para_supplier.apply(df, collect)
