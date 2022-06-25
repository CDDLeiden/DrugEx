"""
tests

Created by: Martin Sicho
On: 28.04.22, 14:08
"""
import os.path
import tempfile
from unittest import TestCase

from drugex.corpus.corpus import SequenceCorpus
from drugex.corpus.vocabulary import VocGraph
from drugex.molecules.converters.standardizers import DefaultStandardizer
from drugex.molecules.suppliers import StandardizedSupplier


class CorpusTest(TestCase):

    @staticmethod
    def getTempFilePath(name):
        return os.path.join(tempfile.gettempdir(),name)

    @staticmethod
    def getMols():
        smiles = StandardizedSupplier(
            ["CCO", "N[C@@H]1C[C@@H]1C(=O)NCCN1CCNCC1", "CNN[C@@H]1N=CN(C)C(=O)[C@H]1[N+](=O)[O-]", "N[C@@H]1C[C@@H]1C(=O)NCCN1CCNCC1"],
            standardizer=DefaultStandardizer()
        )

        return smiles

    def test_sequence_corpus_smiles(self):
        smiles = self.getMols()
        corpus = SequenceCorpus(smiles)
        lines = []
        for line in corpus:
            self.assertTrue("@" not in line['seq'])
            self.assertTrue("@" not in line['token'])
            lines.append(line)
        self.assertTrue(len(lines) == 2)

    def test_sequence_corpus_file(self):
        smiles = self.getMols()
        corpus = SequenceCorpus(
            smiles
        )

        count = 0
        for line in corpus:
            seq = line['seq']
            token = line['token']
            self.assertTrue(token.replace(' ', '').startswith(seq))
            count += 1
        self.assertTrue(count > 0)

    def test_graph_voc(self):
        voc = VocGraph()
        df = voc.toDataFrame()
        self.assertTrue(len(df) == len(set(voc.words) - set(voc.control)))

