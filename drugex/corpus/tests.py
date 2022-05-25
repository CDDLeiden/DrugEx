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
from drugex.corpus.writers import SequenceFileWriter
from drugex.molecules.converters.standardizers import DrExStandardizer
from drugex.molecules.suppliers import TestSupplier, StandardizedSupplier


class CorpusTest(TestCase):

    @staticmethod
    def getTempFilePath(name):
        return os.path.join(tempfile.gettempdir(),name)

    @staticmethod
    def getMols():
        molecules  = TestSupplier(["CCO", "N[C@@H]1C[C@@H]1C(=O)NCCN1CCNCC1", "CNN[C@@H]1N=CN(C)C(=O)[C@H]1[N+](=O)[O-]", "N[C@@H]1C[C@@H]1C(=O)NCCN1CCNCC1"])
        smiles = StandardizedSupplier(
            molecules,
            standardizer=DrExStandardizer()
        )

        return smiles

    def test_sequence_corpus_smiles(self):
        smiles = self.getMols()
        corpus = SequenceCorpus(smiles)
        lines = []
        for line in corpus:
            self.assertTrue("@" not in line.split()[1])
            lines.append(line)
        self.assertTrue(len(lines) == 2)

    def test_sequence_corpus_file(self):
        smiles = self.getMols()
        out_file = self.getTempFilePath("corpus.txt")
        corpus = SequenceCorpus(
            smiles,
            out_writer=SequenceFileWriter(out_file)
        )

        lines = []
        for line in corpus:
            lines.append(line)

        with open(out_file, "r", encoding="utf-8") as out:
            next(out)
            for idx,line in enumerate(out):
                self.assertTrue(line == lines[idx])

    def test_graph_voc(self):
        voc = VocGraph()
        df = voc.toDataFrame()
        self.assertTrue(len(df) == len(set(voc.words) - set(voc.control)))
