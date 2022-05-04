"""
corpus

Created by: Martin Sicho
On: 26.04.22, 16:47
"""
from drugex.corpus.interfaces import Corpus
from drugex.corpus.vocabulary import VocSmiles


class SequenceCorpus(Corpus):

    def __init__(self, molecules, vocabulary=VocSmiles(), update_voc=True, out_writer=None, check_unique=True):
        super().__init__(molecules, out_writer)
        self.vocabulary = vocabulary
        self.updateVoc = update_voc
        self.checkUnique = check_unique
        self._unique = set()

    def saveVoc(self, path):
        self.vocabulary.toFile(path)

    def getVoc(self):
        return self.vocabulary

    def processMolecule(self, seq):
        if self.checkUnique and seq in self._unique:
            return None

        token = None
        if self.updateVoc:
            token = self.vocabulary.addWordsFromSeq(seq)
        else:
            token = self.vocabulary.splitSequence(seq)

        if token:
            if self.checkUnique:
                self._unique.add(seq)
            return f"{seq}\t{' '.join(token)}\n"
        else:
            return None