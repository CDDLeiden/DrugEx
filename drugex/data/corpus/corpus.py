"""
corpus

Created by: Martin Sicho
On: 26.04.22, 16:47
"""
from rdkit import Chem

from drugex.data.corpus.interfaces import Corpus
from drugex.data.corpus.vocabulary import VocSmiles, VocGraph


class SequenceCorpus(Corpus):

    def __init__(self, molecules, vocabulary=VocSmiles(), update_voc=True, check_unique=True):
        super().__init__(molecules)
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
            return {'seq' : seq, 'token': ' '.join(token)}
        else:
            return None

class ScaffoldSequenceCorpus(SequenceCorpus):

    def __init__(self, molecules, largest, vocabulary=VocSmiles(), update_voc=True, check_unique=True):
        super().__init__(molecules, vocabulary, update_voc, check_unique)
        self.largest = largest
        self.largestToken = self.vocabulary.addWordsFromSeq(self.largest)

    def processMolecule(self, seq):
        if seq == self.largest:
            return None

        processed = super().processMolecule(seq)
        return {
            "mol": self.largestToken,
            "frag": processed['token'].split(' ')
        }

class ScaffoldGraphCorpus(Corpus):

    def __init__(self, molecules, largest, vocabulary=VocGraph()):
        super().__init__(molecules)
        self.largest = largest
        mol = Chem.MolFromSmiles(self.largest)
        total = mol.GetNumBonds()
        if total >= 75:
            raise ValueError("To create dataset largest smiles has to have less than 75 bonds'")
        self.voc = vocabulary

    def processMolecule(self, molecule):
        if molecule == self.largest:
            return None
        output = self.voc.encode([self.largest], [molecule])
        f, s = self.voc.decode(output)
        assert self.largest == s[0]
        code = output[0].reshape(-1).tolist()
        return code

    def getVoc(self):
        return self.voc