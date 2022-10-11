"""
corpus

Created by: Martin Sicho
On: 26.04.22, 16:47
"""
from rdkit import Chem

from drugex.logs import logger
from drugex.data.corpus.interfaces import Corpus
from drugex.data.corpus.vocabulary import VocSmiles, VocGraph


class SequenceCorpus(Corpus):
    """
    A `Corpus` to encode molecules for the sequence-based models.
    """

    def __init__(self, molecules, vocabulary=VocSmiles(), update_voc=True, throw = False, check_unique=True):
        """
        Create a sequence corpus.

        Args:
            molecules: an `iterable`, `MolSupplier` or a `list`-like data structure to supply sequence representations of molecules (i.e. SMILES strings)
            vocabulary: a `SequenceVocabulary` instance to be used for encoding and collecting tokens
            update_voc: `True` if the tokens in the vocabulary should be updated with new tokens derived from the data (the `SequenceVocabulary.addWordsFromSeq()` method is used for splitting instead of doing simply `SequenceVocabulary.splitSequence()`)
            throw: 'True' if molecules that contain tokens that are not in the vocabulary should be thrown out of corpus (the `SequenceVocabulary.removeIfNew()` method is used for splitting instead of doing simply `SequenceVocabulary.splitSequence()`)
            check_unique: Skip identical sequences in "molecules".
        """

        super().__init__(molecules)
        self.vocabulary = vocabulary
        self.updateVoc = update_voc
        self.throw = throw
        if self.updateVoc and self.throw:
            logger.warning(f"update_voc and throw cannot both be true at same time, defaulting to update_voc")
        self.checkUnique = check_unique
        self._unique = set()

    def saveVoc(self, path):
        """
        Save the current state of the vocabulary to a file.

        Args:
            path: Path to the generated file.

        Returns:
            `None`
        """

        self.vocabulary.toFile(path)

    def getVoc(self):
        """
        Return current vocabulary.

        Returns:
            Current vocabulary as a `SequenceVocabulary` instance.
        """

        return self.vocabulary

    def processMolecule(self, seq):
        """
        Generate encoding information for the given molecule sequence.

        Args:
            seq: molecule as a sequence (i.e. SMILES string)

        Returns:
            a `dict` where "seq" is the key to the original sequence and "token" to the generated encoding of this sequence
        """

        if self.checkUnique and seq in self._unique:
            return None

        token = None
        if self.updateVoc:
            token = self.vocabulary.addWordsFromSeq(seq)
        elif self.throw:
            token = self.vocabulary.removeIfNew(seq)
        else:
            token = self.vocabulary.splitSequence(seq)

        if token:
            if self.checkUnique:
                self._unique.add(seq)
            return {'seq' : seq, 'token': ' '.join(token)}
        else:
            return None

class ScaffoldSequenceCorpus(SequenceCorpus):

    def __init__(self, molecules, largest, vocabulary=VocSmiles(), update_voc=True, throw = False, check_unique=True):
        super().__init__(molecules, vocabulary, update_voc, throw, check_unique)
        self.largest = largest
        self.largestToken = self.vocabulary.addWordsFromSeq(self.largest)

    def processMolecule(self, seq):
        if seq == self.largest:
            return None

        processed = super().processMolecule(seq)
        if processed:
            return {
                "mol": self.largestToken,
                "frag": processed['token'].split(' ')
            }
        else:
            return None

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
        code = output[0].reshape(-1).tolist()
        return code

    def getVoc(self):
        return self.voc