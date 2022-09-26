"""
interfaces

Created by: Martin Sicho
On: 26.04.22, 13:12
"""
from abc import ABC, abstractmethod

from drugex.logs import logger
from drugex.molecules.interfaces import MolSupplier


class Vocabulary(ABC):
    """
    Definition of the vocabulary interface. All vocabularies contain "words" that are used for encoding and decoding molecules.
    """

    def __init__(self, words):
        self.words = words

    def __add__(self, other):
        return type(self)(other.words + self.words)

    @abstractmethod
    def encode(self, tokens, frags=None):
        pass

    @abstractmethod
    def decode(self, representation):
        pass

    @staticmethod
    @abstractmethod
    def fromFile(path):
        pass

    @abstractmethod
    def toFile(self, path):
        pass

class SequenceVocabulary(Vocabulary, ABC):
    """
    Generic vocabulary for sequence-based models.
    """

    def __init__(self, words, max_len=100, min_len=10):
        """
        Args:
            words: iterable of words in this vocabulary
            max_len: the maximum number of tokens contained in one SMILES
        """

        super().__init__(words)
        self.control = ('_', 'GO', 'EOS')
        self.special = list(self.control) + ['.']
        self.wordSet = set()
        if words:
            self.wordSet = set(x for x in words if x not in self.special)
        self.updateIndex()
        self.max_len = max_len
        self.min_len = min_len

    @abstractmethod
    def splitSequence(self, seq):
        pass

    def toFile(self, path):
        log = open(path, 'w')
        log.write('\n'.join([x for x in self.words if x not in self.special]))
        log.close()

    def addWordsFromSeq(self, seq, ignoreConstraints=False):
        token = self.splitSequence(seq)
        if ignoreConstraints or (self.min_len < len(token) <= self.max_len):
            diff = set(token) - self.wordSet
            if len(diff) > 0:
                self.wordSet.update(diff)
                self.updateIndex()
            return token
        else:
            logger.warning(f"Molecule does not meet min/max words requirements (min: {self.min_len}, max: {self.max_len}). Words found: {set(token)} (occurrence count: {len(token)}). It will be ignored.")
            return None

    def removeIfNew(self, seq, ignoreConstraints=False):
        token = self.splitSequence(seq)
        if ignoreConstraints or (self.min_len < len(token) <= self.max_len):
            diff = set(token) - self.wordSet - {'EOS'}
            if len(diff) > 0:
                logger.warning(f"Tokens: {set(diff)} do not occur in voc. Molecule: {seq} will be ignored.")
                return None
            else:
                return token
        else:
            logger.warning(f"Molecule does not meet min/max words requirements (min: {self.min_len}, max: {self.max_len}). Words found: {set(token)} (occurrence count: {len(token)}). It will be ignored.")
            return None

    def updateIndex(self):
        self.words = self.special + [x for x in sorted(self.wordSet) if x not in self.special]
        self.size = len(self.words)
        self.tk2ix = dict(zip(self.words, range(len(self.words))))
        self.ix2tk = {v: k for k, v in self.tk2ix.items()}

class Corpus(MolSupplier, ABC):
    """
    A `MolSupplier` that generates encoded molecule data from the given input.
    """

    def __init__(self, molecules):
        """

        Args:
            molecules: an `iterable`, `MolSupplier` or a `list`-like data structure to supply molecules
        """
        super().__init__()
        self.molecules = molecules if hasattr(molecules, "__next__") else iter(molecules)

    def next(self):
        return next(self.molecules)

    def convert(self, representation):
        try:
            ret = self.processMolecule(representation)
        except Exception as exp:
            logger.warning(f'Exception occurred when generating corpus data for molecule: {molecule}. Cause:')
            logger.exception(exp)
            return next(self)
        return ret

    @abstractmethod
    def processMolecule(self, molecule):
        """
        Process one molecule.

        Args:
            molecule: a molecule instance (representation depend on the implementation).

        Returns:
            encoded data of the molecule (i.e. data associated with one input sample to the desired DrugEx model)
        """

        pass

    @abstractmethod
    def getVoc(self):
        """
        Corpus should keep track of the 'Vocabulary' used to encode molecules. This method should return its current state.

        Returns:
            currently used `Vocabulary`
        """

        pass



