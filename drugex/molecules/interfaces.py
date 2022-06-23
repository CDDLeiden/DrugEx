"""
interfaces

Created by: Martin Sicho
On: 06.04.22, 16:54
"""
from abc import ABC, abstractmethod

from drugex.logs import logger
from drugex.molecules.converters.interfaces import ConversionException


class AnnotationException(Exception):
    pass

class Molecule(ABC):

    @abstractmethod
    def annotate(self, key, value):
        pass

    @abstractmethod
    def getAnnotation(self, key):
        pass

    @abstractmethod
    def getMetadata(self):
        pass

    def __eq__(self, other):
        return self.getUniqueID() == other.getUniqueID()

    def __hash__(self):
        return hash(self.getUniqueID())

    def __str__(self):
        return f"{self.__class__} (ID: {self.getUniqueID()})"

    @abstractmethod
    def getUniqueID(self):
        pass

    @abstractmethod
    def asRDKit(self):
        pass

class MolSupplier(ABC):

    def __iter__(self):
        return self

    def __next__(self):
        next_item = self.next()
        while not next_item:
            next_item = self.next()

        mol_data = next_item
        annotations = dict()
        if type(mol_data) is tuple:
            mol_data = next_item[0]
            annotations = next_item[1]

        # use the converter to convert data to molecule
        mol =  None
        try:
            mol = self.convertMol(mol_data)
        except ConversionException as exp:
            logger.warning(f"An exception occurred when converting molecule data: {mol_data}\n Cause: {exp.__class__}: {exp}")
            return None

        # annotate the instance with metadata
        for key in annotations:
            try:
                self.annotateMol(mol, key, annotations[key])
            except AnnotationException as exp:
                logger.warning(f"An exception occurred when annotating molecule: {mol_data}\n with data: {key}={annotations[key]}\n {exp.__class__}: {exp}")
                continue

        return mol

    @abstractmethod
    def next(self):
        """
        Implement this method so that it provides iteration over molecules item by item. It should fetch next item from a generator, line from a file or next item from a remote API. If there are no more items, raise `StopIteration`.

        Raises:
            StopIteration:  no more items to  return

        Returns:
            molecule: one instance of a molecule
            annotations (optional): molecule associated metadata as a `dict`
        """

        pass

    @abstractmethod
    def convertMol(self, representation):
        """
        Use this to convert a molecule from the supplied representation to a different one. This method is called automatically on the first output of `self.next()`.

        It can also just be identity, but you can use this method to implement different types of standardization for the same type of molecule supplier, for example.

        Parameters
        ----------
        representation -

        Returns
        -------

        molecule - molecule converted to the required representation

        """

        pass

    @abstractmethod
    def annotateMol(self, mol, key, value):
        """
        Use to add metadata to the molecule after conversion. The converted representation will have different mechanisms to store metadata so you might want to override this to match.

        Parameters
        ----------
        mol - converted molecule instance
        key - key to access new metadata value
        value - value of the metadata

        Returns
        -------

        """
        pass

    def toList(self):
        return [x for x in self]


class BaseMolSupplier(MolSupplier, ABC):

    def __init__(
            self,
            converter, # instance returned by converter should implement the Molecule interface
            hide_duplicates=False
    ):
        self.converter = converter
        self.hide_duplicates = hide_duplicates
        self._prev_ids = set()

    def __next__(self):
        mol = super().__next__()
        if not mol:
            return next(self)

        if self.hide_duplicates:
            id_ = mol.getUniqueID()
            if id_ in self._prev_ids:
                return next(self)
            else:
                self._prev_ids.add(id_)

        return mol

    def convertMol(self, representation):
        if self.converter:
            ret = self.converter(representation)
            if not ret:
                raise ConversionException(f"Converter returned an empty molecule instance for representation: {representation}")
            return ret

    def annotateMol(self, mol, key, value):
        mol.annotate(key, value)