"""
interfaces

Created by: Martin Sicho
On: 06.04.22, 16:54
"""
from abc import ABC, abstractmethod

from drugex.logs import logger
from drugex.molecules.converters.interfaces import ConversionException

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

class ItemException(Exception):
    pass

class MolSupplier(ABC):
    """
    Generic class that defines the interface for data suppliers from molecules. Implementations of this class
    are used to wrap functionality that can be reused and evaluated in parallel with the `ParallelSupplierEvaluator`.

    Suppliers are simply just Python generators that produce the desired output one item at a time. It is also possible to implement the `convert` method to customize the produced output.
    """

    def __iter__(self):
        return self

    def __next__(self):
        next_item = None
        while not next_item:
            try:
                next_item = self.next()
                next_item = self.convert(next_item)
            except ItemException as exp:
                logger.warning(f"Failed to generate next item in {repr(self)}\n\t Cause: {repr(exp)}")
            except ConversionException as exp:
                logger.warning(f"Failed to convert item {next_item} to the new representation in {repr(self)}\n\t Cause: {repr(exp)}")

        return next_item

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

    def convert(self, representation):
        """
        Can be used to convert a molecule from the supplied representation to a different one. This method is called automatically on the output of `next`. By default, it returns the produced representation as is.

        Parameters
        ----------
        representation - the output produced by `next`

        Returns
        -------

        molecule - molecule converted from "representation" to the desired output

        """

        return representation

    def toList(self):
        return [x for x in self]


class BaseMolSupplier(MolSupplier, ABC):
    """
    Extended `MolSupplier` that produces instances of `DrExMol` that implements identification of duplicates and other useful features.

    """

    def __init__(
            self,
            converter,
            hide_duplicates=False
    ):
        """

        Args:
            converter: a `MolConverter` that produces an instance of `Molecule` from the given input.
            hide_duplicates: If `True`, the returned instances will be tested on uniqueness and only the first encountered item will be processed.
        """
        super().__init__()
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
                logger.info(f"Molecule with ID '{id_}' was skipped because it was already encountered.")
                return next(self)
            else:
                self._prev_ids.add(id_)

        return mol

    def convert(self, representation):
        ret = self.converter(representation)
        if not ret:
            raise ConversionException(f"Converter returned an empty molecule instance for representation: {representation}")
        return ret