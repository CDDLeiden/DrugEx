from abc import ABC, abstractmethod


class ConversionException(Exception):
    pass


class MolConverter(ABC):

    @abstractmethod
    def __call__(self, representation):
        pass