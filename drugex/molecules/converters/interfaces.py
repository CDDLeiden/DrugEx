"""
abc

Created by: Martin Sicho
On: 22.04.22, 10:40
"""
from abc import ABC, abstractmethod


class ConversionException(Exception):
    pass


class MolConverter(ABC):

    @abstractmethod
    def __call__(self, representation):
        pass