"""
interfaces

Created by: Martin Sicho
On: 22.04.22, 11:50
"""
from abc import ABC, abstractmethod

from drugex.molecules.converters.default import SmilesToDrEx
from drugex.molecules.interfaces import BaseMolSupplier


class FileParseException(Exception):
    pass


class TextFileSupplier(BaseMolSupplier, ABC):

    def __init__(
            self,
            path,
            afterParse = None,
            beforeParse = None,
            converter = SmilesToDrEx(),
            hide_duplicates = False
    ):
        super(TextFileSupplier, self).__init__(converter=converter, hide_duplicates=hide_duplicates)
        self.path = path
        self.generator = self.getGenerator()
        self.before = beforeParse
        self.after = afterParse

    def getGenerator(self):
        with open(self.path, mode="r", encoding="utf-8") as text_file:
            for line in text_file:
                yield line

    def next(self):
        ret = next(self.generator)
        if self.before:
            ret = self.before(ret)
        ret = self.parseMol(ret)
        if self.after:
            ret = self.after(ret)
        return ret

    @abstractmethod
    def parseMol(self, item):
        pass