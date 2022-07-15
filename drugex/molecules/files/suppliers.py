"""
files

Created by: Martin Sicho
On: 06.04.22, 17:05
"""
import gzip
import logging

from rdkit import Chem

from drugex.molecules.converters.default import SmilesToDrEx, RDKitToDrEx
from drugex.molecules.files.interfaces import FileParseException, TextFileSupplier


class CSVSupplier(TextFileSupplier):

    def __init__(
            self,
            path,
            mol_col='SMILES',
            sep=',',
            extra_cols = tuple(),
            afterParse = None,
            beforeParse = None,
            converter = SmilesToDrEx(),
            hide_duplicates=False
    ):
        super(CSVSupplier, self).__init__(path, afterParse, beforeParse, converter, hide_duplicates)
        self.separator = sep
        self.molecule_column = mol_col
        self.header  = None
        self.metadata_columns = extra_cols
        self.header =  dict()
        self.n_cols = None
        self.current_line = None

    def parseHeader(self, cols):
        self.n_cols = len(cols)
        cols =  [x.strip() for x in cols]
        if self.molecule_column not in cols:
            raise FileParseException(f"Could not find molecule  column: {self.molecule_column}")
        for idx,header_item in enumerate(cols):
            if header_item == self.molecule_column or header_item in self.metadata_columns:
                self.header[header_item] = idx

        return self.header

    def checkLineIntegrity(self, line, col_data, warn=True):
        if self.n_cols != len(col_data):
            if warn:
                logging.warning(f'Skipping line: {self.current_line}. Found different number of data columns than expected ({len(col_data)}  != {self.n_cols}). Offending data: {line}')
            return False

        return True

    def getColumnData(self, line):
        return line.split(self.separator)

    def getMolData(self, col_data):
        return col_data[self.header[self.molecule_column]]

    def getMolMetadata(self, col_data):
        metadata = dict()
        for col in self.metadata_columns:
            metadata[col] = col_data[self.header[col]]
        return metadata

    def parseMol(self, line):
        if not self.current_line:
            self.current_line = 0
        self.current_line +=  1

        # extract data for all columns
        col_data = self.getColumnData(line)

        # detect header
        if not self.header:
            self.parseHeader(col_data)
            return None

        # check integrity of current line
        if not self.checkLineIntegrity(line, col_data):
            return None

        # extract desired data
        return self.getMolData(col_data), self.getMolMetadata(col_data)

class SDFSupplier(TextFileSupplier):

    def __init__(
            self,
            path,
            afterParse = None,
            beforeParse = None,
            converter = RDKitToDrEx(),
            hide_duplicates = False
    ):
        super().__init__(
            path,
            afterParse,
            beforeParse,
            converter,
            hide_duplicates
        )

    def getGenerator(self):
        path = self.path
        if path.endswith('.gz'):
            return Chem.ForwardSDMolSupplier(gzip.open(path))
        return Chem.ForwardSDMolSupplier(path)

    def parseMol(self, item):
        return item