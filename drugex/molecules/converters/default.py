"""
converters

Created by: Martin Sicho
On: 21.04.22, 12:20
"""
from rdkit import Chem

from drugex.molecules.converters.interfaces import ConversionException, MolConverter
from drugex.molecules.mol import DrExMol


class SmilesToDrEx(MolConverter):

    def __call__(self, smiles):
        mol = None
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception as exp:
            raise ConversionException(exp)
        if not mol:
            raise ConversionException(f"Failed to parse SMILES: {smiles}")
        return DrExMol(mol, identifier=smiles.strip())


class RDKitToDrEx(MolConverter):

    def __call__(self, rd_mol):
        if not rd_mol:
            raise ConversionException("RDKit supplied an empty molecule.")
        else:
            return DrExMol(rd_mol)

class DrExToSMILES(MolConverter):

    def __call__(self, drugex_mol):
        return drugex_mol.smiles