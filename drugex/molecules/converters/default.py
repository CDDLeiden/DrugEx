"""
converters

Created by: Martin Sicho
On: 21.04.22, 12:20
"""
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

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

class CleanSMILES(MolConverter):

    def __init__(self, is_deep=True):
        self.deep  = is_deep

    def __call__(self, smile):
        orig_smile = smile
        smile = smile.replace('[O]', 'O').replace('[C]', 'C') \
        .replace('[N]', 'N').replace('[B]', 'B') \
        .replace('[2H]', '[H]').replace('[3H]', '[H]')
        try:
            mol = Chem.MolFromSmiles(smile)
            if self.deep:
                mol = rdMolStandardize.ChargeParent(mol)
            smileR = Chem.MolToSmiles(mol, 0)
            smile = Chem.CanonSmiles(smileR)
        except:
            raise ConversionException(f"Cleanup error: {orig_smile}")
        return smile