"""
standardizers

Created by: Martin Sicho
On: 21.04.22, 12:19
"""
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from drugex.logs import logger
from drugex.molecules.converters.default import DrExToSMILES, Identity, SmilesToDrEx
from drugex.molecules.converters.interfaces import MolConverter, ConversionException
from drugex.molecules.mol import DrExMol


class DrExStandardizer(MolConverter):

    outputConvertors = {
        'SMILES' : DrExToSMILES(),
        'DrExMol' : Identity()
    }

    inputConvertors = {
        'SMILES' : SmilesToDrEx(),
        'DrExMol' : Identity()
    }

    def __init__(self, output='SMILES', input='DrExMol'):
        if callable(output):
            self.outputConvertor = output
        else:
            self.outputConvertor = self.outputConvertors[output]

        if callable(input):
            self.inputConvertor = input
        else:
            self.inputConvertor = self.inputConvertors[input]

    def __call__(self, mol):
        mol = self.inputConvertor(mol)
        rd_mol = mol.asRDKit()
        try:
            charger = rdMolStandardize.Uncharger()
            chooser = rdMolStandardize.LargestFragmentChooser()
            disconnector = rdMolStandardize.MetalDisconnector()
            normalizer = rdMolStandardize.Normalizer()
            carbon = Chem.MolFromSmarts('[#6]')
            salts = Chem.MolFromSmarts('[Na,Zn]')
            rd_mol = disconnector.Disconnect(rd_mol)
            rd_mol = normalizer.normalize(rd_mol)
            rd_mol = chooser.choose(rd_mol)
            rd_mol = charger.uncharge(rd_mol)
            rd_mol = disconnector.Disconnect(rd_mol)
            rd_mol = normalizer.normalize(rd_mol)
            smileR = Chem.MolToSmiles(rd_mol, 0)
            # remove SMILES that do not contain carbon
            if len(rd_mol.GetSubstructMatches(carbon)) == 0:
                raise StandardizationException(f"No carbon in SMILES: {smileR}")
            # remove SMILES that still contain salts
            if len(rd_mol.GetSubstructMatches(salts)) > 0:
                raise StandardizationException(f"Salt removal failed: {smileR}")
            new_mol = DrExMol(Chem.MolFromSmiles(smileR))
            for key in mol.getMetadata():
                new_mol.annotate(key, mol.getAnnotation(key))
            return self.outputConvertor(new_mol)
        except Exception as exp:
            logger.exception(f'Parsing Error: {Chem.MolToSmiles(rd_mol)}')
            raise StandardizationException(exp)


class StandardizationException(ConversionException):
    pass


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