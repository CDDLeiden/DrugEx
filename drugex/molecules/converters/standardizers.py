from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from drugex.logs import logger
from drugex.molecules.converters.interfaces import MolConverter, ConversionException

class StandardizationException(ConversionException):
    """
    Custom exception class to recognize and catch standardization errors.
    """

    pass

class DefaultStandardizer(MolConverter):
    """
    Original standardization implementation from the original DrugEx v3.

    """

    def __call__(self, smiles):
        """
        Takes smiles of the input molecule and converts it to a standardized represenation.

        Raises:
            StandardizationException: thrown when the standardizer encountered a failure

        Args:
            smiles: input molecule as SMILES

        Returns:
            converted SMILES as `str`

        """

        try:
            rd_mol = Chem.MolFromSmiles(smiles)
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
            return smileR
        except StandardizationException as exp:
            raise exp
        except Exception as exp:
            logger.exception(f'Unexpected error during standardization: {smiles}')
            raise StandardizationException(exp)

class CleanSMILES(MolConverter):
    """
    Converter used to clean SMILES strings at some places. At the moment the reasons for its existence are unclear...
    """

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