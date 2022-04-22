"""
standardizers

Created by: Martin Sicho
On: 21.04.22, 12:19
"""
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from drugex.molecules.converters.default import DrExToSMILES
from drugex.molecules.converters.interfaces import MolConverter
from drugex.molecules.mol import DrExMol


class DrExStandardizer(MolConverter):

    convertors = {
        'SMILES' : DrExToSMILES(),
        'DrExMol' : lambda x:x
    }

    def __init__(self, output='SMILES'):
        if callable(output):
            self.converter = output
        else:
            self.converter = self.convertors[output]

        self.charger = rdMolStandardize.Uncharger()
        self.chooser = rdMolStandardize.LargestFragmentChooser()
        self.disconnector = rdMolStandardize.MetalDisconnector()
        self.normalizer = rdMolStandardize.Normalizer()
        self.carbon = Chem.MolFromSmarts('[#6]')
        self.salts = Chem.MolFromSmarts('[Na,Zn]')

    def __call__(self, mol):
        mol = mol.asRDKit()
        try:
            mol = self.disconnector.Disconnect(mol)
            mol = self.normalizer.normalize(mol)
            mol = self.chooser.choose(mol)
            mol = self.charger.uncharge(mol)
            mol = self.disconnector.Disconnect(mol)
            mol = self.normalizer.normalize(mol)
            smileR = Chem.MolToSmiles(mol, 0)
            # remove SMILES that do not contain carbon
            if len(mol.GetSubstructMatches(self.carbon)) == 0:
                raise StandardizationException(f"No carbon in SMILES: {smileR}")
            # remove SMILES that still contain salts
            if len(mol.GetSubstructMatches(self.salts)) > 0:
                raise StandardizationException(f"Salt removal failed: {smileR}")
            new_mol = DrExMol(Chem.MolFromSmiles(smileR))
            return self.converter(new_mol)
        except Exception as exp:
            raise StandardizationException(exp)


class StandardizationException(Exception):
    pass