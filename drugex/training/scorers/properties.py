"""
properties

Created by: Martin Sicho
On: 06.06.22, 20:17
"""
import re

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.GraphDescriptors import BertzCT

from drugex.training.interfaces import Scorer

from rdkit.Chem import Descriptors as desc, Crippen, AllChem, Lipinski
from rdkit.Chem.QED import qed

from drugex.training.scorers.sascorer import calculateScore
import tqdm

from drugex.training.scorers.modifiers import Gaussian
from drugex.utils.fingerprints import get_fingerprint


class Property(Scorer):

    def __init__(self, prop='MW', modifier=None):
        super().__init__(modifier)
        self.prop = prop
        self.prop_dict = {'MW': desc.MolWt,
                          'logP': Crippen.MolLogP,
                          'HBA': AllChem.CalcNumLipinskiHBA,
                          'HBD': AllChem.CalcNumLipinskiHBD,
                          'Rotable': AllChem.CalcNumRotatableBonds,
                          'Amide': AllChem.CalcNumAmideBonds,
                          'Bridge': AllChem.CalcNumBridgeheadAtoms,
                          'Hetero': AllChem.CalcNumHeteroatoms,
                          'Heavy': Lipinski.HeavyAtomCount,
                          'Spiro': AllChem.CalcNumSpiroAtoms,
                          'FCSP3': AllChem.CalcFractionCSP3,
                          'Ring': Lipinski.RingCount,
                          'Aliphatic': AllChem.CalcNumAliphaticRings,
                          'Aromatic': AllChem.CalcNumAromaticRings,
                          'Saturated': AllChem.CalcNumSaturatedRings,
                          'HeteroR': AllChem.CalcNumHeterocycles,
                          'TPSA': AllChem.CalcTPSA,
                          'Valence': desc.NumValenceElectrons,
                          'MR': Crippen.MolMR,
                          'QED': qed,
                          'SA': calculateScore,
                          'Bertz': BertzCT}

    def getScores(self, mols, frags=None):
        scores = np.zeros(len(mols))
        for i, mol in enumerate(mols):
            try:
                scores[i] = self.prop_dict[self.prop](mol)
            except:
                continue
        return scores

    def getKey(self):
        return self.prop

class AtomCounter(Scorer):

    def __init__(self, element: str, modifier=None) -> None:
        """
        Args:
            element: element to count within a molecule
        """
        super().__init__(modifier)
        self.element = element

    def getScores(self, mols, frags=None):
        """
        Count the number of atoms of a given type.
        Args:
            mol: molecule
        Returns:
            The number of atoms of the given type.
        """
        # if the molecule contains H atoms, they may be implicit, so add them
        scores = np.zeros(len(mols))
        for i, mol in enumerate(mols):
            try:
                if self.element in ['', 'H']:
                    mol = Chem.AddHs(mol)
                if self.element == '':
                    scores[i] = len(mol.GetAtoms())
                else:
                    scores[i] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == self.element)
            except: continue
        return scores

    def getKey(self):
        return f"AtomCounter (element={self.element})"

class Isomer(Scorer):
    """
    Scoring function for closeness to a molecular formula.
    The score penalizes deviations from the required number of atoms for each element type, and for the total
    number of atoms.
    F.i., if the target formula is C2H4, the scoring function is the average of three contributions:
    - number of C atoms with a Gaussian modifier with mu=2, sigma=1
    - number of H atoms with a Gaussian modifier with mu=4, sigma=1
    - total number of atoms with a Gaussian modifier with mu=6, sigma=2
    """

    def __init__(self, formula: str, mean_func='geometric', modifier=None) -> None:
        """
        Args:
            formula: target molecular formula
            mean_func: which function to use for averaging: 'arithmetic' or 'geometric'
        """
        super().__init__(modifier)
        self.objs, self.mods = self.scoring_functions(formula)
        self.mean_func = mean_func

    @staticmethod
    def parse_molecular_formula(formula: str):
        """
        Parse a molecular formulat to get the element types and counts.
        Args:
            formula: molecular formula, f.i. "C8H3F3Br"
        Returns:
            A list of tuples containing element types and number of occurrences.
        """
        matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)

        # Convert matches to the required format
        results = []
        for match in matches:
            # convert count to an integer, and set it to 1 if the count is not visible in the molecular formula
            count = 1 if not match[1] else int(match[1])
            results.append((match[0], count))

        return results

    def scoring_functions(self, formula: str):
        element_occurrences = self.parse_molecular_formula(formula)

        total_n_atoms = sum(element_tuple[1] for element_tuple in element_occurrences)

        # scoring functions for each element
        objs = [AtomCounter(element) for element, n_atoms in element_occurrences]
        mods = [Gaussian(mu=n_atoms, sigma=1.0) for element, n_atoms in element_occurrences]
        # scoring functions for the total number of atoms
        objs.append(AtomCounter(''))
        mods.append(Gaussian(mu=total_n_atoms, sigma=2.0))

        return objs, mods

    def getScores(self, mols: list, frags=None) -> np.array:
        # return the average of all scoring functions
        score = np.array([self.mods[i](obj(mols)) for i, obj in enumerate(self.objs)])
        scores = score.prod(axis=0) ** (1.0 / len(score)) if self.mean_func == 'geometric' else np.mean(score, axis=0)
        return scores

    def getKey(self):
        return f"Isomer (mean_func={self.mean_func})"

class Similarity(Scorer):
    def __init__(self, smile, fp_type, modifier=None):
        super().__init__(modifier)
        self.smile = smile
        self.mol = Chem.MolFromSmiles(smile)
        self.fp_type = fp_type
        self.fp = get_fingerprint(self.mol, fp_type=fp_type)

    def getScores(self, mols, frags=None):
        scores = np.zeros(len(mols))
        for i, mol in enumerate(tqdm.tqdm(mols)):
            try:
                fp = get_fingerprint(mol, fp_type=self.fp_type)
                scores[i] = DataStructs.TanimotoSimilarity(self.fp, fp)
            except: continue
        return scores

    def getKey(self):
        return f"Similarity (fp_type={self.fp_type}, smile={self.smile})"


class Scaffold(Scorer):
    def __init__(self, smart, is_match, modifier=None):
        super().__init__(modifier)
        self.smart = smart
        self.frag = Chem.MolFromSmarts(smart)
        self.is_match = is_match

    def getScores(self, mols, frags=None):
        scores = np.zeros(len(mols))
        for i, mol in enumerate(tqdm.tqdm(mols)):
            try:
                match = mol.HasSubstructMatch(self.frag)
                scores[i] = (match == self.is_match)
            except: continue
        return scores

    def getKey(self):
        return f"Scaffold(smart={self.smart},is_match={self.is_match})"
