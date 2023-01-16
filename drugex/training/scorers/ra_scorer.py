from typing import List

import numpy as np
from rdkit.Chem import MolToSmiles
from RAscore import RAscore_XGB

from drugex.training.interfaces import Scorer

XGB_MODEL_PATH = None

class RetrosyntheticAccessibilityScorer(Scorer):

    """
    Given a SMILES string, returns a score in [0-1] that indicates how
    likely RA Score predicts it is to find a synthesis route.
    Args:
        mol (str): a SMILES string representing a molecule.


    Returns: A score between 0 and 1 indicating how likely a synthesis route is to be found by the underlying CASP tool (AiZynthFinder).
    """

    def __init__(self, modifier=None):
        super().__init__(modifier=modifier)
        self.scorer = RAscore_XGB.RAScorerXGB(model_path=XGB_MODEL_PATH)

    def getScores(self, mols: List[str], frags=None):
        scores = np.zeros(shape=len(mols), dtype="float64")
        for i, mol in enumerate(mols):
            if mol is None:
                scores[i] = .0
                continue
            if not isinstance(mol, str):
                mol = MolToSmiles(mol)

            scores[i] = self.scorer.predict(mol)
        return scores

    def getKey(self):
        return "RAscore"

