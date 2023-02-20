from typing import List

import numpy as np
from rdkit.Chem import MolToSmiles
from RAscore import RAscore_XGB

from drugex.training.scorers.interfaces import Scorer

XGB_MODEL_PATH = None

class RetrosyntheticAccessibilityScorer(Scorer):

    """
    Given a SMILES string, returns a score in [0-1] that indicates how
    likely RA Score predicts it is to find a synthesis route by the underlying CASP tool (AiZynthFinder).
    """

    def __init__(self, modifier=None):
        super().__init__(modifier=modifier)
        """ 
        Initialize the Retrosynthetic Accessibility Scorer.
        
        Parameters
        ----------
        modifier : ScorerModifier
            A modifier that can be used to modify the scores returned by this scorer.
        """
        self.scorer = RAscore_XGB.RAScorerXGB(model_path=XGB_MODEL_PATH)

    def getScores(self, mols: List[str], frags=None):
        """ 
        Get RA scores for a list of molecules.
        
        Parameters
        ----------
        mols : List[str]
            A list of SMILES strings representing molecules.
        frags : List[str], optional
            A list of fragments used to generate the molecules. This is not used by this scorer.
        
        Returns
        -------
        scores : np.ndarray
            A numpy array of scores for the molecules.
        """
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

