from typing import Optional, Union, List

import numpy as np
from rdkit.Chem import MolToSmiles
from RAscore import RAscore_NN, RAscore_XGB

from drugex.training.interfaces import Scorer

NN_MODEL_PATH = None
XGB_MODEL_PATH = None

def calculate_score(mol: str, use_xgb_model: bool = False) -> Optional[float]:
    """
    Given a SMILES string, returns a score in [0-1] that indicates how
    likely RA Score predicts it is to find a synthesis route.
    Args:
        mol (str): a SMILES string representing a molecule.
        use_xgb_model (bool): Determines if the XGB-based model for RA Score
                              should be used instead of NN-based. False by default.

    Returns: A score between 0 and 1 indicating how likely a synthesis route is to be found by the underlying CASP tool (AiZynthFinder).
    """
    scorer = (
        RAscore_XGB.RAScorerXGB(model_path=XGB_MODEL_PATH)
        if use_xgb_model
        else RAscore_NN.RAScorerNN(model_path=NN_MODEL_PATH)
    )

    score = scorer.predict(smiles=mol)
    return score


class RetrosyntheticAccessibilityScorer(Scorer):
    def __init__(self, use_xgb_model: bool = False, modifier=None):
        super().__init__(modifier=modifier)
        self.scorer = (
            RAscore_XGB.RAScorerXGB(model_path=XGB_MODEL_PATH)
            if use_xgb_model
            else RAscore_NN.RAScorerNN(model_path=NN_MODEL_PATH)
        )

    def getScores(self, mols: List[str], frags=None):
        #os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
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


# TODO: Extract to dedicated test module
def test_calc_ra_score():
    omeprazole = "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC"
    score = calculate_score(mol=omeprazole)
    assert 0 <= score <= 1
