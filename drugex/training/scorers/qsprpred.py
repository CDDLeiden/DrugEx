"""
qsprpred

Created by: Martin Sicho
On: 17.02.23, 13:44
"""
import numpy as np
from rdkit import Chem

from drugex.logs import logger
from drugex.training.scorers.interfaces import Scorer
from qsprpred.models.tasks import ModelTasks

class QSPRPredScorer(Scorer):

    def __init__(self, model, invalids_score=0.0, modifier=None):
        super(QSPRPredScorer, self).__init__(modifier)
        self.model = model
        self.invalidsScore = invalids_score

    def getScores(self, mols, frags=None):
        if type(mols[0]) != str:
            invalids = 0
            for idx, mol in enumerate(mols):
                try:
                    Chem.SanitizeMol(mol)
                    mol = Chem.MolToSmiles(mol) if mol and mol.GetNumAtoms() > 1 else "INVALID"
                except Exception as exp:
                    logger.error(f"Error processing molecule: {Chem.MolToSmiles(mol)} {exp}")
                    mol = "INVALID"
                if mol == "INVALID":
                    invalids += 1
                mols[idx] = mol

            if invalids == len(mols):
                return np.array([self.invalidsScore] * len(mols))

        if self.model.task == ModelTasks.REGRESSION:
            return self.model.predictMols(mols)
        else:
            # FIXME: currently we only assume that the model is a binary classifier with the positive class being the last one in the list of probabilities
            return np.array([probas[-1] if not np.isnan(probas[-1]) else self.invalidsScore for probas in  self.model.predictMols(mols, use_probas=True)])

    def getKey(self):
        return f"QSPRpred_{self.model.name}"