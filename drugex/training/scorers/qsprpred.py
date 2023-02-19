"""
qsprpred

Created by: Martin Sicho
On: 17.02.23, 13:44
"""
import numpy as np
from rdkit import Chem

from drugex.training.scorers.interfaces import Scorer
from qsprpred.models.tasks import ModelTasks

class QSPRPredScorer(Scorer):

    def __init__(self, model, invalids_score=0.0):
        self.model = model
        self.invalidsScore = invalids_score

    def getScores(self, mols, frags=None):
        if type(mols[0]) != str:
            mols = [Chem.MolToSmiles(mol) if mol else "INVALID" for mol in mols]

        if self.model.task == ModelTasks.REGRESSION:
            return self.model.predictMols(mols)
        else:
            # FIXME: currently we only assume that the model is a binary classifier with the positive class being the last one in the list of probabilities
            return np.array([probas[-1] if not np.isnan(probas[-1]) else self.invalidsScore for probas in  self.model.predictMols(mols, use_probas=True)])

    def getKey(self):
        return f"QSPRpred_{self.model.name}"