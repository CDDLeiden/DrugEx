"""
qsprpred

Created by: Martin Sicho
On: 17.02.23, 13:44
"""
from rdkit import Chem

from drugex.training.scorers.interfaces import Scorer
from qsprpred.models.tasks import ModelTasks

class QSPRPredScorer(Scorer):

    def __init__(self, model):
        self.model = model

    def getScores(self, mols, frags=None):
        if type(mols[0]) != str:
            mols = [Chem.MolToSmiles(mol) for mol in  mols]

        if self.model.task == ModelTasks.REGRESSION:
            return self.model.predictMols(mols)
        else:
            # FIXME: currently we only assume that the model is a binary classifier with the positive class being the last one in the list of probabilities
            return self.model.predictMols(mols, use_probas=True)[:,-1]

    def getKey(self):
        return f"QSPRpred_{self.model.name}"