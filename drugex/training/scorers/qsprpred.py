"""
qsprpred

Created by: Martin Sicho
On: 17.02.23, 13:44
"""
import numpy as np
from drugex.logs import logger
from drugex.training.scorers.interfaces import Scorer
from qsprpred.models.tasks import ModelTasks
from rdkit import Chem


class QSPRPredScorer(Scorer):

    def __init__(self, model, invalids_score=0.0, modifier=None, **kwargs):
        super(QSPRPredScorer, self).__init__(modifier)
        self.model = model
        self.invalidsScore = invalids_score
        self.kwargs = kwargs

    def getScores(self, mols, frags=None):
        parsed_mols = []
        if not isinstance(mols[0], str):
            invalids = 0
            for mol in mols:
                parsed_mol = None
                try:
                    parsed_mol = Chem.MolToSmiles(mol) if mol and mol.GetNumAtoms() > 1 else "INVALID"
                    if parsed_mol and parsed_mol != "INVALID":
                        Chem.SanitizeMol(Chem.MolFromSmiles(parsed_mol))
                except Exception as exp:
                    logger.debug(f"Error processing molecule: {parsed_mol} -> \n\t {exp}")
                    parsed_mol = "INVALID"
                if parsed_mol == "INVALID":
                    invalids += 1
                parsed_mols.append(parsed_mol)

            if invalids == len(parsed_mols):
                return np.array([self.invalidsScore] * len(parsed_mols))
        else:
            parsed_mols = mols

        if self.model.task == ModelTasks.REGRESSION:
            return self.model.predictMols(parsed_mols, **self.kwargs)
        else:
            # FIXME: currently we only assume that the model is a binary classifier
            # with the positive class being the last one in the list of probabilities
            return np.array(
                [probas[-1]
                 if not np.isnan(probas[-1]) else self.invalidsScore
                 for probas in self.model.predictMols(parsed_mols, use_probas=True, **self.kwargs)])

    def getKey(self):
        return f"QSPRpred_{self.model.name}"