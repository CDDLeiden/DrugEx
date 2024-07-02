import numpy as np
from drugex.logs import logger
from drugex.training.scorers.interfaces import Scorer
from qsprpred.tasks import ModelTasks

class QSPRPredScorer(Scorer):

    def __init__(self, model, invalids_score=0.0, modifier=None, **kwargs):
        super(QSPRPredScorer, self).__init__(modifier)
        self.model = model
        self.invalidsScore = invalids_score
        self.kwargs = kwargs

    def getScores(self, mols, frags=None):
        if len(mols) == 0:
            logger.warning("No molecules to score. Returning empty list...")
            return []

        valid_mols = [mol for mol in mols if mol is not None]

        if self.model.task == ModelTasks.REGRESSION:
            scores = self.model.predictMols(valid_mols, **self.kwargs)
        else:
            # FIXME: currently we only assume that the model is a binary classifier
            # with the positive class being the last one in the list of probabilities
            scores = self.model.predictMols(
                valid_mols,
                use_probas=True,
                **self.kwargs
            )[-1][:, -1]
            
        scores = scores.tolist()
        
        # Replace missing values with invalidsScore
        full_scores = np.array([
            scores.pop(0) if mol is not None and scores[0] is not None else self.invalidsScore
            for mol in mols
        ])

        return full_scores

    def getKey(self):
        return f"QSPRpred_{self.model.name}"