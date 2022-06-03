"""
scorers

Created by: Martin Sicho
On: 03.06.22, 13:28
"""
import pandas as pd

from drugex.training.interfaces import ModelEvaluator
from drugex.utils.objective import Env

class SmilesChecker(ModelEvaluator):

    def __call__(self, smiles, frags=None):
        scores =  Env.check_smiles(smiles, frags)
        return pd.DataFrame(scores, columns=['VALID', 'DESIRE'])