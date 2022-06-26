"""
environment

Created by: Martin Sicho
On: 06.06.22, 16:51
"""
import pandas as pd
from rdkit import Chem

from drugex.training.interfaces import Environment
from drugex.training.scorers.smiles import SmilesChecker


class DrugExEnvironment(Environment):

    def getScores(self, smiles, is_modified=True, frags=None):
        """
        Calculate the scores of all objectives for all of samples
        Args:
            mols (List): a list of molecules
            is_smiles (bool): if True, the type of element in mols should be SMILES sequence, otherwise
                it should be the Chem.Mol
            is_modified (bool): if True, the function of modifiers will work, otherwise
                the modifiers will ignore.

        Returns:
            preds (DataFrame): The scores of all objectives for all of samples which also includes validity
                and desirability for each SMILES.
        """
        preds = {}
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        for scorer in self.scorers:
            score = scorer(mols)
            preds[scorer.getKey()] = score
        preds = pd.DataFrame(preds)
        undesire = (preds < self.thresholds)  # ^ self.objs.on
        preds['DESIRE'] = (undesire.sum(axis=1) == 0).astype(int)
        preds['VALID'] = SmilesChecker.checkSmiles(smiles, frags=frags).all(axis=1).astype(int)

        preds[preds.VALID == 0] = 0
        return preds
