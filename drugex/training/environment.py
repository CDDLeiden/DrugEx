"""
environment

Created by: Martin Sicho
On: 06.06.22, 16:51
"""
import numpy as np
import pandas as pd
from rdkit import Chem

from drugex.training.interfaces import Environment
from drugex.training.scorers.smiles import SmilesChecker
from drugex.utils import similarity_sort, nsgaii_sort
from drugex.utils.fingerprints import get_fingerprint


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

    @staticmethod
    def calc_fps(mols, fp_type='ECFP6'):
        fps = []
        for i, mol in enumerate(mols):
            try:
                fps.append(get_fingerprint(mol, fp_type))
            except:
                fps.append(None)
        return fps

    def getRewards(self, smiles, scheme='WS', frags=None):
        """
        Calculate the single value as the reward for each molecule used for reinforcement learning
        Args:
            smiles (List):  a list of SMILES-based molecules
            scheme (str): the label of different rewarding schemes, including
                'WS': weighted sum
                'PR': Pareto ranking with Tanimoto distance,
                'CD': Pareto ranking with crowding distance and

        Returns:
            rewards (np.ndarray): n-d array in which the element is the reward for each molecule, and
                n is the number of array which equals to the size of smiles.
        """
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        preds = self.getScores(smiles, frags=frags)
        valid = preds.VALID.values
        desire = preds.DESIRE.sum()
        undesire = len(preds) - desire
        preds = preds[self.getScorerKeys()].values

        if scheme == 'PR':
            fps = self.calc_fps(mols)
            rewards = np.zeros((len(smiles), 1))
            ranks = similarity_sort(preds, fps, is_gpu=True)
            score = (np.arange(undesire) / undesire / 2).tolist() + (np.arange(desire) / desire / 2 + 0.5).tolist()
            rewards[ranks, 0] = score
        elif scheme == 'CD':
            rewards = np.zeros((len(smiles), 1))
            ranks = nsgaii_sort(preds, is_gpu=True)
            rewards[ranks, 0] = np.arange(len(preds)) / len(preds)
        else:
            weight = ((preds < self.thresholds).mean(axis=0, keepdims=True) + 0.01) / \
                     ((preds >= self.thresholds).mean(axis=0, keepdims=True) + 0.01)
            weight = weight / weight.sum()
            rewards = preds.dot(weight.T)
        rewards[valid == 0] = 0
        return rewards
