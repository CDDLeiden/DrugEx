"""
scorers

Created by: Martin Sicho
On: 03.06.22, 13:28
"""
import numpy as np
import pandas as pd
from rdkit import Chem

class SmilesChecker:

    @staticmethod
    def checkSmiles(smiles, frags=None):
        shape = (len(smiles), 1) if frags is None else (len(smiles), 2)
        valids = np.zeros(shape)
        for j, smile in enumerate(smiles):
            # 1. Check if SMILES can be parsed by rdkit
            try:
                mol = Chem.MolFromSmiles(smile)
                if not smile:
                    mol = None
                valids[j, 0] = 0 if mol is None else 1
            except:
                valids[j, 0] = 0
            if frags is not None:
                # 2. Check if SMILES contain given fragments
                try:
                    subs = frags[j].split('.')
                    subs = [Chem.MolFromSmiles(sub) for sub in subs]
                    valids[j, 1] = np.all([mol.HasSubstructMatch(sub) for sub in subs])
                except:
                    valids[j, 1] = 0
        return pd.DataFrame(valids, columns=['VALID', 'DESIRE']) if frags is not None else valids
