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
    def checkSmiles(smiles, frags=None, no_multifrag_smiles=True):
        """
        This method is used to check the validity of the SMILES strings and to check if they contain given fragments.

        Parameters
        ----------
        smiles : list of str
            List of SMILES strings to check.
        frags : list of str, optional
            List of SMILES strings of fragments to check for.
        no_multifrag_smiles : bool, optional
            If True, SMILES strings that contain more than one fragment will be marked as invalid.

        Returns
        -------
        scores : pd.DataFrame
            Dataframe with the validity and accuracy of the SMILES strings.
        """
        
        scores = pd.DataFrame()
        
        #valids = np.zeros(shape)
        if no_multifrag_smiles:
            # Check if SMILES is not fragmented
            smiles = [smi if smi.count('.') == 0 else None for smi in smiles]
        
        for j, smile in enumerate(smiles):
            # 1. Check if SMILES can be parsed by rdkit
            try:
                mol = Chem.MolFromSmiles(smile)
                if not smile:
                    mol = None
                scores.loc[j, 'Valid'] = 0 if mol is None else 1
            except:
                scores.loc[j, 'Valid'] = 0

                
            if frags is not None:
                # 2. Check if SMILES contain given fragments
                try:
                    subs = frags[j].split('.')
                    subs = [Chem.MolFromSmiles(sub) for sub in subs]
                    scores.loc[j, 'Accurate'] = 1 if np.all([mol.HasSubstructMatch(sub) for sub in subs]) else 0
                except:
                    scores.loc[j, 'Accurate'] = 0
            # else : 
            #     #scores['Accurate'] = np.nan
    
        return scores
