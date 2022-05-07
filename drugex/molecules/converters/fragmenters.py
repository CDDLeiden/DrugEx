"""
fragmenters

Created by: Martin Sicho
On: 06.05.22, 12:19
"""
import re
from itertools import combinations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Recap, BRICS

from drugex.molecules.converters.interfaces import ConversionException
from drugex.molecules.converters.standardizers import CleanSMILES


class Fragmenter(CleanSMILES):

    def __init__(self, n_frags, n_combs, method='recap', deep_clean=True):
        super().__init__(deep_clean)
        self.nFrags = n_frags
        self.nCombs = n_combs
        self.method = method
        if self.method not in ('recap', 'brics'):
            raise ConversionException(f"Unknown fragmentation method: {self.method}")


    def __call__(self, smiles):
        ret_frags = []
        smiles = super().__call__(smiles)
        mol = Chem.MolFromSmiles(smiles)
        # break SMILES up into leaf fragments
        if self.method == 'recap':
            frags = np.array(sorted(Recap.RecapDecompose(mol).GetLeaves().keys()))
        else:
            frags = BRICS.BRICSDecompose(mol)
            frags = np.array(sorted({re.sub(r'\[\d+\*\]', '*', f) for f in frags}))
        if len(frags) == 1:
            return None
        # replace connection tokens with [H]
        du, hy = Chem.MolFromSmiles('*'), Chem.MolFromSmiles('[H]')
        subs = np.array([Chem.MolFromSmiles(f) for f in frags])
        subs = np.array([Chem.RemoveHs(Chem.ReplaceSubstructs(f, du, hy, replaceAll=True)[0]) for f in subs])
        subs = np.array([m for m in subs if m.GetNumAtoms() > 1])
        # remove fragments that contain other fragments (or are contained in other fragments?)
        match = np.array([[m.HasSubstructMatch(f) for f in subs] for m in subs])
        frags = subs[match.sum(axis=0) == 1]
        # sort the fragments and only keep n_frag largest ones
        frags = sorted(frags, key=lambda x:-x.GetNumAtoms())[:self.nFrags]
        frags = [Chem.MolToSmiles(Chem.RemoveHs(f)) for f in frags]

        max_comb = min(self.nCombs, len(frags))
        for ix in range(1, max_comb+1):
            # combine leaf fragments into larger fragments
            combs = combinations(frags, ix)
            for comb in combs:
                comb_frags = '.'.join(comb)
                #remove pair of fragment combinations if longer than original SMILES
                if len(comb_frags) > len(smiles): continue
                # check if substructure is in original molecule
                if mol.HasSubstructMatch(Chem.MolFromSmarts(comb_frags)):
                    ret_frags.append((comb_frags, smiles))

        return ret_frags

