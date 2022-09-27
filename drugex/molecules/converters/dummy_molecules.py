"""
dummy molecules

Created by: Sohvi Luukkonen
On: 27.09.22, 13:40
"""

from rdkit import Chem

from drugex.logs import logger

class dummyMolsFromFragments():

    def addCBrToFrag(self, frag):

        repl = Chem.MolFromSmiles('CBr')
        patt = Chem.MolFromSmarts('[#1;$([#1])]')   

        fragH = Chem.AddHs(Chem.MolFromSmiles(frag))   
        molH = Chem.ReplaceSubstructs(fragH, patt, repl, replaceAll=False)
        mol = Chem.RemoveHs(molH[0])
        return Chem.MolToSmiles(mol)

    def createMolsFromSingleFragments(self, frags):  

        pairs = []
        for frag in frags:
            try:
                smiles = self.addCBrToFrag(frag)
                pairs.append((frag, smiles))
            except:
                logger.debug(f"Skipped: couldn't build a molecule from {frag}.")

        return pairs

    def createMolsFromMultipleFragments(self, frags_list):

        pairs=[]
        for frags in frags_list:
            try:
                frags_rdkit = [Chem.MolFromSmiles(f) for f in frags.split('.') ]
                mol = frags_rdkit[0]
                for i in range(1,len(frags_rdkit)):
                    n = mol.GetNumAtoms()
                    comb = Chem.EditableMol( Chem.CombineMols(mol, frags_rdkit[i]))
                    comb.AddBond(n-1, n, order=Chem.rdchem.BondType.SINGLE)
                    mol = comb.GetMol()
                pairs.append( (frags, Chem.MolToSmiles(mol)))
            except:
                print(frags)
        
        return pairs
                    

    def __call__(self, frags):

        """ 
        Create molecules from list of by adding CBr to single fragments and bridge multiple fragment
        
        Args:
            frags : SMILES of fragments
        Return:
            a list of `tuple`s of format  (fragment, smiles), fragment is the same as the input in "fragments"
        """

        single_frags = [f for f in frags if '.' not in f]
        multiple_frags = [f for f in frags if '.' in f]

        pairs = []
        if single_frags:
            pairs += self.createMolsFromSingleFragments(single_frags)
        if multiple_frags:
            pairs += self.createMolsFromMultipleFragments(multiple_frags)

        return pairs