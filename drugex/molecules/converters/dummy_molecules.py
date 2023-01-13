"""
dummy molecules

Created by: Sohvi Luukkonen
On: 27.09.22, 13:40
"""

from rdkit import Chem

from drugex.logs import logger

class dummyMolsFromFragments():

    def addCBrToFragments(self, frag):

        repl = Chem.MolFromSmiles('CC')
        patt = Chem.MolFromSmarts('[#1;$([#1])]')   

        try:
            fragH = Chem.AddHs(Chem.MolFromSmiles(frag))   
            molH = Chem.ReplaceSubstructs(fragH, patt, repl, replaceAll=False)
            mol = Chem.RemoveHs(molH[0])
            return Chem.MolToSmiles(mol)
        except:
            logger.debug(f"Skipped: couldn't build a molecule from the {frag} fragment.")
            return None

    def bridgeFragments(self, frags):

        try:
            frags_rdkit = [Chem.MolFromSmiles(f) for f in frags.split('.') ]
            mol = frags_rdkit[0] # set 1st fragment as base of molecule
            
            for i in range(1,len(frags_rdkit)): # iterate over other fragments to be combined with the base molecule
                frag = frags_rdkit[i] 
                natoms_frag = frag.GetNumAtoms()
                natoms_mol = mol.GetNumAtoms()

                # iterate over bridging positions until creation of valid molecule
                for j in range(natoms_mol): 
                    for k in range(natoms_frag):
                        comb = Chem.EditableMol( Chem.CombineMols(mol, frag ))
                        mpos = natoms_mol - j -1
                        fpos = natoms_mol + k
                        comb.AddBond(mpos, fpos, order=Chem.rdchem.BondType.SINGLE)
                        smiles = Chem.MolToSmiles(comb.GetMol())
                        if Chem.MolFromSmiles(smiles) is not None: break   
            return smiles
            
        except:
            logger.warning(f"Skipped: couldn't build a molecule from the {frags} fragements.")
            return None
            

    def __call__(self, frag):

        """ 
        Create molecule from by adding CC to single fragments or bridge multiple fragment
        
        Args:
            frags : SMILES of fragment(s)
        Return:
            a list of `tuple`s of format  (fragment, smiles), fragment is the same as the input in "fragments"
        """

        try:
            if '.' in frag: # multiple leaf fragments
                smiles = self.bridgeFragments(frag)
            else: # single leaf fragment
                smiles = self.addCBrToFragments(frag)
            return [(frag, smiles)]
        except:
            logger.warning(f"Skipped: couldn't build a molecule from {frag}.")
            return None     