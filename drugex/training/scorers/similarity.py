"""
similarity

Created by: Sohvi Luukkonen
On: 07.10.22, 15:05
"""
import tqdm
import networkx

import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem.Fraggle import FraggleSim
from rdkit.Chem import Descriptors as  rdFMCS

from drugex.utils.fingerprints import get_fingerprint
from drugex.training.interfaces import Scorer

class TverskyFingerprintSimilarity(Scorer):

    """ 
    Scoring function for similarity to a reference molecule. Tversky similarity between fingerprints.
    If both alpha and beta are set to 1, reduces to Tanimoto similarity. 
    """

    def __init__(self, smiles : str, fp_type : str, alpha : float = 1., beta : float = 1., modifier=None):
        """
        Args:
            smiles (str)        : reference compound
            fp_type (str)       : fingerprint type
            alpha (float), opt  : weight of the features of the reference compound
            beta (float), opt   : weight of the features of the compound to be scored
        """
        super().__init__(modifier)
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.fp_type = fp_type
        self.fp = get_fingerprint(self.mol, fp_type=fp_type)
        self.alpha = alpha
        self.beta = beta

    def getScores(self, mols, frags=None):
        scores = np.zeros(len(mols))
        for i, mol in enumerate(tqdm.tqdm(mols)):
            try:
                fp = get_fingerprint(mol, fp_type=self.fp_type)
                scores[i] = DataStructs.TverskySimilarity(self.fp, fp, self.alpha, self.beta)
            except: continue
        return scores

    def getKey(self):
        return f"Fingerprint similarity (fp_type={self.fp_type}, Tversky weights={self.alpha},{self.beta}, smiles={self.smiles})"

class TverskyGraphSimilarity(Scorer):
    
    """ 
    Scoring function for similarity to a reference molecule. Tversky similarity between graphs.
    If both alpha and beta are set to 1, reduces to Tanimoto similarity. 
    """
    
    def __init__(self, smiles : str, alpha : float = 1., beta : str = 1., modifier=None):
        """
        Args:
            smiles (str)        : reference compound
            alpha (float), opt  : weight of the features of the reference compound
            beta (float), opt   : weight of the features of the compound to be scored
        """
        super().__init__(modifier)
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.alpha = alpha
        self.beta = beta

    def getScores(self, mols, frags=None):
        scores = np.zeros(len(mols))
        for i, mol in enumerate(tqdm.tqdm(mols)):
            try:
                mcs = rdFMCS.FindMCS(mols)
                nmcs = mcs.numAtoms
                nref = self.mol.GetNumAtoms() - nmcs
                nmol = mol.GetNumAtoms() - nmcs
                
                scores[i] = nmcs / (nmcs + self.alpha * nref + self.beta * nmol)
            except: continue
        return scores
    
    def getKey(self):
        return f"Graph similarity (Tversky weights={self.alpha},{self.beta}, smiles={self.smiles})"

class FraggleSimilarity(Scorer):
    
    """ 
    Scoring function for similarity to a reference molecule. Fraggle similarity from python source 
    for an implementation of the fraggle similarity algorithm developed at GSK and described in this RDKit 
    UGMpresentation: https://github.com/rdkit/UGM_2013/blob/master/Presentations/Hussain.Fraggle.pdf
    """
    
    def __init__(self, smiles : str, trevsky_th : float = 0.8, modifier=None):
        """
        Args:
            smiles (str)            : reference compound
            trevsky_th (float), opt : Trevsky threshold used by Fraggle
        """
        super().__init__(modifier)
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.th = trevsky_th

    def getScores(self, mols, frags=None):
        scores = np.zeros(len(mols))
        for i, mol in enumerate(tqdm.tqdm(mols)):
            try: 
                scores[i] = FraggleSim.GetFraggleSimilarity(self.mol, mol)
            except: continue
        return scores
    
    def getKey(self):
        return f"Fraggle similarity (Tversky threshold={self.th}, smiles={self.smiles})"

class GraphEditInverseDistance(Scorer):

    """
    Scoring function for similarity to a reference molecule. 
    Inverse of Graph Edit distance between two molecular graphs.
    
    WARNING: Extremly slow! 
    TODO : See, if possible to speed up
    
    """
    def __init__(self, smiles, modifier=None):
        super().__init__(modifier)
        self.mol = Chem.MolFromSmiles(smiles)
        self.graph = self.get_graph(self.mol)

    def get_graph(self, mol):
        Chem.Kekulize(mol)
        atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        am = Chem.GetAdjacencyMatrix(mol,useBO=True)
        for i,atom in enumerate(atoms):
            am[i,i] = atom
        G = networkx.from_numpy_matrix(am)
        return G

    def getScores(self, mols, frags=None):
        scores = np.zeros(len(mols))
        for i, mol in enumerate(tqdm.tqdm(mols)):
            try:
                graph = self.get_graph(mol)
                print(i)
                for v in networkx.optimize_graph_edit_distance(self.graph, graph, edge_match=lambda a,b: a['weight']==b['weight']):
                    dist = v
                scores[i] = 1 / np.sqrt(dist)
            except: continue
        return scores

    def getKey(self):
        return f"Graph similarity (Tversky weights={self.alpha},{self.beta}, smiles={self.smiles})"