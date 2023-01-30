"""
rewards

Created by: Martin Sicho
On: 26.06.22, 18:07
"""
import numpy as np
from drugex.logs import logger
from drugex.training.interfaces import RankingStrategy, RewardScheme
from drugex.utils.fingerprints import get_fingerprint
from rdkit import Chem, DataStructs


class NSGAIIRanking(RankingStrategy):

    def __call__(self, smiles, scores):
        """
        Crowding distance algorithm to rank the solutions in the same pareto frontier.

        Paper: Deb, Kalyanmoy, et al. "A fast and elitist multiobjective genetic algorithm: NSGA-II." IEEE transactions on
        evolutionary computation 6.2 (2002): 182-197.

        Parameters
        ----------
        smiles : list
            List of SMILES sequence to be ranked
            TODO : why here smiles are needed?
        scores : np.ndarray
            matrix of scores for the multiple objectives

        Returns
        -------
        rank : np.array
            Indices of the SMILES sequences ranked with the NSGA-II crowding distance method
        """        

        fronts = self.getParetoFronts(scores)

        rank = []
        # sort all fronts by crowding distance
        for t, front in enumerate(fronts):
            distance = np.zeros(len(front))
            for i in range(scores.shape[1]):
                # sort front small to large for value objective i
                cpu_tensor = scores[front.cpu(), i]
                order = cpu_tensor.argsort()
                front = front[order]
                # set distance value smallest and largest value objective i to large value
                distance[order[0]] = 10 ** 4
                distance[order[-1]] = 10 ** 4
                # get all values of objective i in current front
                m_values = [scores[j, i] for j in front]
                # scale for crowding distance by difference between max and min of objective i in front
                scale = max(m_values) - min(m_values)
                if scale == 0:
                    scale = 1
                # calculate crowding distance
                for j in range(1, len(front) - 1):
                    distance[order[j]] += (scores[front[j + 1], i] - scores[front[j - 1], i]) / scale
            # replace front by front sorted according to crowding distance
            fronts[t] = front[np.argsort(distance)]
            rank.extend(fronts[t].tolist())
        return rank


class SimilarityRanking(RankingStrategy):

    """
    Revised crowding distance algorithm to rank the solutions in the same pareto frontier with Tanimoto-distance.
    """

    @staticmethod
    def calc_fps(mols, fp_type='ECFP6'):
        """
        Calculate fingerprints for a list of molecules.

        Parameters
        ----------
        mols : list
            List of RDKit molecules
        fp_type : str
            Type of fingerprint to use

        Returns
        -------
        fps : list
            List of RDKit fingerprints
        """   
        fps = []
        for i, mol in enumerate(mols):
            try:
                fps.append(get_fingerprint(mol, fp_type))
            except BaseException:
                fps.append(None)
        return fps

    def __call__(self, smiles, scores, func='min'):
        """
        Revised crowding distance algorithm to rank the solutions in the same fronter with Tanimoto-distance.

        Parameters
        ----------
        smiles : list
            List of SMILES sequence to be ranked
        scores : np.ndarray
            matrix of scores for the multiple objectives
        func : str
            'min' takes minimium tanimoto distance, 'avg' takes average tanimoto distance in the front

        Returns
        -------
        rank : np.array
            Indices of the SMILES sequences ranked with the NSGA-II crowding distance method
        """

        func = np.min if func == 'min' else np.mean
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        fps = self.calc_fps(mols)

        fronts = self.getParetoFronts(scores)

        rank = []
        for i, front in enumerate(fronts):
            front_fps = [fps[f] for f in front]
            if len(front) > 2 and None not in front_fps:
                dist = np.zeros(len(front))
                # find the min/average tanimoto distance for each fingerprint to all other fingerprints in the front
                dist = np.array(
                    [func(1 - np.array(DataStructs.BulkTanimotoSimilarity(fp, list(np.delete(front_fps, idx)))))
                     for idx, fp in enumerate(front_fps)])
                fronts[i] = front[dist.argsort()]
            elif None in front_fps:
                logger.warning("Invalid molecule in front. Front not ranked.")
            rank.extend(fronts[i].tolist())
        return rank


class ParetoTanimotoDistance(RewardScheme):
    """
    Reward scheme that uses the Tanimoto distance ranking strategy to rank the solutions in the same Pareto frontier.
    """

    def __init__(self, ranking=SimilarityRanking()):
        super().__init__(ranking)
        """
        Parameters
        ----------
        ranking : RankingStrategy
            Ranking strategy to use
        """

    def __call__(self, smiles, scores, desire, undesire, thresholds):
        """
        Reward scheme that uses the Tanimoto distance ranking strategy to rank the solutions in the same Pareto frontier.

        Parameters
        ----------
        smiles : list
            List of SMILES sequence to be ranked
        scores : np.ndarray
            matrix of scores for the multiple objectives
        desire : int
            Number of desired molecules
        undesire : int
            Number of undesired molecules
        thresholds : np.ndarray
            Thresholds for the multiple objectives
        """

        if not self.ranking:
            raise self.RewardException(f"{self.__class__.__name__} reward scheme requires a ranking strategy.")

        ranks = self.ranking(smiles, scores)
        rewards = np.zeros((len(smiles), 1))
        score = (np.arange(undesire) / undesire / 2).tolist() + (np.arange(desire) / desire / 2 + 0.5).tolist()
        rewards[ranks, 0] = score
        return rewards


class ParetoCrowdingDistance(RewardScheme):
    """
    Reward scheme that uses the NSGA-II crowding distance ranking strategy to rank the solutions in the same Pareto frontier.
    """

    def __init__(self, ranking=NSGAIIRanking()):
        super().__init__(ranking)
        """
        Parameters
        ----------
        ranking : RankingStrategy
            Ranking strategy to use
        """

    def __call__(self, smiles, scores, desire, undesire, thresholds):
        """
        Reward scheme that uses the NSGA-II crowding distance ranking strategy to rank the solutions in the same Pareto frontier.
        
        Parameters
        ----------
        smiles : list  
            List of SMILES sequence to be ranked
        scores : np.ndarray
            matrix of scores for the multiple objectives
        desire : int
            array of desired molecules
        undesire : int
            array of undesired molecules
        thresholds : np.ndarray
            Thresholds for the multiple objectives

        Returns
        -------
        rewards : np.ndarray
            Array of rewards for the SMILES sequences
        """
        ranks = self.ranking(smiles, scores)
        rewards = np.zeros((len(smiles), 1))
        rewards[ranks, 0] = np.arange(len(scores)) / len(scores)
        return rewards


class WeightedSum(RewardScheme):
    """
    Reward scheme that uses the weighted sum ranking strategy to rank the solutions.
    """

    def __call__(self, smiles, scores, desire, undesire, thresholds):
        """
        Reward scheme that uses the weighted sum ranking strategy to rank the solutions.

        Parameters
        ----------
        smiles : list
            List of SMILES sequence to be ranked
        scores : np.ndarray
            matrix of scores for the multiple objectives
        desire : int
            array of desired molecules
        undesire : int
            array of undesired molecules
        thresholds : np.ndarray
            Thresholds for the multiple objectives

        Returns
        -------
        rewards : np.ndarray
            Array of rewards for the SMILES sequences
        """
        
        weight = ((scores < thresholds).mean(axis=0, keepdims=True) + 0.01) / \
            ((scores >= thresholds).mean(axis=0, keepdims=True) + 0.01)
        weight = weight / weight.sum()
        return scores.dot(weight.T)
