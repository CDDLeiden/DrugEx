"""
rewards

Created by: Martin Sicho
On: 26.06.22, 18:07
"""
import numpy as np

from drugex import DEFAULT_DEVICE
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
            List of SMILES sequence to be ranked (not used in the calculation -> just a requirement of the interface because some ranking strategies need it)
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


class MinMeanSimilarityRanking(RankingStrategy):
    """
    Revised crowding distance algorithm to rank the solutions in the same pareto frontier with Tanimoto-distance.
    """

    def __init__(self, device=DEFAULT_DEVICE, func='mean'):
        """
        Args:
            device: pytorch device
            func: 'min' or 'mean' - how to calculate the similarity between the molecules in the same front
        """

        super().__init__(device=device)
        self.func = np.min if func == 'min' else np.mean

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

    def getFPs(self, smiles):
        """
        Calculate fingerprints for a list of molecules.

        Args:
            smiles: smiles to calculate fingerprints for

        Returns:
            list of RDKit fingerprints
        """

        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        return self.calc_fps(mols)


    def __call__(self, smiles, scores):
        """
        Revised crowding distance algorithm to rank the solutions in the same frontier with Tanimoto-distance.
        Molecules with the lowest Tanimoto-distance to the other molecules in the same front are ranked first.

        Args:
            smiles (list): List of SMILES sequence to be ranked
            scores (np.ndarray): matrix of scores for the multiple objectives

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
        fps = self.getFPs(smiles)
        fronts = self.getParetoFronts(scores)

        rank = []
        for i, front in enumerate(fronts):
            front_fps = [fps[f] for f in front]
            if len(front) > 2 and None not in front_fps:
                # find the min/average tanimoto distance for each molecule to all other molecules in the front
                dist = np.array(
                    [self.func(1 - np.array(DataStructs.BulkTanimotoSimilarity(fp, list(np.delete(front_fps, idx)))))
                     for idx, fp in enumerate(front_fps)])
                fronts[i] = front[dist.argsort()] # sort front (molecule with the lowest min/average distance to the others is first)
            elif None in front_fps:
                logger.warning("Invalid molecule in front. Front not ranked.")
            rank.extend(fronts[i].tolist())
        return rank # sorted list of all molecules in all fronts (molecules with the lowest min/average distance to the others are ranked first)

class MutualSimilaritySortRanking(MinMeanSimilarityRanking):
    """
    Alternative implementation of the `MinMeanSimilarityRanking` ranking strategy that tries
    to more closely emulate the crowding distance with fingerprint similarities. Adapted
    from the original code by @XuhanLiu (https://github.com/XuhanLiu/DrugEx/blob/cd384f4a8ed4982776e92293f77afd4ea78644f9/utils/nsgaii.py#L92).
    """

    def __call__(self, smiles, scores):
        fps = self.getFPs(smiles)
        fronts = self.getParetoFronts(scores)

        rank = []
        for i, front in enumerate(fronts):
            fp = [fps[f] for f in front]
            if len(front) > 2 and None not in fp:
                dist = np.zeros(len(front)) # array of cumulative crowded similarity distance scores for each molecule in the front -> the higher the score, the more diverse the 'crowd' around the molecule -> the better the reward in the end
                for j in range(len(front)): # for each molecule in the front
                    tanimoto = 1 - np.array(DataStructs.BulkTanimotoSimilarity(fp[j], fp)) # calculate tanimoto distance to all other molecules in the front
                    order = tanimoto.argsort() # get the order from lowest to highest tanimoto distance (the molecule j itself is at index 0)
                    dist[order[0]] += 0 # the molecule itself is scored 0 because we want a diverse set and this is the baseline lowest score
                    dist[order[-1]] += 10 ** 4 # ensure the farthest molecule always gets the highest score
                    for k in range(1, len(order) - 1): # for all other molecules in the front
                        # save the sum of the differences in distances between molecule j and the other molecules in the front
                        # in other words: measure how the other molecules 'crowd' together around molecule j
                        dist[order[k]] += tanimoto[order[k + 1]] - tanimoto[order[k - 1]]
                fronts[i] = front[dist.argsort()]
            rank.extend(fronts[i].tolist())
        return rank


class ParetoTanimotoDistance(RewardScheme):
    """
    Reward scheme that uses the Tanimoto distance ranking strategy to rank the solutions in the same Pareto frontier.
    """

    def __init__(self, ranking=MinMeanSimilarityRanking()):
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
