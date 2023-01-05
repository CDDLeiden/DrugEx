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
        Args:
            swarm (np.ndarray): m x n scoring matrix, where m is the number of samples
                and n is the number of objectives.

            is_gpu (bool): if True, the algorithm will be implemented by PyTorch and ran on GPUs, otherwise,
                it will be implemented by Numpy and ran on CPUs.

        Returns:
            rank (np.array): m-d vector as the index of well-ranked solutions.
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

    @staticmethod
    def calc_fps(mols, fp_type='ECFP6'):
        fps = []
        for i, mol in enumerate(mols):
            try:
                fps.append(get_fingerprint(mol, fp_type))
            except BaseException:
                fps.append(None)
        return fps

    def __call__(self, smiles, scores):
        """
        Revised crowding distance algorithm to rank the solutions in the same fronter with Tanimoto-distance.
        Args:
            swarm (np.ndarray): m x n scoring matrix, where m is the number of samples
                and n is the number of objectives.
            fps (np.ndarray): m-d vector as fingerprints for all the molecules

            is_gpu (bool): if True, the algorithm will be implemented by PyTorch and ran on GPUs, otherwise,
                it will be implemented by Numpy and ran on CPUs.

        Returns:
            rank (np.array): m-d vector as the index of well-ranked solutions.
        """

        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        fps = self.calc_fps(mols)

        fronts = self.getParetoFronts(scores)

        rank = []
        for i, front in enumerate(fronts):
            front_fps = [fps[f] for f in front]
            if len(front) > 2 and None not in front_fps:
                dist = np.zeros(len(front))
                # find the avarage tanimoto distance for each fingerprint to all other fingerprints in the front
                dist = np.array([
                    np.sum(1 - np.array(DataStructs.BulkTanimotoSimilarity(fp, front_fps))) / len(front)
                    for fp in front_fps])
                fronts[i] = front[dist.argsort()]
            elif None in front_fps:
                logger.warning("Invalid molecule in front. Front not ranked.")
            rank.extend(fronts[i].tolist())
        return rank


class ParetoSimilarity(RewardScheme):

    def __init__(self, ranking=SimilarityRanking()):
        super().__init__(ranking)

    def __call__(self, smiles, scores, valid, desire, undesire, thresholds):
        if not self.ranking:
            raise self.RewardException(f"{self.__class__.__name__} reward scheme requires a ranking strategy.")

        ranks = self.ranking(smiles, scores)
        rewards = np.zeros((len(smiles), 1))
        score = (np.arange(undesire) / undesire / 2).tolist() + (np.arange(desire) / desire / 2 + 0.5).tolist()
        rewards[ranks, 0] = score
        return rewards


class ParetoCrowdingDistance(RewardScheme):

    def __init__(self, ranking=NSGAIIRanking()):
        super().__init__(ranking)

    def __call__(self, smiles, scores, valid, desire, undesire, thresholds):
        ranks = self.ranking(smiles, scores)
        rewards = np.zeros((len(smiles), 1))
        rewards[ranks, 0] = np.arange(len(scores)) / len(scores)
        return rewards


class WeightedSum(RewardScheme):

    def __call__(self, smiles, scores, valid, desire, undesire, thresholds):
        weight = ((scores < thresholds).mean(axis=0, keepdims=True) + 0.01) / \
            ((scores >= thresholds).mean(axis=0, keepdims=True) + 0.01)
        weight = weight / weight.sum()
        return scores.dot(weight.T)
