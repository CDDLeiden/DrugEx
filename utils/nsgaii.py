import numpy as np
from rdkit import DataStructs
import torch
import utils
from pymoo.factory import get_reference_directions
from pymoo.algorithms.moo.nsga3 import HyperplaneNormalization, associate_to_niches, calc_niche_count

def dominate(ind1: np.ndarray, ind2: np.ndarray):
    """
    Determine if soulution ind1 is dominated by ind2.
    Args:
        ind1 (np.ndarray): m-d vector represented the scores of a solution for all of objectives.
        ind2 (np.ndarray): m-d vector represented the scores of a solution for all of objectives.

    Returns:
        True if ind1 is dominated by ind2, otherwise False.
    """
    assert ind1.shape == ind2.shape
    all = np.all(ind1 <= ind2)
    any = np.any(ind1 < ind2)
    return all & any


def gpu_non_dominated_sort(swarm: torch.Tensor):
    """
    The GPU version of non-dominated sorting algorithms
    Args:
        swarm (np.ndarray): m x n scoring matrix, where m is the number of samples
            and n is the number of objectives.

    Returns:
        fronts (List): a list of Pareto fronts, in which the dominated solutions are on the top,
            and non-dominated solutions are on the bottom.
    """
    domina = (swarm.unsqueeze(1) <= swarm.unsqueeze(0)).all(-1)
    domina_any = (swarm.unsqueeze(1) < swarm.unsqueeze(0)).any(-1)
    domina = (domina & domina_any).half()

    fronts = []
    while (domina.diag() == 0).any():
        count = domina.sum(dim=0)
        front = torch.where(count == 0)[0]
        fronts.append(front)
        domina[front, :] = 0
        domina[front, front] = -1
    return fronts


# Function to carry out NSGA-II's fast non dominated sort
def cpu_non_dominated_sort(swarm: np.ndarray):
    """
    The CPU version of non-dominated sorting algorithms
    Args:
        swarm (np.ndarray): m x n scoring matrix, where m is the number of samples
            and n is the number of objectives.

    Returns:
        fronts (List): a list of Pareto fronts, in which the dominated solutions are on the top,
            and non-dominated solutions are on the bottom.
    """
    domina = [[] for _ in range(len(swarm))]
    front = []
    count = np.zeros(len(swarm), dtype=int)
    ranks = np.zeros(len(swarm), dtype=int)
    for p, ind1 in enumerate(swarm):
        for q in range(p + 1, len(swarm)):
            ind2 = swarm[q]
            if dominate(ind1, ind2):
                    domina[p].append(q)
                    count[q] += 1
            elif dominate(ind2, ind1):
                domina[q].append(p)
                count[p] += 1
        if count[p] == 0:
            ranks[p] = 0
            front.append(p)

    fronts = [np.sort(front)]
    i = 0
    while len(fronts[i]) > 0:
        temp = []
        for f in fronts[i]:
            for d in domina[f]:
                count[d] -= 1
                if count[d] == 0:
                    ranks[d] = i + 1
                    temp.append(d)
        i = i + 1
        fronts.append(np.sort(temp))
    del fronts[len(fronts) - 1]
    return fronts


def similarity_sort(swarm, fps, is_gpu=False):
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
    if is_gpu:
        swarm = torch.Tensor(swarm).to(utils.dev)
        fronts = gpu_non_dominated_sort(swarm)
    else:
        fronts = cpu_non_dominated_sort(swarm)
    rank = []
    for i, front in enumerate(fronts):
        fp = [fps[f] for f in front]
        if len(front) > 2 and None not in fp:
            dist = np.zeros(len(front))
            for j in range(len(front)):
                tanimoto = 1 - np.array(DataStructs.BulkTanimotoSimilarity(fp[j], fp))
                order = tanimoto.argsort()
                dist[order[0]] += 0
                dist[order[-1]] += 10 ** 4
                for k in range(1, len(order)-1):
                    dist[order[k]] += tanimoto[order[k+1]] - tanimoto[order[k-1]]
            fronts[i] = front[dist.argsort()]
        rank.extend(fronts[i].tolist())
    return rank


def nsgaii_sort(swarm, is_gpu=False):
    """
    Crowding distance algorithm to rank the solutions in the same fronter.
    Args:
        swarm (np.ndarray): m x n scoring matrix, where m is the number of samples
            and n is the number of objectives.

        is_gpu (bool): if True, the algorithm will be implemented by PyTorch and ran on GPUs, otherwise,
            it will be implemented by Numpy and ran on CPUs.

    Returns:
        rank (np.array): m-d vector as the index of well-ranked solutions.
    """
    if is_gpu:
        swarm = torch.Tensor(swarm).to(utils.dev)
        fronts = gpu_non_dominated_sort(swarm)
    else:
        fronts = cpu_non_dominated_sort(swarm)
    rank = []
    #sort all fronts by crowding distance
    for t, front in enumerate(fronts):
        distance = np.zeros(len(front))
        for i in range(swarm.shape[1]):
            #sort front small to large for value objective i
            order = swarm[front, i].argsort()
            front = front[order]
            #set distance value smallest and largest value objective i to large value
            distance[order[0]] = 10 ** 4
            distance[order[-1]] = 10 ** 4
            #get all values of objective i in current front
            m_values = [swarm[j, i] for j in front]
            #scale for crowding distance by difference between max and min of objective i in front
            scale = max(m_values) - min(m_values)
            if scale == 0: scale = 1
            #calculate crowding distance
            for j in range(1, len(front) - 1):
                distance[order[j]] += (swarm[front[j + 1], i] - swarm[front[j - 1], i]) / scale
        #replace front by front sorted according to crowding distance
        fronts[t] = front[np.argsort(distance)]
        rank.extend(fronts[t].tolist())
    return rank

def nsgaiii_sort(swarm, is_gpu=False, ref_dirs=None):
    """
    Crowding distance algorithm to rank the solutions in the same front for multi-objective optimization.
    Based on the implementation in pymoo (pymoo.org)
    Args:
        swarm (np.ndarray): m x n scoring matrix, where m is the number of samples
            and n is the number of objectives.

        is_gpu (bool): if True, the algorithem will be implemented by PyTorch and ran on GPUs, otherwise,
            it will be implemented by Numpy and ran on CPUs.

    Returns:
        rank (np.array): m-d vector as the index of well-ranked solutions.
    """
    #Get fronts by non_dominated sorting
    if is_gpu:
        swarm = torch.Tensor(swarm).to(utils.dev)
        fronts = gpu_non_dominated_sort(swarm)
    else:
        fronts = cpu_non_dominated_sort(swarm)
    
    non_dominated = fronts[0]

    if ref_dirs is None:
        ref_dirs = get_reference_directions("das-dennis", swarm.shape[1], n_partitions=12)

    # Find hyperplane through extreme points
    # update the hyperplane based boundary estimation
    hyp_norm = HyperplaneNormalization(ref_dirs.shape[1])
    hyp_norm.update(swarm, nds=non_dominated)
    ideal_point, nadir_point = hyp_norm.ideal_point, hyp_norm.nadir_point
    
    # associate individuals to niches
    niche_of_individuals, dist_to_niche, _ = \
        associate_to_niches(swarm, ref_dirs, ideal_point, nadir_point)

    #niche of individuals is for each individual the niche to which it has the shortest perpendicular distance
    
    rank = []
    for f, front in enumerate(fronts):
        
        # count the number of individuals per front until the current front
        if f==0:
            until_current_front = np.array([], dtype=int)
            niche_count = np.zeros(len(ref_dirs), dtype=int)
        else:
            until_current_front = np.concatenate(fronts[:f])
            #number of individuals in niche
            niche_count = calc_niche_count(len(ref_dirs), niche_of_individuals[until_current_front])

        S = niching(swarm[front], niche_count, niche_of_individuals[front],
                    dist_to_niche[front])

        rank.extend(front[np.argsort(S)].tolist())
    return rank


def niching(pop, niche_count, niche_of_individuals, dist_to_niche):
    '''
    Copy of the niching function from pymoo nsgaiii, but with adjusted for loop, so all indiviuals will be sorted.
    '''

    survivors = []

    # boolean array of elements that are considered for each iteration
    mask = np.full(len(pop), True)

    while len(survivors) < len(pop):
        # number of individuals to select in this iteration
        n_select = len(pop) - len(survivors)

        # all niches where new individuals can be assigned to and the corresponding niche count
        next_niches_list = np.unique(niche_of_individuals[mask])
        next_niche_count = niche_count[next_niches_list]

        # the minimum niche count
        min_niche_count = next_niche_count.min()

        # all niches with the minimum niche count (truncate randomly if more niches than remaining individuals)
        next_niches = next_niches_list[np.where(next_niche_count == min_niche_count)[0]]
        next_niches = next_niches[np.random.permutation(len(next_niches))[:n_select]]

        for next_niche in next_niches:

            # indices of individuals that are considered and assign to next_niche
            next_ind = np.where(np.logical_and(niche_of_individuals == next_niche, mask))[0]

            # shuffle to break random tie (equal perp. dist) or select randomly
            np.random.shuffle(next_ind)

            if niche_count[next_niche] == 0:
                next_ind = next_ind[np.argmin(dist_to_niche[next_ind])]  
            else:
                # already randomized through shuffling
                next_ind = next_ind[0]

            # add the selected individual to the survivors
            mask[next_ind] = False
            survivors.append(int(next_ind))

            # increase the corresponding niche count
            niche_count[next_niche] += 1

    return survivors

