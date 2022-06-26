import numpy as np
import torch


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