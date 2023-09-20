import numpy as np
import torch

def get_Pareto_fronts(scores):
    """Identify the Pareto fronts from a given set of scores.
    
    Parameters
    ----------
    scores : numpy.ndarray
        An (n_points, n_scores) array of scores.
            
    Returns
    -------
    list of numpy.ndarray
        A list containing the indices of points belonging to each Pareto front.
    """
    
    # Initialize
    population_size = scores.shape[0]
    population_ids = np.arange(population_size)
    all_fronts = []

    # Identify Pareto fronts
    while population_size > 0:
        # Identify the current Pareto front
        pareto_front = np.ones(population_size, dtype=bool)
        for i in range(population_size):
            for j in range(population_size):
                # Strictly j better than i in all scores (i dominated by j) 
                # -> i not in Pareto front
                if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                    pareto_front[i] = 0
                    break

        # Add the current Pareto front to the list of all fronts
        current_front_ids = population_ids[pareto_front]
        all_fronts.append(current_front_ids)

        # Remove the current Pareto front from consideration in future iterations
        scores = scores[~pareto_front]
        population_ids = population_ids[~pareto_front]
        population_size = scores.shape[0]

    return all_fronts