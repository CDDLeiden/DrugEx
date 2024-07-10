import numpy as np
from drugex.logs import logger
from drugex.training.scorers.interfaces import Scorer
from qsprpred.tasks import TargetProperty, TargetTasks

class QSPRPredScorer(Scorer):

    def __init__(
        self,
        model,
        use_probas=True,
        multi_task=None,
        multi_class=None,
        invalids_score=0.0,
        modifier=None,
        **kwargs):
        """ Scorer that uses a QSPRpred predictor to score molecules.
            Can return the probabilities or the predictions of the model.
            Multi-task models can be used, with the option to select specific tasks.

        Args:
            model (QSPRpredModel): QSPRpred predictor model.
            use_probas (bool, optional): Whether to use the probabilities instead of the predictions. Defaults to True.
            multi_task (list[str], optional): If the model is a multitask model, a list of tasks to use. Defaults to None (use all tasks).
            if multi_task is None, all tasks will be used.
            multi_class (list[int], optional): Which classes to use for multi-class models.
                If use_probas, the different classes will be returned as separate tasks, with their own
                key (task name with suffix "_{class number}". If single-class, the
                probabilities of the positive class will be returned. Defaults to None.
            invalids_score (float | list[float], optional): Score to return for invalid molecules. Defaults to 0.0.
                invalids_score can be a list of scores if multi-task, one for each task. If a single score is given,
                it will be broadcasted to all tasks.
            modifier (callable, optional): Function to modify the scores. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the model's predictMols method.
        """
        super(QSPRPredScorer, self).__init__(modifier)
        self.model = model
        self.use_probas = use_probas
        self.multi_task = multi_task
        if multi_task is not None:
            assert all(task in TargetProperty.getNames(model.targetProperties) for task in multi_task), \
            f"Tasks {multi_task} not found in model tasks {model.targetProperties}"
        self.multi_class = multi_class
        self.invalidsScore = invalids_score
        if isinstance(invalids_score, list):
            assert len(invalids_score) == self.nTasks, \
                "Invalids score list must have the same length as the number of tasks"
            assert all(isinstance(score, float) for score in invalids_score), \
                "Invalids score list must contain only floats"
        else:
            assert isinstance(invalids_score, float), "Invalids score must be a float"
        self.kwargs = kwargs

    def getScores(self, mols, frags=None):
        if len(mols) == 0:
            logger.warning("No molecules to score. Returning empty list...")
            return []

        # Get valid molecules
        mols_array = np.array(mols, dtype=object)
        valid_mask = mols_array != None
        valid_mols = mols_array[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_mols) == 0:
            logger.warning("No valid molecules to score. Returning all invalidsScore...")
            return np.full((len(mols), self.nTasks), self.invalidsScore)

        # Get the predictions
        if self.model.task.isRegression() or not self.use_probas:
            # If regression or not using probabilities, the scores are 2D array
            scores = self.model.predictMols(valid_mols, **self.kwargs)
            if self.model.isMultiTask and self.multi_task is not None:
                # take the column of the predictions where task is in multi_task
                target_props = TargetProperty.getNames(self.model.targetProperties)
                column_idx = [target_props.index(task) for task in self.multi_task]
                scores = scores[:, column_idx]
        else:
            # If classification and using probabilities, the scores are a list of 2D arrays
            scores = self.model.predictMols(
                valid_mols,
                use_probas=self.use_probas,
                **self.kwargs
            )
            for i, scores_per_task in enumerate(scores):
                if scores_per_task.shape[1] == 2:
                    # Take the probabilities of the positive class if binary
                    scores[i] = scores_per_task[:, 1].reshape(-1, 1)
                elif self.multi_class is not None:
                    # Take the probabilities of the selected classes if multi-class
                    scores[i] = scores_per_task[:, self.multi_class]
            
            if self.model.isMultiTask and self.multi_task is not None:
                # take the list items where task is in multi_task
                target_props = TargetProperty.getNames(self.model.targetProperties)
                if self.multi_task is not None:
                    scores = [scores[target_props.index(task)] for task in self.multi_task]
            
            # Concatenate the scores list into a single 2D array
            if isinstance(scores, list):
                scores = np.concatenate(scores, axis=1)

        # Place the valid scores into their corresponding positions in the array with invalid scores
        # check if any None values in scores, if so, replace with invalidsScore
        if isinstance(self.invalidsScore, list):
            full_scores = np.tile(self.invalidsScore, (len(mols), 1))
        else:
            full_scores = np.full((len(mols), self.nTasks), self.invalidsScore)
            
        # Identify rows containing NaN values
        nan_mask = np.isnan(scores.astype(float)).any(axis=1)

        # Remove rows with NaN values from scores and valid_indices
        if np.any(nan_mask):
            logger.warning("Some scores are NaN. Dropping these scores...")
            scores = scores[~nan_mask]
            valid_indices = valid_indices[~nan_mask]
        
        full_scores[valid_indices, :] = scores
        
        # return 1D array if only one task
        if full_scores.shape[1] == 1:
            return full_scores.flatten()

        return full_scores

    def getKey(self):
        base_key = f"QSPRpred_{self.model.name}"
        keys = []
        for target_prop in self.model.targetProperties:
            # add the task name to the key if a multi-task model
            if self.model.isMultiTask:
                if ((self.multi_task is not None and target_prop.name in self.multi_task)
                    or self.multi_task is None):
                    task_key = f"{base_key}_{target_prop.name}"
                else:
                    continue
            else:
                task_key = f"{base_key}"
                
            # if multiclass probabilities, include the classes as separate tasks
            if target_prop.task == TargetTasks.MULTICLASS and self.use_probas:
                idx = self.multi_class if self.multi_class is not None else range(target_prop.nClasses)
                for i in idx:
                    keys.append(f"{task_key}_{i}")
            else:
                keys.append(task_key)

        if len(keys) == 1:
            return keys[0]
        return keys
    
    @property
    def nTasks(self):
        tasks = self.getKey()
        return len(tasks) if isinstance(tasks, list) else 1
