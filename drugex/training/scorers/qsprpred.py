import numpy as np
from drugex.logs import logger
from drugex.training.scorers.interfaces import Scorer
from qsprpred.tasks import TargetProperty, TargetTasks
from rdkit import Chem

class QSPRPredScorer(Scorer):

    def __init__(
        self,
        model,
        use_probas=True,
        multi_task=None,
        multi_class=None,
        app_domain=False,
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
            app_domain (bool | str, optional): Whether to also return the applicability of molecules according to an attached applicability domain.
                This will be returned as a seperate task with the key suffix "_app_domain". Defaults to False.
                To mark a non-applicable molecule as invalid and return the invalids_score, set app_domain to 'invalid'.
            invalids_score (float | list[float], optional): Score to return for invalid molecules. Defaults to 0.0.
                invalids_score can be a list of scores if multi-task, one for each task. If a single score is given,
                it will be broadcasted to all tasks. Note. if a modifier is specified it will also be applied
                to invalid molecules.
            modifier (callable, optional): Function to modify the scores. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the model's predictMols method.
        """
        super(QSPRPredScorer, self).__init__(modifier)
        self.model = model
        self.use_probas = use_probas
        self.multi_task = multi_task
        self.app_domain = app_domain
        assert self.app_domain in [True, False, 'invalid'], "app_domain must be a boolean or 'invalid'"
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

        valid_mols, valid_indices = self._get_valid_molecules(mols)
        
        # Create an array to store the scores filled with the invalidsScore
        if isinstance(self.invalidsScore, list):
            scores = np.tile(self.invalidsScore, (len(mols), 1))
        else:
            scores = np.full((len(mols), self.nTasks), self.invalidsScore)

        if len(valid_mols) == 0:
            logger.warning("No valid molecules to score. Returning all invalidsScore...")
            if self.nTasks == 1:
                return scores.flatten()
            return scores

        valid_scores = self._get_predictions(valid_mols)
            
        # Remove rows with NaN values, e.g. if the model returns NaN scores
        # due to failed standardization
        nan_mask = np.isnan(valid_scores.astype(float)).any(axis=1)
        if np.any(nan_mask):
            logger.warning("Some scores are NaN. Dropping these scores...")
            valid_scores = valid_scores[~nan_mask]
            valid_indices = valid_indices[~nan_mask]
        
        # place the valid scores in the correct indices of the full scores array
        scores[valid_indices, :] = valid_scores
        
        # return 1D array if only one task
        if scores.shape[1] == 1:
            return scores.flatten()

        return scores
    
    def _get_valid_molecules(self, mols):
        """Get the valid molecules and their indices in the input list."""
        # check if any is a str, if so, convert to rdkit mol
        if any(isinstance(mol, str) for mol in mols):
            mols = [Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol for mol in mols]
        
        mols_array = np.array(mols, dtype=object)
        valid_mask = mols_array != None
        valid_mols = mols_array[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        return valid_mols, valid_indices
    
    def _get_predictions(self, mols):
        """Get the predictions or probabilities of the QSPRmodel as a 2D array."""
        include_app = self.app_domain in [True, 'invalid']
        scores = self.model.predictMols(
            mols,
            use_probas=self.use_probas,
            use_applicability_domain = self.app_domain if include_app else False,
            **self.kwargs
        )
        if include_app:
            # separate the predictions and applicability of the molecules
            scores, app = scores
        
        if self.model.task.isRegression() or not self.use_probas:
            # If regression or non-probabilistic classification, the scores are a 2D array
            scores = self.handle_regression_task(scores)
        else:
            # If classification and using probabilities, the scores are a list of 2D arrays
            scores = self.handle_classification_task(scores)
        
        # add app domain as separate task or set molecules outside app to invalid
        if include_app:
            scores = self.handle_app_domain(scores, app)
        
        return scores
    
    def handle_regression_task(self, scores):
        """Handle the regression or non-probabilistic classification scores."""
        if self.model.isMultiTask and self.multi_task is not None:
            # take the column of the predictions where task is in multi_task
            target_props = TargetProperty.getNames(self.model.targetProperties)
            column_idx = [target_props.index(task) for task in self.multi_task]
            scores = scores[:, column_idx]
        return scores
    
    def handle_classification_task(self, scores):
        """Handle probabilistic classification scores."""
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
        return scores
    
    def handle_app_domain(self, scores, app):
        """Handle the applicability domain scores."""
        if self.app_domain == 'invalid':
            # flatten app to 1D array and convert to boolean
            app = np.array(app).flatten().astype(bool)
            # Set the scores of invalid molecules to the invalidsScore
            scores[~app] = self.invalidsScore
        else:
            # Add the applicability domain scores as a separate task
            scores = np.concatenate([scores, app.reshape(-1, 1)], axis=1)
        return scores

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
        
        if self.app_domain == True:
            keys.append(f"{base_key}_app_domain")

        if len(keys) == 1:
            return keys[0]
        return keys
    
    @property
    def nTasks(self):
        tasks = self.getKey()
        return len(tasks) if isinstance(tasks, list) else 1
