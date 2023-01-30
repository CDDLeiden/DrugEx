from abc import ABC, abstractmethod


class ScoreModifier(ABC):
    """
    Defines a function to modify a score value.
    """
    @abstractmethod
    def __call__(self, x):
        """
        Apply the modifier on x.

        Parameters
        ----------
        x : float or np.array
            The value to modify

        Returns
        -------
        float or np.array
            The modified value
        """
        pass

class Scorer(ABC):
    """
    Used by the `Environment` to calculate customized scores.
    """

    def __init__(self, modifier=None):
        """
        Initialize the scorer.

        Parameters
        ----------
        modifier : ScoreModifier, optional
            A `ScoreModifier` object to modify the scores, by default None.
        """
        self.modifier = modifier

    @abstractmethod
    def getScores(self, mols, frags=None):
        """
        Returns the raw scores for the input molecules.

        Parameters
        ----------
        mols : list of rdkit molecules
            The molecules to be scored.
        frags : list of rdkit molecules, optional
            The fragments used to generate the molecules, by default None.

        Returns
        -------
        scores : np.array
            The scores for the molecules.
        """
        pass

    def __call__(self, mols, frags=None):
        """
        Actual call method. Modifies the scores before returning them.

        Parameters
        ----------
        mols : list of rdkit molecules
            The molecules to be scored.
        frags : list of rdkit molecules, optional
            The fragments used to generate the molecules, by default None.

        Returns
        -------
        scores : np.array
            The scores for the molecules.
        """

        return self.getModifiedScores(self.getScores(mols, frags))

    def getModifiedScores(self, scores):
        """
        Modify the scores with the given `ScoreModifier`.

        Parameters
        ----------
        scores : np.array
            The scores to modify.

        Returns
        -------
        np.array
            The modified scores.
        """

        if self.modifier:
            return self.modifier(scores)
        else:
            return scores

    @abstractmethod
    def getKey(self):
        pass

    def setModifier(self, modifier):
        self.modifier = modifier

    def getModifier(self):
        return self.modifier
