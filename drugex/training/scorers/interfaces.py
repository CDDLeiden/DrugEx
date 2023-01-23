from abc import ABC, abstractmethod


class ScoreModifier(ABC):
    """
    Defines a function to modify a score value.

    """
    @abstractmethod
    def __call__(self, x):
        """
        Apply the modifier on x.
        Args:
            x: float or np.array to modify
        Returns:
            float or np.array (depending on the type of x) after application of the distance function.
        """

class Scorer(ABC):
    """
    Used by the `Environment` to calculate customized scores.

    """

    def __init__(self, modifier=None):
        self.modifier = modifier

    @abstractmethod
    def getScores(self, mols, frags=None):
        """
        Returns scores for the input molecules.

        Args:
            mols: molecules to score
            frags: input fragments

        Returns:
            scores (list): `list` of scores for "mols"
        """

        pass

    def __call__(self, mols, frags=None):
        """
        Actual call method. Modifies the scores before returning them.

        Args:
            mols: molecules to score
            frags: input fragments

        Returns:
            scores (DataFrame): a data frame with columns name 'VALID' and 'DESIRE' indicating the validity of the SMILES and the degree of desirability
        """

        return self.getModifiedScores(self.getScores(mols, frags))

    def getModifiedScores(self, scores):
        """
        Modify the scores with the given `ScoreModifier`.

        Args:
            scores:

        Returns:

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
