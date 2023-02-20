from functools import partial
from typing import List

import numpy as np

from drugex.training.scorers.interfaces import ScoreModifier


class Chained(ScoreModifier):
    """
    Calls several modifiers one after the other, for instance:
        score = modifier3(modifier2(modifier1(raw_score)))
    """

    def __init__(self, modifiers: List[ScoreModifier]) -> None:
        """
        Initializes the modifier with a list of modifiers to be called in order.

        Parameters
        ----------
        modifiers : list of ScoreModifier
            A list of modifiers to be called in order.
        """
        self.modifiers = modifiers

    def __call__(self, x):
        """
        Calls the modifiers in order.

        Parameters
        ----------
        x : float
            The input score.

        Returns
        -------
        float
            The modified score.
        """
        score = x
        for modifier in self.modifiers:
            score = modifier(score)
        return score


class Linear(ScoreModifier):
    """
    Score modifier that multiplies the score by a scalar (default: 1, i.e. do nothing).
    """

    def __init__(self, slope=1.0):
        """
        Initializes the Linear modifier with a given slope.

        Parameters
        ----------
        slope : float
            The slope of the linear function.
        """
        self.slope = slope

    def __call__(self, x):
        """
        Linear modifier: multiplies the score by the slope.

        Parameters
        ----------
        x : float
            The input score.

        Returns
        -------
        float
            The modified score.
        """
        return self.slope * x


class Squared(ScoreModifier):
    """
    Score modifier that has a maximum at a given target value, and decreases
    quadratically with increasing distance from the target value.
    """

    def __init__(self, target_value: float, coefficient=1.0) -> None:
        """
        Initializes the Squared modifier with a target value and a coefficient.

        Parameters
        ----------
        target_value : float
            The target value.
        coefficient : float
            The coefficient of the quadratic function.
        """
        self.target_value = target_value
        self.coefficient = coefficient

    def __call__(self, x):
        """
        Squared modifier:  1 - coefficient * (target_value - x)^2

        Parameters
        ----------
        x : float
            The input score.

        Returns
        -------
        float
            The modified score.
        """
        return 1.0 - self.coefficient * np.square(self.target_value - x)


class AbsoluteScore(ScoreModifier):
    """
    Score modifier that has a maximum at a given target value, and decreases
    linearly with increasing distance from the target value.
    """

    def __init__(self, target_value: float) -> None:
        """
        Initializes the AbsoluteScore modifier with a target value.

        Parameters
        ----------
        target_value : float
            The target value.
        """
        self.target_value = target_value

    def __call__(self, x):
        """
        AbsoluteScore modifier:  1 - |target_value - x|

        Parameters
        ----------
        x : float
            The input score.
        
        Returns
        -------
        float
            The modified score.
        """
        return 1. - np.abs(self.target_value - x)


class Gaussian(ScoreModifier):
    """
    Score modifier that reproduces a Gaussian bell shape.
    """

    def __init__(self, mu: float, sigma: float) -> None:
        """ 
        Initializes the Gaussian modifier with a mean and a standard deviation.

        Parameters
        ----------
        mu : float
            The mean of the Gaussian.
        sigma : float
            The standard deviation of the Gaussian.
        """
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        """
        Gaussian modifier:  exp(-0.5 * (x - mu)^2 / sigma^2)

        Parameters
        ----------
        x : float
            The input score.
        
        Returns
        -------
        float
            The modified score.
        """
        return np.exp(-0.5 * np.power((x - self.mu) / self.sigma, 2.))


class MinMaxGaussian(ScoreModifier):
    """
    Score modifier that reproduces a half Gaussian bell shape.
    For minimize==True, the function is 1.0 for x <= mu and decreases to zero for x > mu.
    For minimize==False, the function is 1.0 for x >= mu and decreases to zero for x < mu.
    """

    def __init__(self, mu: float, sigma: float, minimize=False) -> None:
        """
        Initializes the MinMaxGaussian modifier with a mean, a standard deviation, and a flag
        indicating whether the function should be minimized or maximized.

        Parameters
        ----------
        mu : float
            The mean of the Gaussian.
        sigma : float
            The standard deviation of the Gaussian.
        minimize : bool
            Whether the function should be minimized or maximized.
        """
        self.mu = mu
        self.sigma = sigma
        self.minimize = minimize
        self._full_gaussian = Gaussian(mu=mu, sigma=sigma)

    def __call__(self, x):
        """
        MinMaxGaussian modifier:  exp(-0.5 * (x - mu)^2 / sigma^2)
        For minimize==True, the function is 1.0 for x <= mu and decreases to zero for x > mu.
        For minimize==False, the function is 1.0 for x >= mu and decreases to zero for x < mu.

        Parameters
        ----------
        x : float
            The input score.
        
        Returns
        -------
        float
            The modified score.
        """
        if self.minimize:
            mod_x = np.maximum(x, self.mu)
        else:
            mod_x = np.minimum(x, self.mu)
        return self._full_gaussian(mod_x)


MinGaussian = partial(MinMaxGaussian, minimize=True)
MaxGaussian = partial(MinMaxGaussian, minimize=False)


class ClippedScore(ScoreModifier):
    r"""
    Clips a score between specified low and high scores, and does a linear interpolation in between.
    The function looks like this:
       upper_x < lower_x                 lower_x < upper_x
    __________                                   ____________
              \                                 /
               \                               /
                \__________          _________/
    This class works as follows:
    First the input is mapped onto a linear interpolation between both specified points.
    Then the generated values are clipped between low and high scores.
    """

    def __init__(self, upper_x: float, lower_x=0.0, high_score=1.0, low_score=0.0) -> None:
        """
        Initializes the ClippedScore modifier with a target value and a coefficient.

        Parameters
        ----------
        upper_x : float
            x-value from which (or until which if smaller than lower_x) the score is maximal
        lower_x : float
            x-value until which (or from which if larger than upper_x) the score is minimal
        high_score : float
            maximal score to clip to
        low_score : float
            minimal score to clip to
        """
        assert low_score < high_score

        self.upper_x = upper_x
        self.lower_x = lower_x
        self.high_score = high_score
        self.low_score = low_score

        self.slope = (high_score - low_score) / (upper_x - lower_x)
        self.intercept = high_score - self.slope * upper_x

    def __call__(self, x):
        """
        ClippedScore modifier: max(min(slope * x + intercept, high_score), low_score)

        Parameters
        ----------
        x : float
            The input score.

        Returns
        -------
        float
            The modified score.
        """
        y = self.slope * x + self.intercept
        return np.clip(y, self.low_score, self.high_score)


class SmoothClippedScore(ScoreModifier):
    """
    Smooth variant of ClippedScore.
    Implemented as a logistic function that has the same steepness as ClippedScore in the
    center of the logistic function.
    """

    def __init__(self, upper_x: float, lower_x=0.0, high_score=1.0, low_score=0.0) -> None:
        """
        Initializes the SmoothClippedScore modifier with a target value and a coefficient.

        Parameters
        ----------
        upper_x : float
            x-value from which (or until which if smaller than lower_x) the score is maximal
        lower_x : float
            x-value until which (or from which if larger than upper_x) the score is minimal
        high_score : float
            maximal score to clip to
        low_score : float
            minimal score to clip to
        """
        assert low_score < high_score

        self.upper_x = upper_x
        self.lower_x = lower_x
        self.high_score = high_score
        self.low_score = low_score

        # Slope of a standard logistic function in the middle is 0.25 -> rescale k accordingly
        self.k = 4.0 / (upper_x - lower_x)
        self.middle_x = (upper_x + lower_x) / 2
        self.L = high_score - low_score

    def __call__(self, x):
        """
        SmoothClippedScore modifier: low_score + L / (1 + exp(-k * (x - middle_x)))

        Parameters
        ----------
        x : float
            The input score.
        """
        return self.low_score + self.L / (1 + np.exp(-self.k * (x - self.middle_x)))


class ThresholdedLinear(ScoreModifier):
    """
    Returns a value of min(input, threshold)/threshold.
    """

    def __init__(self, threshold: float) -> None:
        """
        Initializes the ThresholdedLinear modifier.

        Parameters
        ----------
        threshold : float
            threshold value
        """
        self.threshold = threshold

    def __call__(self, x):
        """
        ThresholdedLinear modifier: min(x, threshold)/threshold

        Parameters
        ----------
        x : float
            The input score.

        Returns
        -------
        float
            The modified score.
        """
        return np.minimum(x, self.threshold) / self.threshold

class SmoothHump(ScoreModifier):
    """
    Score modifier that reproduces a smooth bump function.
    The function is 1.0 for x between (lower_x, upper_x) and decreases to zero with a half Gaussian for x < lower_x and x > upper_x.
    """

    def __init__(self, lower_x : float, upper_x : float, sigma: float ) -> None:
        """
        Initializes the SmoothHump modifier.

        Parameters
        ----------
        lower_x : float
            x-value until which the score is minimal
        upper_x : float
            x-value from which the score is maximal
        sigma : float
            sigma of the Gaussian
        """
        self.sigma = sigma
        self.lower_x = lower_x
        self.upper_x = upper_x
        self._maximize_gaussian = MinMaxGaussian(mu=lower_x, sigma=sigma, minimize=False)
        self._minimize_gaussian = MinMaxGaussian(mu=upper_x, sigma=sigma, minimize=True)

    def __call__(self, x):
        """
        SmoothHump modifier: min(max(gaussian(x, lower_x, sigma), gaussian(x, upper_x, sigma)), 1.0)

        Parameters
        ----------
        x : float
            The input score.

        Returns
        -------
        float
            The modified score.
        """
        y_maximize = self._maximize_gaussian(x)
        y_minimize = self._minimize_gaussian(x)
        return np.minimum(y_maximize, y_minimize) 