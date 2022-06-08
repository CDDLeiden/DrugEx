"""
splitters

Created by: Martin Sicho
On: 31.05.22, 12:25
"""
import math

from sklearn.model_selection import train_test_split

from drugex.datasets.interfaces import DataSplitter
from drugex.logs import logger


class RandomTrainTestSplitter(DataSplitter):

    def __init__(self, test_size, max_test_size=1e4, shuffle=True):
        self.testSize = test_size
        self.maxSize = max_test_size
        self.shuffle = shuffle

    def __call__(self, data):
        test_size = min(int(math.ceil(len(data) * self.testSize)), int(self.maxSize))
        if len(data) * self.testSize > int(self.maxSize):
            logger.warning(f'To speed up the training, the test set is reduced to a random sample of {self.maxSize} from the original test!')
        # data = np.asarray(data)
        return train_test_split(data, test_size=test_size, shuffle=self.shuffle)