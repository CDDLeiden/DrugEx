import pandas as pd

from drugex.manipulations.splitting import TrainTestSplitter


class FragmentPairsSplitter(TrainTestSplitter):

    def __init__(self, ratio=0.2, max_test_samples=None, train_collector=None, test_collector=None):
        super().__init__(train_collector, test_collector)
        self.ratio = ratio
        self.maxTestSamples = max_test_samples

    def __call__(self, pairs):
        df = pd.DataFrame(pairs, columns=['Frags', 'Smiles'])
        frags = set(df.Frags)
        if len(frags) > int(1e5):
            print('WARNING: to speed up the training, the test set size was capped at 10,000 fragments instead of the default 10% of original data, which is: {}!'.format(len(frags)//(100 * self.ratio)))
            test_in = df.Frags.drop_duplicates().sample(int(self.maxTestSamples))
        else:
            test_in = df.Frags.drop_duplicates().sample(len(frags) // 10)
        test = df[df.Frags.isin(test_in)]
        train = df[~df.Frags.isin(test_in)]

        if self.trainCollect:
            self.trainCollect(train)
        if self.testCollect:
            self.testCollect(test)

        return test, train