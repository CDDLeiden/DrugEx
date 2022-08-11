from unittest import TestCase
import os
import pandas as pd
import numpy as np
from drugex.environment.classifier import STFullyConnected
from drugex.environment.data import QSARDataset
import torch
from torch.utils.data import DataLoader, TensorDataset

class TestData(TestCase):

    def test_data(self):
        df = pd.read_csv(f'{os.path.dirname(__file__)}/test_files/data/test_data.tsv', sep='\t')
        dataset = QSARDataset(input_df=df, target="P29274")
        self.assertIsInstance(dataset, QSARDataset)

        dataset.split_dataset()
        self.assertIsInstance(dataset.X, np.ndarray)
        self.assertIsInstance(dataset.X_ind, np.ndarray)
        self.assertIsInstance(dataset.y, np.ndarray)
        self.assertIsInstance(dataset.y_ind, np.ndarray)

        self.assertEqual(dataset.X.shape, (len(dataset.y), 19 + 2048)) # 19 (no. of physchem desc) +  2048 (fp bit len)
        self.assertEqual(dataset.X_ind.shape, (len(dataset.y_ind), 19 + 2048))

        # default case
        self.assertEqual(dataset.X.shape[0] + dataset.X_ind.shape[0], 8) # two of 10 datapoints removed, low quality
        self.assertEqual(dataset.X_ind.shape[0], 1) # test_size 0.1, should be 1 test sample
        self.assertEqual(dataset.X.shape[0], 7) # test_size 0.1, should be 1 test sample
        
        # regression is true
        self.assertEqual(np.min(np.concatenate((dataset.y, dataset.y_ind))), 6.460)
        self.assertEqual(np.max(np.concatenate((dataset.y, dataset.y_ind))), 8.960)

        # with test size is 3
        dataset = QSARDataset(input_df=df, target="P29274", test_size=3)
        dataset.split_dataset()
        self.assertEqual(dataset.X_ind.shape[0], 3) # test size of 3
        self.assertEqual(dataset.X.shape[0], 5) # 8 - 3 is 5

        # with timesplit on 2015
        dataset = QSARDataset(input_df=df, target="P29274", timesplit=2015, test_size=3)
        dataset.split_dataset()
        self.assertEqual(dataset.X_ind.shape[0], 2) # two sample year > 2015
        self.assertEqual(dataset.X.shape[0], 6) # 8 - 2 is 6

        # with timesplit on 2015
        dataset = QSARDataset(input_df=df, target="P29274", keep_low_quality=True)
        dataset.split_dataset()
        self.assertEqual(dataset.X_ind.shape[0], 1) # test_size 0.1, should be 1 test sample
        self.assertEqual(dataset.X.shape[0], 9) # no datapoints removed, 10-1=9

        # with timesplit on 2015
        dataset = QSARDataset(input_df=df, target="P29274", keep_low_quality=True)
        dataset.split_dataset()
        self.assertEqual(dataset.X_ind.shape[0], 1) # test_size 0.1, should be 1 test sample
        self.assertEqual(dataset.X.shape[0], 9) # no datapoints removed, 10-1=9

        # with classification
        dataset = QSARDataset(input_df=df, target="P29274", reg=False, th=7)
        dataset.split_dataset()
        self.assertTrue(np.min(np.concatenate((dataset.y, dataset.y_ind))) == 0)
        self.assertTrue(np.max(np.concatenate((dataset.y, dataset.y_ind))) == 1)
        self.assertEqual(np.sum(np.concatenate((dataset.y, dataset.y_ind)) < 1), 3) # only 3 value below threshold of 7

class TestClassifiers(TestCase):
    def test_STFullyConnected(self):
        df = pd.read_csv(f'{os.path.dirname(__file__)}/test_files/data/test_data_large.tsv', sep='\t')
        df = df.sample(500)
        data = QSARDataset(input_df=df, target="P29274")
        data.split_dataset()
        y = data.y.reshape(-1,1)
        y_ind = data.y_ind.reshape(-1,1)
        trainloader = DataLoader(TensorDataset(torch.Tensor(data.X), torch.Tensor(y)), batch_size=100)
        testloader = DataLoader(TensorDataset(torch.Tensor(data.X_ind), torch.Tensor(y_ind)), batch_size=100)
        model = STFullyConnected(n_dim = data.X.shape[1])
        model.fit(trainloader, testloader, out=f'{os.path.dirname(__file__)}/test_files/data/testmodel')

# class TestModels(TestCase):
#     def test(self):
#         pass