import shutil
from unittest import TestCase
import os

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

from drugex.environment.neural_network import STFullyConnected
from drugex.environment.data import QSARDataset
from drugex.environment.models import QSARsklearn, QSARDNN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from xgboost import XGBRegressor, XGBClassifier

class TestData(TestCase):

    def test_data(self):
        df = pd.read_csv(f'{os.path.dirname(__file__)}/test_files/data/test_data.tsv', sep='\t')
        dataset = QSARDataset(input_df=df, target="P29274")
        self.assertIsInstance(dataset, QSARDataset)

        dataset.splitDataset()
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
        dataset.splitDataset()
        self.assertEqual(dataset.X_ind.shape[0], 3) # test size of 3
        self.assertEqual(dataset.X.shape[0], 5) # 8 - 3 is 5

        # with timesplit on 2015
        dataset = QSARDataset(input_df=df, target="P29274", timesplit=2015, test_size=3)
        dataset.splitDataset()
        self.assertEqual(dataset.X_ind.shape[0], 2) # two sample year > 2015
        self.assertEqual(dataset.X.shape[0], 6) # 8 - 2 is 6

        # with timesplit on 2015
        dataset = QSARDataset(input_df=df, target="P29274", keep_low_quality=True)
        dataset.splitDataset()
        self.assertEqual(dataset.X_ind.shape[0], 1) # test_size 0.1, should be 1 test sample
        self.assertEqual(dataset.X.shape[0], 9) # no datapoints removed, 10-1=9

        # with timesplit on 2015
        dataset = QSARDataset(input_df=df, target="P29274", keep_low_quality=True)
        dataset.splitDataset()
        self.assertEqual(dataset.X_ind.shape[0], 1) # test_size 0.1, should be 1 test sample
        self.assertEqual(dataset.X.shape[0], 9) # no datapoints removed, 10-1=9

        # with classification
        dataset = QSARDataset(input_df=df, target="P29274", reg=False, th=7)
        dataset.splitDataset()
        self.assertTrue(np.min(np.concatenate((dataset.y, dataset.y_ind))) == 0)
        self.assertTrue(np.max(np.concatenate((dataset.y, dataset.y_ind))) == 1)
        self.assertEqual(np.sum(np.concatenate((dataset.y, dataset.y_ind)) < 1), 3) # only 3 value below threshold of 7

class PathMixIn:
    datapath = f'{os.path.dirname(__file__)}/test_files/data'
    envspath = f'{os.path.dirname(__file__)}/test_files/envs'

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls.envspath):
            os.mkdir(cls.envspath)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.envspath)

class TestClassifiers(PathMixIn, TestCase):

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        os.remove(f'{cls.datapath}/testmodel.log')
        os.remove(f'{cls.datapath}/testmodel.pkg')

    def prep_testdata(self, reg=True):

        # prepare test dataset
        df = pd.read_csv(f'{self.datapath}/test_data_large.tsv', sep='\t')
        data = QSARDataset(input_df=df, target="P29274", reg=reg)
        data.splitDataset()
        data.X, data.X_ind = data.dataStandardization(data.X, data.X_ind)

        # prepare data for torch DNN
        y = data.y.reshape(-1,1)
        y_ind = data.y_ind.reshape(-1,1)
        trainloader = DataLoader(TensorDataset(torch.Tensor(data.X), torch.Tensor(y)), batch_size=100)
        testloader = DataLoader(TensorDataset(torch.Tensor(data.X_ind), torch.Tensor(y_ind)), batch_size=100)

        return data.X.shape[1], trainloader, testloader

    def test_STFullyConnected(self):
        # prepare test dataset
        no_features, trainloader, testloader = self.prep_testdata(reg=True)

        # fit model with default settings
        model = STFullyConnected(n_dim = no_features)
        model.fit(trainloader, testloader, out=f'{self.datapath}/testmodel')

        # fit model with non-default epochs and learning rate
        model = STFullyConnected(n_dim = no_features, n_epochs = 50, lr = 0.5)
        model.fit(trainloader, testloader, out=f'{self.datapath}/testmodel')

        # fit model with non-default settings for rate
        model = STFullyConnected(n_dim = no_features, neurons_h1=2000, neurons_hx=500, extra_layer=True)
        model.fit(trainloader, testloader, out=f'{self.datapath}/testmodel')

        # prepare classification test dataset
        no_features, trainloader, testloader = self.prep_testdata(reg=False)

        # fit model with regression is false
        model = STFullyConnected(n_dim = no_features, is_reg=False)
        model.fit(trainloader, testloader, out=f'{self.datapath}/testmodel')


class TestModels(PathMixIn, TestCase):

    def prep_testdata(self, reg=True):
        
        # prepare test dataset
        df = pd.read_csv(f'{os.path.dirname(__file__)}/test_files/data/test_data_large.tsv', sep='\t')
        data = QSARDataset(input_df=df, target="P29274", reg=reg)
        data.splitDataset()
        data.X, data.X_ind = data.dataStandardization(data.X, data.X_ind)
        
        return data

    def QSARsklearn_models_test(self, alg, alg_name, reg):
        data = self.prep_testdata(reg=reg)
        themodel = QSARsklearn(base_dir = f'{os.path.dirname(__file__)}/test_files/',
                               data=data, alg = alg, alg_name=alg_name)
        themodel.fit()
        themodel.evaluate()
        fname = f'{os.path.dirname(__file__)}/test_files/search_space_test.json'
        grid_params = QSARsklearn.loadParamsGrid(fname, "bayes", alg_name)
        search_space_bs = grid_params[grid_params[:,0] == alg_name,1][0]
        themodel.bayesOptimization(search_space_bs=search_space_bs, n_trials=1, save_m=False)
        grid_params = QSARsklearn.loadParamsGrid(fname, "grid", alg_name)
        search_space_gs = grid_params[grid_params[:,0] == alg_name,1][0]
        themodel.gridSearch(search_space_gs=search_space_gs, save_m=False)

    def testRF(self):
        alg_name = "RF"
        #test regression
        alg = RandomForestRegressor()
        self.QSARsklearn_models_test(alg, alg_name, reg=True)

        #test classifier
        alg = RandomForestClassifier()
        self.QSARsklearn_models_test(alg, alg_name, reg=False)

    def testKNN(self):
        alg_name = "KNN"
        #test regression
        alg = KNeighborsRegressor()
        self.QSARsklearn_models_test(alg, alg_name, reg=True)

        #test classifier
        alg = KNeighborsClassifier()
        self.QSARsklearn_models_test(alg, alg_name, reg=False)

    def testXGB(self):
        alg_name = "XGB"
        #test regression
        alg = XGBRegressor(objective='reg:squarederror')
        self.QSARsklearn_models_test(alg, alg_name, reg=True)

        #test classifier
        alg = XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')
        self.QSARsklearn_models_test(alg, alg_name, reg=False)

    def testSVM(self):
        alg_name = "SVM"
        #test regression
        alg = SVR()
        self.QSARsklearn_models_test(alg, alg_name, reg=True)

        #test classifier
        alg = SVC(probability=True)
        self.QSARsklearn_models_test(alg, alg_name, reg=False)

    def testPLS(self):
        alg_name = "PLS"
        #test regression
        alg = PLSRegression()
        self.QSARsklearn_models_test(alg, alg_name, reg=True)

    def testNB(self):
        alg_name = "NB"
        #test classfier
        alg = GaussianNB()
        self.QSARsklearn_models_test(alg, alg_name, reg=False)

    def test_QSARDNN(self):
        data = self.prep_testdata(reg=True)
        themodel = QSARDNN(base_dir = f'{os.path.dirname(__file__)}/test_files/', data=data, gpus=[3,2])
        themodel.fit()
        themodel.evaluate()
        fname = f'{os.path.dirname(__file__)}/test_files/search_space_test.json'
        grid_params = QSARDNN.loadParamsGrid(fname, "grid", "DNN")
        search_space_gs = grid_params[grid_params[:,0] == "DNN",1][0]
        themodel.gridSearch(search_space_gs=search_space_gs, save_m=False)