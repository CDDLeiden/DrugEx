#!/usr/bin/env python
import os
import sys
import json
import torch
import joblib
import argparse

import numpy as np
import pandas as pd

from copy import deepcopy
from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor, XGBClassifier

#import models
import utils

def SVM(X, y, X_ind, y_ind, reg=False):
    """ Cross validation and Independent test for SVM classifion/regression model.
        Arguments:
            X (np.ndarray): m x n feature matrix for cross validation, where m is the number of samples
                and n is the number of features.
            y (np.ndarray): m-d label array for cross validation, where m is the number of samples and
                equals to row of X.
            X_ind (np.ndarray): m x n Feature matrix for independent set, where m is the number of samples
                and n is the number of features.
            y_ind (np.ndarray): m-d label array for independent set, where m is the number of samples and
                equals to row of X_ind, and l is the number of types.
            reg (bool): it True, the training is for regression, otherwise for classification.
         Returns:
            cvs (np.ndarray): m x l result matrix for cross validation, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.
            inds (np.ndarray): m x l result matrix for independent test, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.
    """
    if reg:
        folds = KFold(5).split(X)
        alg = SVR()
    else:
        folds = StratifiedKFold(5).split(X, y)
        alg = SVC(probability=True)
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    gs = GridSearchCV(deepcopy(alg), {'C': 2.0 ** np.array([-15, 15]), 'gamma': 2.0 ** np.array([-15, 15])}, n_jobs=10)
    gs.fit(X, y)
    params = gs.best_params_
    #print(params)
    for i, (trained, valided) in enumerate(folds):
        model = deepcopy(alg)
        model.C = params['C']
        model.gamma = params['gamma']
        if not reg:
            model.probability=True
        model.fit(X[trained], y[trained], sample_weight=[1 if v >= 4 else 0.1 for v in y[trained]])
        if reg:
            cvs[valided] = model.predict(X[valided])
            inds += model.predict(X_ind)
        else:
            cvs[valided] = model.predict_proba(X[valided])[:, 1]
            inds += model.predict_proba(X_ind)[:, 1]
    return cvs, inds / 5


def RF(X, y, X_ind, y_ind, reg=False):
    """ Cross validation and Independent test for RF classifion/regression model.
        Arguments:
            X (np.ndarray): m x n feature matrix for cross validation, where m is the number of samples
                and n is the number of features.
            y (np.ndarray): m-d label array for cross validation, where m is the number of samples and
                equals to row of X.
            X_ind (np.ndarray): m x n Feature matrix for independent set, where m is the number of samples
                and n is the number of features.
            y_ind (np.ndarray): m-d label array for independent set, where m is the number of samples and
                equals to row of X_ind, and l is the number of types.
            reg (bool): it True, the training is for regression, otherwise for classification.
         Returns:
            cvs (np.ndarray): m x l result matrix for cross validation, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.
            inds (np.ndarray): m x l result matrix for independent test, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.
    """
    if reg:
        folds = KFold(5).split(X)
        alg = RandomForestRegressor
    else:
        folds = StratifiedKFold(5).split(X, y)
        alg = RandomForestClassifier
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    for i, (trained, valided) in enumerate(folds):
        model = alg(n_estimators=1000, n_jobs=10)
        model.fit(X[trained], y[trained], sample_weight=[1 if v >= 4 else 0.1 for v in y[trained]])
        if reg:
            cvs[valided] = model.predict(X[valided])
            inds += model.predict(X_ind)
        else:
            cvs[valided] = model.predict_proba(X[valided])[:, 1]
            inds += model.predict_proba(X_ind)[:, 1]
    return cvs, inds / 5


def XGB(X, y, X_ind, y_ind, reg=False):
    """ Cross validation and Independent test for XGboost classifion/regression model.

    Arguments:
        X (np.ndarray): m x n feature matrix for cross validation, where m is the number of samples
            and n is the number of features.

        y (np.ndarray): m-d label array for cross validation, where m is the number of samples and
            equals to row of X.

        X_ind (np.ndarray): m x n Feature matrix for independent set, where m is the number of samples
            and n is the number of features.

        y_ind (np.ndarray): m-d label array for independent set, where m is the number of samples and
            equals to row of X_ind, and l is the number of types.

        reg (bool): it True, the training is for regression, otherwise for classification.


        Returns:
        cvs (np.ndarray): m x l result matrix for cross validation, where m is the number of samples and
            equals to row of X, and l is the number of types and equals to row of X.

        inds (np.ndarray): m x l result matrix for independent test, where m is the number of samples and
            equals to row of X, and l is the number of types and equals to row of X.

    """
    if reg:
        folds = KFold(5).split(X)
        alg = XGBRegressor
    else:
        folds = StratifiedKFold(5).split(X, y)
        alg = XGBClassifier(use_label_encoder=False)
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    for (trained, valided) in folds:
        if reg:
            model = alg(objective='reg:squarederror')#, use_label_encoder=False)
        else:
            model = alg
        model.fit(X[trained], y[trained], sample_weight=[1 if v >= 4 else 0.1 for v in y[trained]])
        if reg:
            cvs[valided] = model.predict(X[valided])
            inds += model.predict(X_ind)
        else:
            cvs[valided] = model.predict_proba(X[valided])[:, 1]
            inds += model.predict_proba(X_ind)[:, 1]
    return cvs, inds / 5


def KNN(X, y, X_ind, y_ind, reg=False):
    """ Cross validation and Independent test for KNN classifion/regression model.
        Arguments:
            X (np.ndarray): m x n feature matrix for cross validation, where m is the number of samples
                and n is the number of features.
            y (np.ndarray): m-d label array for cross validation, where m is the number of samples and
                equals to row of X.
            X_ind (np.ndarray): m x n Feature matrix for independent set, where m is the number of samples
                and n is the number of features.
            y_ind (np.ndarray): m-d label array for independent set, where m is the number of samples and
                equals to row of X_ind, and l is the number of types.
            reg (bool): it True, the training is for regression, otherwise for classification.
         Returns:
            cvs (np.ndarray): m x l result matrix for cross validation, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.
            inds (np.ndarray): m x l result matrix for independent test, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.
    """
    if reg:
        folds = KFold(5).split(X)
        alg = KNeighborsRegressor
    else:
        folds = StratifiedKFold(5).split(X, y)
        alg = KNeighborsClassifier
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    for i, (trained, valided) in enumerate(folds):
        model = alg(n_jobs=10)
        model.fit(X[trained], y[trained])
        if reg:
            cvs[valided] = model.predict(X[valided])
            inds += model.predict(X_ind)
        else:
            cvs[valided] = model.predict_proba(X[valided])[:, 1]
            inds += model.predict_proba(X_ind)[:, 1]
    return cvs, inds / 5


def NB(X, y, X_ind, y_ind):
    """ Cross validation and Independent test for Naive Bayes classifion model.
        Arguments:
            X (np.ndarray): m x n feature matrix for cross validation, where m is the number of samples
                and n is the number of features.
            y (np.ndarray): m-d label array for cross validation, where m is the number of samples and
                equals to row of X.
            X_ind (np.ndarray): m x n Feature matrix for independent set, where m is the number of samples
                and n is the number of features.
            y_ind (np.ndarray): m-d label array for independent set, where m is the number of samples and
                equals to row of X_ind, and l is the number of types.
         Returns:
            cvs (np.ndarray): m x l result matrix for cross validation, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.
            inds (np.ndarray): m x l result matrix for independent test, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.
    """
    folds = KFold(5).split(X)
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    for i, (trained, valided) in enumerate(folds):
        model = GaussianNB()
        model.fit(X[trained], y[trained], sample_weight=[1 if v >= 4 else 0.1 for v in y[trained]])
        cvs[valided] = model.predict_proba(X[valided])[:, 1]
        inds += model.predict_proba(X_ind)[:, 1]
    return cvs, inds / 5


def PLS(X, y, X_ind, y_ind):
    """ Cross validation and Independent test for PLS regression model.
        Arguments:
            X (np.ndarray): m x n feature matrix for cross validation, where m is the number of samples
                and n is the number of features.
            y (np.ndarray): m-d label array for cross validation, where m is the number of samples and
                equals to row of X.
            X_ind (np.ndarray): m x n Feature matrix for independent set, where m is the number of samples
                and n is the number of features.
            y_ind (np.ndarray): m-d label array for independent set, where m is the number of samples and
                equals to row of X_ind, and l is the number of types.
            reg (bool): it True, the training is for regression, otherwise for classification.
         Returns:
            cvs (np.ndarray): m x l result matrix for cross validation, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.
            inds (np.ndarray): m x l result matrix for independent test, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.
    """
    folds = KFold(5).split(X)
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    for i, (trained, valided) in enumerate(folds):
        model = PLSRegression()
        model.fit(X[trained], y[trained])
        cvs[valided] = model.predict(X[valided])[:, 0]
        inds += model.predict(X_ind)[:, 0]
    return cvs, inds / 5


def DNN(X, y, X_ind, y_ind, out, reg=False):
    
    """ 
    Cross validation and Independent test for DNN classifion/regression model.
    
    Arguments:
        X (np.ndarray): m x n feature matrix for cross validation, where m is the number of samples
            and n is the number of features.
        y (np.ndarray): m x l label matrix for cross validation, where m is the number of samples and
            equals to row of X, and l is the number of types.
        X_ind (np.ndarray): m x n Feature matrix for independent set, where m is the number of samples
            and n is the number of features.
        y_ind (np.ndarray): m-d label arrays for independent set, where m is the number of samples and
            equals to row of X_ind, and l is the number of types.
        reg (bool): it True, the training is for regression, otherwise for classification.
     Returns:
        cvs (np.ndarray): m x l result matrix for cross validation, where m is the number of samples and
            equals to row of X, and l is the number of types and equals to row of X.
        inds (np.ndarray): m x l result matrix for independent test, where m is the number of samples and
            equals to row of X, and l is the number of types and equals to row of X.
    """
    if y.shape[1] > 1 or reg:
        folds = KFold(5).split(X)
    else:
        folds = StratifiedKFold(5).split(X, y[:, 0])
    NET = models.STFullyConnected if y.shape[1] == 1 else models.MTFullyConnected
    indep_set = TensorDataset(torch.Tensor(X_ind), torch.Tensor(y_ind))
    indep_loader = DataLoader(indep_set, batch_size=BATCH_SIZE)
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    for i, (trained, valided) in enumerate(folds):
        train_set = TensorDataset(torch.Tensor(X[trained]), torch.Tensor(y[trained]))
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
        valid_set = TensorDataset(torch.Tensor(X[valided]), torch.Tensor(y[valided]))
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)
        net = NET(X.shape[1], y.shape[1], is_reg=reg)
        net.fit(train_loader, valid_loader, out='%s_%d' % (out, i), epochs=N_EPOCH, lr=LR)
        cvs[valided] = net.predict(valid_loader)
        inds += net.predict(indep_loader)
    return cvs, inds / 5


def Train_RF(X, y, out, reg=False):
    
    """ 
    Training of RF classifier or regressor.
    
    Arguments:
        X (np.ndarray): m x n feature matrix for cross validation, where m is the number of samples
            and n is the number of features.
        y (np.ndarray): m x l label matrix for cross validation, where m is the number of samples and
            equals to row of X, and l is the number of types.
        out (str): file path where the model is saved 
        reg (bool): it True, the training is for regression, otherwise for classification.
            
    """
    
    if reg:
        model = RandomForestRegressor(n_estimators=1000, n_jobs=10)
    else:
        model = RandomForestClassifier(n_estimators=1000, n_jobs=10)
    model.fit(X, y, sample_weight=[1 if v >= 4 else 0.1 for v in y])
    joblib.dump(model, out, compress=3)
    
def Train_SVM(X, y, out, reg=False):
    
    """ 
    Training of SVM classifier or regressor.
    
    Arguments:
        X (np.ndarray): m x n feature matrix for cross validation, where m is the number of samples
            and n is the number of features.
        y (np.ndarray): m x l label matrix for cross validation, where m is the number of samples and
            equals to row of X, and l is the number of types.
        out (str): file path where the model is saved 
        reg (bool): it True, the training is for regression, otherwise for classification.
            
    """
    
    if reg:
        alg = SVR()
    else:
        alg = SVC(probability=True)
    gs = GridSearchCV(deepcopy(alg), {'C': 2.0 ** np.array([-15, 15]), 'gamma': 2.0 ** np.array([-15, 15])}, n_jobs=10)
    gs.fit(X, y)
    params = gs.best_params_
    print(params)
    model = deepcopy(alg)
    model.C = params['C']
    model.gamma = params['gamma']
    if not reg:
        model.probability=True
    model.fit(X, y, sample_weight=[1 if v >= 4 else 0.1 for v in y])
    joblib.dump(model, out, compress=3)

def Train_DNN(X,y,out, reg=False):
    
    """ 
    Training of DNN classifier or regressor.
    
    Arguments:
        X (np.ndarray): m x n feature matrix for cross validation, where m is the number of samples
            and n is the number of features.
        y (np.ndarray): m x l label matrix for cross validation, where m is the number of samples and
            equals to row of X, and l is the number of types.
        out (str): file path where the model is saved 
        reg (bool): it True, the training is for regression, otherwise for classification.
            
    """
    NET = models.STFullyConnected
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    train_set = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
    valid_set = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)
    net = NET(X.shape[1], y.shape, is_reg=reg)
    net.fit(train_loader, valid_loader, out=out, epochs=N_EPOCH, lr=LR)
    
def Train_model(alg,X,y,out, reg=False):
    
    """ 
    Wrapping function to train and save a model.
    
    Arguments:
        alg (str): algorithm to use for clasification/regression
        X (np.ndarray): m x n feature matrix for cross validation, where m is the number of samples
            and n is the number of features.
        y (np.ndarray): m x l label matrix for cross validation, where m is the number of samples and
            equals to row of X, and l is the number of types.
        out (str): file path where the model is saved 
        reg (bool): it True, the training is for regression, otherwise for classification.
            
    """

    if alg == 'RF':
        if reg:
            model = RandomForestRegressor(n_estimators=1000, n_jobs=10)
        else:
            model = RandomForestClassifier(n_estimators=1000, n_jobs=10)
    elif alg == 'SVM':
        print('Train SVM...')
        Train_SVM(X,y,out, reg)
        return
    elif alg == 'KNN':
        if reg:
            model = KNeighborsRegressor(n_jobs=10)
        else:
            model = KNeighborsClassifier(n_jobs=10)
    elif alg == 'NB':
        model = GaussianNB()
    elif alg == 'PLS':
        model = PLSRegression()
    elif alg == 'DNN':
        Train_DNN(X, y, out, reg)
        return
    elif alg == 'XGB':
        if reg:
            alg = XGBRegressor(n_estimators=1000, n_jobs=10)
        else:
            alg = XGBClassifier(n_estimators=1000, n_jobs=10)
    if alg in ['KNN', 'PLS']:
        model.fit(X,y)
    else:
        model.fit(X, y, sample_weight=[1 if v >= 4 else 0.1 for v in y])
    joblib.dump(model, out, compress=3)


def mt_task(alg, args, reg=False):
    
    """
    Runs multitask classification/regression model

    Arguments:
        alg (str): algorithm to use for clasification/regression
        reg (bool): if True, the training is for regression, otherwise for classification.
        args (NameSpace): namespace containing input parameters from parser and input file column names 
    """
    
    alg = alg[3:]
    columns = args.columns
    
    # Load data
    df = pd.read_table(args.base_dir + '/data/' + args.input).dropna(subset=[columns['smiles'], columns['pchembl']])
    df = df[df[columns['target']].isin(args.targets)]
    df = df[list(columns.values())].set_index([columns['target'], columns['smiles']])
    
    # Create output dir and output file prefix
    d = '%s/envs/multi' % args.base_dir
    if not os.path.exists(d):
        os.makedirs(d)    
    targets = '_'.join(x for x in args.targets)
    out = '%s/%s_%s_%s' % (d, alg, 'REG' if reg else 'CLS', targets)
    print(out, end='\t')
    
    # Temporal split
    years = df[columns['year']].groupby(columns['smiles']).min().dropna()
    test_idx = years[years > args.year].index 
    
    # TO DO : implement keep_low_quality for MT
    
    # Calculate mean if multiple datapoints exist and drop unnecessary columns
    df = df[columns['pchembl']].groupby([columns['target'], columns['smiles']]).mean().dropna()

    # Output values
    print("active:", len(df[df >= args.activity_threashold]), \
            "not active:", len(df[df< args.activity_threashold]), end='\t')
    if not reg:
        df = (df > args.activity_threashold).astype(float)
    # Pivote data
    df = df.unstack(columns['target'])
    
    # Temporal split  
    test = df.loc[test_idx]
    data = df.drop(test.index)
    data = data.sample(len(data))
    print("train set:", len(df)-len(test_idx), "test set:", len(test_idx) )
        
    #calculate ecfp and physiochemical properties as input for the predictors
    test_x = utils.Predictor.calc_fp([Chem.MolFromSmiles(mol) for mol in test.index])
    data_x = utils.Predictor.calc_fp([Chem.MolFromSmiles(mol) for mol in data.index])

    #Perform standardization by centering and scaling
    scaler = Scaler(); scaler.fit(data_x)
    test_x = scaler.transform(test_x)
    data_x = scaler.transform(data_x)
    
    if args.save_model:
        #train model on all data and save model
        X = np.concatenate([data_x, test_x], axis=0)
        y = np.concatenate([data.values, test.values], axis=0)
        Train_model(alg, X, y, out=out + '.pkg', reg=reg)
    else:
        #cross validation
        data_p, test_p = DNN(data_x, data.values, test_x, test.values, out=out, reg=reg)
        data_p = pd.DataFrame(data_p, columns=[t+'_Score' for t in kwargs['targets']], index=data.index)
        test_p = pd.DataFrame(test_p, columns=[t+'_Score' for t in kwargs['targets']], index=test.index)
        data = pd.concat([data, data_p], axis=1)
        test = pd.concat([test, test_p], axis=1)
        data.to_csv(out + '.cv.tsv', sep='\t')

def single_task(target, args, alg='RF', reg=False):
    """
    Runs single task classification/regression model

    Arguments:
        target (str): target name
        alg (str): algorithm to use for clasification/regression
        reg (bool): if True, the training is for regression, otherwise for classification.
        args (NameSpace): namespace containing input parameters from parser and input file column names 
    """

    
    columns = args.columns    
    # Load target specific data
    df = pd.read_table(args.base_dir + '/data/' + args.input).dropna(subset=[columns['smiles'], columns['pchembl']])
    df = df[df[columns['target']] == target]
    df = df[list(columns.values())].set_index(columns['smiles'])
    
    d = '%s/envs/single' % args.base_dir
    if not os.path.exists(d):
        os.makedirs(d)    
    out = '%s/%s_%s_%s' % (d, alg, 'REG' if reg else 'CLS', target)
    print(out)

    # TO DO : fix keep_low_quality
    # If doing regression or if keep_low_quality False, remove low quality data points
#     if not kwargs['keep_low_quality'] or reg:
#         print(df[columns['quality']].unique())
#         df = df[df[columns['quality'] != 'Low']]
    
    # Output values
    activity = df[columns['pchembl']]
    print("active:", len(activity[activity >= args.activity_threashold]), \
            "not active:", len(activity[activity < args.activity_threashold]))
    if not reg:
        activity = (activity > args.activity_threashold).astype(float)
         
    # Suffle dataframe
    activity = activity.sample(len(activity))   
    
    # Temporal split
    years = df[columns['year']].groupby(columns['smiles']).min().dropna()
    test_idx = years[years > args.year].index    
    test = activity.loc[test_idx]
    data = activity.drop(test.index)
    print("train set:", len(activity)-len(test_idx), "test set:", len(test_idx) )
        
    #calculate ecfp and physiochemical properties as input for the predictors
    test_x = utils.Predictor.calc_fp([Chem.MolFromSmiles(mol) for mol in test.index])
    data_x = utils.Predictor.calc_fp([Chem.MolFromSmiles(mol) for mol in data.index])
    
    if alg != 'RF':
        #Perform standardization by centering and scaling
        scaler = Scaler(); scaler.fit(data_x)
        test_x = scaler.transform(test_x)
        data_x = scaler.transform(data_x)
    
    if args.save_model:
        #train model on all data and save model
        X = np.concatenate([data_x, test_x], axis=0)
        y = np.concatenate([data.values, test.values], axis=0)
        Train_model(alg, X, y, out=out + '.pkg', reg=reg)
    else:
        #cross validation
        data, test = data.to_frame(name='Label'), test.to_frame(name='Label')
        data['Score'], test['Score'] = cross_validation(data_x, data.values, test_x, test.values, alg, out, reg=reg)
        data.to_csv(out + '.cv.tsv', sep='\t')
        test.to_csv(out + '.ind.tsv', sep='\t')


def cross_validation(X, y, X_ind, y_ind, alg='DNN', out=None, reg=False):
    """ 
    Wrapping function to do cross validation and independent test of a model.
    
    Arguments:
        X (np.ndarray): m x n feature matrix for cross validation, where m is the number of samples
            and n is the number of features.
        y (np.ndarray): m-d label array for cross validation, where m is the number of samples and
            equals to row of X.
        X_ind (np.ndarray): m x n Feature matrix for independent set, where m is the number of samples
            and n is the number of features.
        y_ind (np.ndarray): m-d label array for independent set, where m is the number of samples and
            equals to row of X_ind, and l is the number of types.
        reg (bool): it True, the training is for regression, otherwise for classification.
     Returns:
        cvs (np.ndarray): m x l result matrix for cross validation, where m is the number of samples and
            equals to row of X, and l is the number of types and equals to row of X.
        inds (np.ndarray): m x l result matrix for independent test, where m is the number of samples and
            equals to row of X, and l is the number of types and equals to row of X.  
    """
    
    if alg == 'RF':
        cv, ind = RF(X, y[:, 0], X_ind, y_ind[:, 0], reg=reg)
    elif alg == 'SVM':
        cv, ind = SVM(X, y[:, 0], X_ind, y_ind[:, 0], reg=reg)
    elif alg == 'KNN':
        cv, ind = KNN(X, y[:, 0], X_ind, y_ind[:, 0], reg=reg)
    elif alg == 'NB':
        cv, ind = NB(X, y[:, 0], X_ind, y_ind[:, 0])
    elif alg == 'PLS':
        cv, ind = PLS(X, y[:, 0], X_ind, y_ind[:, 0])
    elif alg == 'DNN':
        cv, ind = DNN(X, y, X_ind, y_ind, out=out, reg=reg)
    elif alg == 'XGB':
        cv, ind = XGB(X, y[:, 0], X_ind, y_ind[:, 0], reg=reg)
    return cv, ind


def EnvironmentArgParser(txt=None):
    """ Define and read command line arguments """
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains a folder 'data' with input files")
    parser.add_argument('-i', '--input', type=str, default='LIGAND_RAW.tsv',
                        help="Input file name in 'data' containing raw data")   
    parser.add_argument('-s', '--save_model', action='store_true',
                        help="If included then then the model will be trained on all data and saved")   
    parser.add_argument('-m', '--model_types', type=str, nargs='*', default=['RF', 'XGB', 'DNN', 'SVM', 'PLS', 'NB', 'KNN', 'MT_DNN'],
                        help="Modeltype, defaults to run all modeltypes, choose from: 'RF', 'XGB', 'DNN', 'SVM', 'PLS' (only with REG), 'NB' (only with CLS) 'KNN' or 'MT_DNN'") 
    parser.add_argument('-r', '--regression', type=str, default=None,
                        help="If True, only regression model, if False, only classification, default both")
    parser.add_argument('-t', '--targets', type=str, nargs='*', default=['P29274', 'P29275', 'P30542','P0DMS8'],
                        help="Target indentifiers") 
    parser.add_argument('-a', '--activity_threashold', type=float, default=6.5,
                        help="Activity threashold")
#     parser.add_argument('-l', '--keep_low_quality', action='store_true',
#                         help="If included keeps low quality data")
    parser.add_argument('-y', '--year', type=int, default=2015,
                        help="Temporal split limit")  
    parser.add_argument('-ncpu', '--ncpu', type=int, default=8,
                        help="Number of CPUs")   
    parser.add_argument('-bs', '--batch_size', type=int, default=2048,
                        help="Batch size")
    parser.add_argument('-e', '--epochs', type=int, default=1000,
                        help="Number of epochs")
    
    if txt:
        args = parser.parse_args(txt)
    else:
        args = parser.parse_args()
    
    # If no regression argument, does bith regression and classification
    if args.regression is None: 
        args.regression = [True, False]
    elif args.regression.lower() == 'true':
        args.regression = [True]
    elif args.regression.lower() == 'false':
        args.regression = [False]
    
    return args 

def Environment(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['OMP_NUM_THREADS'] = str(args.ncpu)
    
    if not os.path.exists(args.base_dir + '/envs'):
        os.makedirs(args.base_dir + '/envs')  
    
    for reg in args.regression:
        LR = 1e-4 if reg else 1e-5
        for model in args.model_types:
            if model.startswith('MT_'):
                # Train and validate multitask model
                mt_task(model, args, reg=reg) 
            elif ( reg is True and model == 'NB' ) or (reg is False and model == 'PLS'):
                # Skip in case of NB regression and PSL classification
                continue
            else:
                # Train and validate single task model
                for target in args.targets:
                    single_task(target, args, model, reg=reg)
    
if __name__ == '__main__':

    args = EnvironmentArgParser()
    columns = {'target' : 'accession',
               'smiles' : 'SMILES',
               'pchembl' : 'pchembl_value_Mean',
               'data_type' : 'Standard_Type',
               'relation' : 'Standard_Type',
               'quality' : 'Quality',
               'year' : 'Year'}
    args.columns = columns
    args.git_commit = utils.commit_hash()    
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    with open(args.base_dir + '/env_args.json', 'w') as f:
        json.dump(vars(args), f)
        
    BATCH_SIZE = args.batch_size
    N_EPOCH = args.epochs
        
    Environment(args)
    
