'''
This script was used to create the test QSAR models in this folder.
Data is QSPRpred tutorial data, which can be downloaded from the following link:
https://1drv.ms/u/s!AtzWqu0inkjX3QRxXOkTFNv7IV7u?e=PPj0O2
'''
import os

import pandas as pd
from qsprpred.data import QSPRDataset
from qsprpred.data.descriptors.fingerprints import MorganFP
from qsprpred.models import SklearnModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer

#####----Load Data----#####
df = pd.read_csv('A2A_LIGANDS.tsv', sep='\t')

df_multitask = pd.read_csv('AR_LIGANDS.tsv', sep='\t')
df_multitask = df_multitask.pivot(
    index="SMILES",
    columns="accession",
    values="pchembl_value_Mean"
)
df_multitask.columns.name = None
df_multitask.reset_index(inplace=True)


#####----Create Single Task Regression Model----#####

dataset = QSPRDataset(
    df=df.copy(),
    name="A2ARTestDataset",
    target_props=[{"name": "pchembl_value_Mean", "task": "REGRESSION"}],
    random_state=42
)
dataset.prepareDataset(
    feature_calculators=[MorganFP(radius=2, nBits=1024)],
)

model = SklearnModel(
    base_dir=".",
    alg=RandomForestRegressor,
    name="A2AR_RF_reg",
    parameters={"n_estimators": 100, "max_depth": 5},
)
model.fitDataset(dataset)
_ = model.save()

#####----Create Single Task Classification Model----#####

dataset = QSPRDataset(
    df=df.copy(),
    name="A2ARTestDataset",
    target_props=[{"name": "pchembl_value_Mean", "task": "SINGLECLASS", "th": [6.5]}],
    random_state=42,
)
dataset.prepareDataset(
    feature_calculators=[MorganFP(radius=2, nBits=1024)],
)

model = SklearnModel(
    base_dir=".",
    alg=RandomForestClassifier,
    name="A2AR_RF_cls",
    parameters={"n_estimators": 100, "max_depth": 5},
)
model.fitDataset(dataset)
_ = model.save()

#####----Create Multi Class Classification Model----#####

dataset = QSPRDataset(
    df=df.copy(),
    name="A2ARTestDataset",
    target_props=[{"name": "pchembl_value_Mean", "task": "MULTICLASS", "th": [0, 5.5, 7, 12]}],
    random_state=42,
)
dataset.prepareDataset(
    feature_calculators=[MorganFP(radius=2, nBits=1024)],
)

model = SklearnModel(
    base_dir=".",
    alg=RandomForestClassifier,
    name="A2AR_RF_multicls",
    parameters={"n_estimators": 100, "max_depth": 5},
)
model.fitDataset(dataset)
_ = model.save()


#####----Create Multi Task Regression Model----#####

target_props = [
    {"name": "P0DMS8", "task": "REGRESSION", "imputer": SimpleImputer(strategy="mean")},
    {"name": "P29274", "task": "REGRESSION", "imputer": SimpleImputer(strategy="mean")},
    {"name": "P29275", "task": "REGRESSION", "imputer": SimpleImputer(strategy="mean")},
    {"name": "P30542", "task": "REGRESSION", "imputer": SimpleImputer(strategy="mean")}]

dataset = QSPRDataset(
    name="MultiTaskTestDataset",
    df=df_multitask.copy(),
    target_props=target_props,
    random_state=42,
)
dataset.prepareDataset(
    feature_calculators=[MorganFP(radius=2, nBits=1024)],
)

model = SklearnModel(
    base_dir=".",
    alg=RandomForestRegressor,
    name="AR_RF_reg",
    parameters={"n_estimators": 100, "max_depth": 5},
)
model.fitDataset(dataset)
_ = model.save()


#####----Create Multi Task Classification Model----#####

target_props = [
    {"name": "P0DMS8", "task": "SINGLECLASS", "th":[6.5], "imputer": SimpleImputer(strategy="mean")},
    {"name": "P29274", "task": "SINGLECLASS", "th":[6.5], "imputer": SimpleImputer(strategy="mean")},
    {"name": "P29275", "task": "SINGLECLASS", "th":[6.5], "imputer": SimpleImputer(strategy="mean")},
    {"name": "P30542", "task": "SINGLECLASS", "th":[6.5], "imputer": SimpleImputer(strategy="mean")}
]

dataset = QSPRDataset(
    name="MultiTaskTestDataset",
    df=df_multitask.copy(),
    target_props=target_props,
    random_state=42,
)
dataset.prepareDataset(
    feature_calculators=[MorganFP(radius=2, nBits=1024)],
)

model = SklearnModel(
    base_dir=".",
    alg=RandomForestClassifier,
    name="AR_RF_cls",
    parameters={"n_estimators": 100, "max_depth": 5},
)
model.fitDataset(dataset)
_ = model.save()