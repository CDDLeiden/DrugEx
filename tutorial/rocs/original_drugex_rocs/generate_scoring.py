
from drugex.training.scorers.interfaces import Scorer
import sys
import pandas as pd
from Model_scorer import ModelScorer
from drugex.training.scorers.properties import Property
from drugex.training.scorers.modifiers import SmoothClippedScore
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

sascore = Property("SA")
model_scorer = ModelScorer(qfnames=['model3-4_v1.sq'], score='TanimotoCombo',experiment_name='generation') 
scorers = [
model_scorer,
    sascore
]
thresholds = [
    0.35, #everything below thrown away(TanimotoCombo score) = desired ratio
    0.1
]

environment = DrugExEnvironment(scorers, thresholds, reward_scheme=ParetoCrowdingDistance())
smiles = pd.read_csv("/home/sem23/fastrocs/rp1-sem-egbers/generated_eps0.01_pt1000_ns10.tsv")['SMILES']
results = environment.getScores(smiles)
results['SMILES'] = smiles
results.to_csv('eps0.01_pt1000_scores.csv', index=False)
