from drugex.training.scorers.interfaces import Scorer
import sys
from openeye import oeomega

import pandas as pd
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors

import os
from openeye import oechem
from openeye import oefastrocs
from openeye import oeshape

from drugex.training.scorers.properties import Property
from drugex.training.scorers.modifiers import SmoothClippedScore
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance

from drugex.training.explorers import SequenceExplorer
import warnings
from drugex.training.generators import SequenceRNN
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.training.monitors import FileMonitor

N_SAMPLES = 10000
GPUS = [0]
EPSILON = 0.01
BATCH_SIZE = 128
PATIENCE = 1000
N_EPOCHS = 1000

EXPERIMENT_NAME = f"RL_ns{N_SAMPLES}_eps{EPSILON}_bs{BATCH_SIZE}_pt{PATIENCE}_ne{N_EPOCHS}"
MODEL_DIR_RL = f"./DrugEx_script/{EXPERIMENT_NAME}"
if not os.path.exists(MODEL_DIR_RL):
    os.makedirs(MODEL_DIR_RL, exist_ok=True)


# redirect all output to the file with experiment name
sys.stdout = open(f"{MODEL_DIR_RL}/{EXPERIMENT_NAME}.log", "w")
# now error output will be redirected to the same file
sys.stderr = sys.stdout

from Model_scorer import ModelScorer
model_scorer = ModelScorer(qfnames=['model3-4_v1.sq'], score='TanimotoCombo', experiment_name=EXPERIMENT_NAME)  


# Environment


sascore = Property("SA")
sascore.setModifier(SmoothClippedScore(lower_x=5, upper_x=3))
# 
scorers = [
    model_scorer,
    sascore
]
thresholds = [
    0.35, #everything below thrown away(TanimotoCombo score)
    0.1
]

environment = DrugExEnvironment(scorers, thresholds, reward_scheme=ParetoCrowdingDistance())

# Explorer

warnings.filterwarnings('ignore')
GPUS = [0] # we will use only one GPU with ID=0, but if you have more, you can list more GPU IDs here
MODEL_DIR = "./DrugEx_script/"
MODELS_PR_PATH = "./DrugEx_v2_PT_Papyrus05.5/"
voc = VocSmiles.fromFile(f"{MODELS_PR_PATH}/Papyrus05.5_smiles_rnn_PT.vocab", encode_frags=False)
pretrained = SequenceRNN(voc, is_lstm=True, use_gpus=GPUS)
pretrained.loadStatesFromFile(f'{MODELS_PR_PATH}/Papyrus05.5_smiles_rnn_PT.pkg')

finetuned = SequenceRNN(voc, is_lstm=True, use_gpus=GPUS)
finetuned.loadStatesFromFile(f'{MODEL_DIR}/CCR2_finetuned.pkg')

explorer = SequenceExplorer(
    agent = pretrained,
    env = environment,
    mutate = finetuned, # network introducing "random mutations" to the generated structures (rate determined by epsilon)
    epsilon = EPSILON,
    n_samples = N_SAMPLES,
    use_gpus = GPUS,
    batch_size = BATCH_SIZE
)

monitor = FileMonitor(os.path.join(MODEL_DIR_RL, 'CCR2_agent'), save_smiles=True, reset_directory=True)
explorer.fit(monitor=monitor, epochs=N_EPOCHS, patience=PATIENCE)
