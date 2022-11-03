"""
commons

Created by: Martin Sicho
On: 26.10.22, 16:08
"""

import drugex
import os
import nvgpu

DATA_DIR = './data'
INPUT_FILE = os.path.join(DATA_DIR, 'chembl_30_100000.smi')
OUT_DIR=f'output'
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_FILE = f'{OUT_DIR}/modeltest' if not 'EXPERIMENT_ID' in os.environ else f'{OUT_DIR}/{os.environ["EXPERIMENT_ID"]}'
N_PROC = int(os.environ['NCPUS'])
GPUS = [int(x['index']) for x in nvgpu.gpu_info()]
CHUNK_SIZE = 1000
BATCH_SIZE = 256
N_EPOCHS = 30 if not 'N_EPOCHS' in os.environ else int(os.environ['N_EPOCHS'])
DRUGEX_VERSION = drugex.__version__


print("Using DrugEx version: ", DRUGEX_VERSION)
