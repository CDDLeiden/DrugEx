import drugex
import os
import nvgpu

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
INPUT_FILE = os.path.join(DATA_DIR, 'chembl_30_1000.smi' if 'INPUT_FILE' not in os.environ else os.environ['INPUT_FILE'])
assert os.path.exists(INPUT_FILE)
OUT_DIR=os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUT_DIR, exist_ok=True)
MODEL = 'rnn' if not 'MODEL' in os.environ else os.environ["MODEL"]
EXPERIMENT_ID = 'modeltest' if not 'EXPERIMENT_ID' in os.environ else os.environ["EXPERIMENT_ID"]
OUTPUT_FILE = f'{OUT_DIR}/{EXPERIMENT_ID}'
N_PROC = int(os.environ['NCPUS']) if 'NCPUS' in os.environ else 12
GPUS = [int(x['index']) for x in nvgpu.gpu_info()]
CHUNK_SIZE = 1000
BATCH_SIZE = int(os.environ['BATCH_SIZE']) if 'BATCH_SIZE' in os.environ else 64
N_EPOCHS = 5 if not 'N_EPOCHS' in os.environ else int(os.environ['N_EPOCHS'])
DRUGEX_VERSION = drugex.__version__
