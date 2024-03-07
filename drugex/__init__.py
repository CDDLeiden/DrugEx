import torch
from rdkit import rdBase

from .about import VERSION

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.info')
rdBase.DisableLog('rdApp.warning')

torch.set_num_threads(1)

DEFAULT_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEFAULT_GPUS = (0,)

__version__ = VERSION
