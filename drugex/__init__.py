"""
__init__.py

Created by: Martin Sicho
On: 06.04.22, 16:51
"""
import torch
from rdkit import rdBase

rdBase.DisableLog('rdApp.info')
rdBase.DisableLog('rdApp.warning')

torch.set_num_threads(1)

DEFAULT_DEVICE = torch.device('cuda')
DEFAULT_DEVICE_ID = 0
