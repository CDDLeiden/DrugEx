from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors as desc
from rdkit.Chem import QED
import pandas as pd
from rdkit import DataStructs
import numpy as np

from scipy import linalg
import torch
from torch.nn import functional as F


def pad_mask(seq, pad_idx=0):
    return seq == pad_idx


def tri_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    masks = torch.triu(torch.ones((len_s, len_s)), diagonal=1)
    masks = masks.bool().to(seq.device)
    return masks


def unique(arr):
    # Finds unique rows in arr and return their indices
    if type(arr) == torch.Tensor:
        arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    idxs = np.sort(idxs)
    if type(arr) == torch.Tensor:
        idxs = torch.LongTensor(idxs).to(arr.get_device())
    return idxs
