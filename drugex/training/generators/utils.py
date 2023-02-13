''' Define the Layers '''
import math
import torch
import torch.nn as nn
import numpy as np

def pad_mask(seq, pad_idx=0):
    return seq == pad_idx


def tri_mask(seq, diag=1):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    masks = torch.ones((len_s, len_s)).triu(diagonal=diag)
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


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise

    def forward(self, x):
        y = self.w_1(x).relu()
        y = self.w_2(y)
        return y


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm
    """
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """ Apply residual connection to any sublayer with the same size"""
        y = sublayer(x)
        y = self.dropout(y)
        y = self.norm(x + y)
        return y


class PositionalEmbedding(nn.Module):
    """ 
    Positional embedding for sequence transformer
    """
    def __init__(self, d_model: int, max_len=100, batch_first=False):
        super(PositionalEmbedding, self).__init__()

        self.batch_first = batch_first
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            y = self.pe[:x.size(1), :].unsqueeze(0).detach()
        else:
            y = self.pe[:x.size(0), :].unsqueeze(1).detach()
        return y


class PositionalEncoding(nn.Module):
    """
    Positional encoding for graph transformer
    """
    def __init__(self, d_model: int, max_len=100, batch_first=False):
        super(PositionalEncoding, self).__init__()

        self.batch_first = batch_first
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        bsize, sqlen = x.size()
        y = x.reshape(bsize * sqlen)
        code = self.pe[y, :].view(bsize, sqlen, -1)
        # if sqlen > 10:
        #     assert code[10, 5, 8] == self.pe[x[10, 5], 8]
        return code