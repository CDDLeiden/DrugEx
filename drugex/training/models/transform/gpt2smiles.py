import tempfile

import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from tqdm.auto import tqdm
import numpy as np


from drugex import DEFAULT_DEVICE, DEFAULT_GPUS
from drugex.data.fragments import SequenceFragmentEncoder, FragmentCorpusEncoder
from drugex.data.processing import Standardization
from drugex.data.datasets import SmilesFragDataSet
from drugex.molecules.converters.fragmenters import Fragmenter
from .layer import PositionalEmbedding, PositionwiseFeedForward, SublayerConnection
from .layer import pad_mask, tri_mask
from drugex.training.models.encoderdecoder import SmilesFragsGeneratorBase
from drugex.utils import ScheduledOptim
from drugex.training.scorers.smiles import SmilesChecker
from torch import optim


class Block(nn.Module):
    def __init__(self, d_model, n_head, d_inner):
        super(Block, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.pffn = PositionwiseFeedForward(d_model, d_inner)
        self.connector = nn.ModuleList([SublayerConnection(d_model) for _ in range(2)])

    def forward(self, x, key_mask=None, atn_mask=None):
        x = self.connector[0](x, lambda x: self.attn(x, x, x, key_mask, attn_mask=atn_mask)[0])
        x = self.connector[1](x, self.pffn)
        return x


class GPT2Layer(nn.Module):
    def __init__(self, voc, d_emb=512, d_model=512, n_head=12, d_inner=1024, n_layer=12, pad_idx=0):
        super(GPT2Layer, self).__init__()
        self.n_layer = n_layer
        self.d_emb = d_emb
        self.d_model = d_model
        self.n_head = n_head
        self.voc = voc
        self.pad_idx = pad_idx

        self.token_emb = nn.Embedding(voc.size, self.d_emb, padding_idx=pad_idx)
        self.posit_emb = PositionalEmbedding(self.d_emb, max_len=voc.max_len + voc.max_len)

        self.blocks = nn.ModuleList([Block(self.d_emb, self.n_head, d_inner=d_inner) for _ in range(self.n_layer)])

        self.layer_norm = nn.LayerNorm(self.d_emb)
        self.word_prj = nn.Linear(self.d_emb, self.voc.size)
        kaiming_normal_(self.word_prj.weight, nonlinearity="linear")

    def forward(self, input: torch.Tensor, key_mask=None, atn_mask=None):
        hidden_states = self.posit_emb(input) + self.token_emb(input)

        for block in self.blocks:
            hidden_states = block(hidden_states, key_mask=key_mask, atn_mask=atn_mask)

        hidden_states = self.word_prj(hidden_states)
        return hidden_states


class GPT2Model(SmilesFragsGeneratorBase):
    def __init__(self, voc_trg, d_emb=512, d_model=512, n_head=8, d_inner=1024, n_layer=12, pad_idx=0, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS):
        super(GPT2Model, self).__init__(device=device, use_gpus=use_gpus)
        self.mol_type = 'smiles'
        self.voc_trg = voc_trg
        self.pad_idx = pad_idx
        self.gpt2 = GPT2Layer(self.voc_trg, d_emb=d_emb, d_model=d_model,
                              n_head=n_head, d_inner=d_inner, n_layer=n_layer,
                              pad_idx=pad_idx)
        self.init_states()
        self.optim = ScheduledOptim(
            optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9), 0.5, d_model)
        # self.optim = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, src, trg=None):
        if trg is not None:
            input = torch.cat([src, trg], dim=1)
            key_mask = pad_mask(input, self.pad_idx)
            atn_mask = tri_mask(input)
            start, end = src.size(1) - 1, -1
            input = input.transpose(0, 1)
            dec = self.gpt2(input, key_mask=key_mask, atn_mask=atn_mask)[start:end, :, :]
            dec = dec.transpose(0, 1).log_softmax(dim=-1)
            out = dec.gather(2, trg.unsqueeze(2)).squeeze(2)
        else:
            seq_len = self.voc_trg.max_len + self.voc_trg.max_len
            out = torch.zeros(len(src), seq_len).long().to(src.device)
            out[:, :src.size(1)] = src
            is_end = torch.zeros(len(src)).bool().to(src.device)
            for step in range(self.voc_trg.max_len):  # decode up to max length
                input = out[:, :src.size(1)+step]
                key_mask = pad_mask(input, self.pad_idx)
                atn_mask = tri_mask(input)
                dec = self.gpt2(input.transpose(0, 1), key_mask=key_mask, atn_mask=atn_mask)
                x = dec.softmax(dim=-1)[-1, :, :].multinomial(1).view(-1)
                # prev = dec[:, -1, :].argmax(-1)
                x[is_end] = self.voc_trg.tk2ix['_']
                is_end |= x == self.voc_trg.tk2ix['EOS']
                out[:, src.size(1)+step] = x
                if is_end.all(): break
            out = out[:, self.voc_trg.max_len:].detach()
        return out

    def sample_smiles(self, input_smiles, num_samples=100, batch_size=32, n_proc=1, fragmenter=None,
                      keep_frags=True, drop_duplicates=True, drop_invalid=True, progress=True, tqdm_kwargs={}):
        standardizer = Standardization(n_proc=n_proc)
        smiles = standardizer.apply(input_smiles)

        fragmenter = Fragmenter(4, 4, 'brics') if not fragmenter else fragmenter
        encoder = FragmentCorpusEncoder(
            fragmenter=fragmenter,
            encoder=SequenceFragmentEncoder(
                self.voc_trg
            ),
            n_proc=n_proc
        )
        out_data = SmilesFragDataSet(tempfile.NamedTemporaryFile().name)
        encoder.apply(smiles, encodingCollectors=[out_data])

        # Duplicate of self.sample to allow dropping molecules
        # and progress bar on the fly without additional
        # overhead caused by calling nn.DataParallel a few times
        net = nn.DataParallel(self, device_ids=self.gpus)

        if progress:
            tqdm_kwargs.update({'total': num_samples, 'desc': 'Generating molecules'})
            pbar = tqdm(**tqdm_kwargs)

        smiles, frags = [], []
        while not len(smiles) >= num_samples:
            with torch.no_grad():
                src = next(iter(out_data.asDataLoader(batch_size, n_samples=batch_size)))
                trg = net(src.to(self.device))
                new_smiles = self.voc_trg.decode(trg, is_tk=False)
                new_frags = self.voc_trg.decode(src, is_tk=False, is_smiles=True)
                # drop duplicates
                if drop_duplicates:
                    new_smiles = np.array(new_smiles)
                    new_frags = np.array(new_frags)[np.logical_not(np.isin(new_smiles, smiles))].tolist()
                    new_smiles = new_smiles[np.logical_not(np.isin(new_smiles, smiles))].tolist()
                # drop invalid smiles
                if drop_invalid:
                    # Make sure both valid molecules and include input fragments
                    scores = SmilesChecker.checkSmiles(new_smiles, frags=new_frags).sum(axis=1)
                    new_smiles = np.array(new_smiles)[scores > 1].tolist()
                    new_frags = np.array(new_frags)[scores > 1].tolist()
                smiles += new_smiles
                frags += new_frags
                # Update progress bar
                if progress:
                    pbar.update(len(new_smiles) if pbar.n + len(new_smiles) <= num_samples else num_samples - pbar.n)
        if progress:
            pbar.close()
        smiles = smiles[:num_samples]
        if keep_frags:
            return smiles, frags
        return smiles

