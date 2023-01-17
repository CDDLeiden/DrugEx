import tempfile

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from drugex import DEFAULT_GPUS, DEFAULT_DEVICE
from drugex.data.fragments import GraphFragmentEncoder, FragmentCorpusEncoder
from drugex.data.processing import Standardization
from drugex.data.datasets import GraphFragDataSet
from drugex.logs import logger
from drugex.molecules.converters.fragmenters import Fragmenter
from drugex.training.generators.transformer_utils import PositionwiseFeedForward, SublayerConnection, PositionalEncoding, tri_mask
from drugex.training.interfaces import Generator
from drugex.utils import ScheduledOptim
from drugex.training.scorers.smiles import SmilesChecker
from torch import optim

from ..monitors import NullMonitor


class Block(nn.Module):
    def __init__(self, d_model, n_head, d_inner):
        super(Block, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.pffn = PositionwiseFeedForward(d_model, d_inner)
        self.connector = nn.ModuleList([SublayerConnection(d_model) for _ in range(2)])

    def forward(self, x, key_mask=None, attn_mask=None):
        x = self.connector[0](x, lambda x: self.attn(x, x, x, key_mask, attn_mask=attn_mask)[0])
        x = self.connector[1](x, self.pffn)
        return x


class AtomLayer(nn.Module):
    def __init__(self, d_model=512, n_head=8, d_inner=1024, n_layer=12):
        super(AtomLayer, self).__init__()
        self.n_layer = n_layer
        self.d_model = d_model
        self.n_head = n_head
        self.blocks = nn.ModuleList([Block(self.d_model, self.n_head, d_inner=d_inner)
                                     for _ in range(self.n_layer)])

    def forward(self, x: torch.Tensor, key_mask=None, attn_mask=None):
        for block in self.blocks:
            x = block(x, key_mask=key_mask, attn_mask=attn_mask)
        return x


class GraphTransformer(Generator):
    def __init__(self, voc_trg, d_emb=512, d_model=512, n_head=8, d_inner=1024, n_layer=12, pad_idx=0, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS):
        super(GraphTransformer, self).__init__(device=device, use_gpus=use_gpus)
        self.mol_type = 'graph'
        self.voc_trg = voc_trg
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.n_grows = voc_trg.max_len - voc_trg.n_frags - 1
        self.n_frags = voc_trg.n_frags + 1
        self.d_emb = d_emb
        self.emb_word = nn.Embedding(voc_trg.size * 4, self.d_emb, padding_idx=pad_idx)
        self.emb_atom = nn.Embedding(voc_trg.size, self.d_emb, padding_idx=pad_idx)
        self.emb_loci = nn.Embedding(self.n_grows, self.d_emb)
        self.emb_site = PositionalEncoding(self.d_emb, max_len=self.n_grows*self.n_grows)
        self.attn = AtomLayer(d_model=d_model, n_head=n_head, d_inner=d_inner, n_layer=n_layer)

        self.rnn = nn.GRUCell(self.d_model, self.d_model)
        self.prj_atom = nn.Linear(d_emb, self.voc_trg.size)
        self.prj_bond = nn.Linear(d_model, 4)
        self.prj_loci = nn.Linear(d_model, self.n_grows)
        self.init_states()

        self.optim = ScheduledOptim(
            optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9), 0.1, d_model)
        # self.optim = optim.Adam(self.parameters(), lr=1e-4)

        self.model_name = 'GraphTransformer'

    def init_states(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.attachToGPUs(self.gpus)

    def attachToGPUs(self, gpus):
        self.gpus = gpus
        self.to(self.device)

    def forward(self, src, is_train=False):
        # src: [batch_size, 80 , 5]
        if is_train:
            # Return loss
            # Src - not using last column (??), trg - not using first column (start token)
            src, trg = src[:, :-1, :], src[:, 1:, :]
            batch, sqlen, _ = src.shape
            triu = tri_mask(src[:, :, 0])

            # dec - atom environment
            emb = self.emb_word(src[:, :, 3] + src[:, :, 0] * 4)
            emb += self.emb_site(src[:, :, 1] * self.n_grows + src[:, :, 2])
            dec = self.attn(emb.transpose(0, 1), attn_mask=triu)

            dec = dec.transpose(0, 1).reshape(batch * sqlen, -1)
            out_atom = self.prj_atom(dec).log_softmax(dim=-1).view(batch, sqlen, -1)
            out_atom = out_atom.gather(2, trg[:, :, 0].unsqueeze(2))

            atom = self.emb_atom(trg[:, :, 0]).reshape(batch * sqlen, -1)
            dec = self.rnn(atom, dec)
            out_bond = self.prj_bond(dec).log_softmax(dim=-1).view(batch, sqlen, -1)
            out_bond = out_bond.gather(2, trg[:, :, 3].unsqueeze(2))

            word = self.emb_word(trg[:, :, 3] + trg[:, :, 0] * 4)
            word = word.reshape(batch * sqlen, -1)
            dec = self.rnn(word, dec)
            out_prev = self.prj_loci(dec).log_softmax(dim=-1).view(batch, sqlen, -1)
            out_prev = out_prev.gather(2, trg[:, :, 2].unsqueeze(2))

            curr = self.emb_loci(trg[:, :, 2]).reshape(batch * sqlen, -1)
            dec = self.rnn(curr, dec)
            out_curr = self.prj_loci(dec).log_softmax(dim=-1).view(batch, sqlen, -1)
            out_curr = out_curr.gather(2, trg[:, :, 1].unsqueeze(2))

            out = [out_atom, out_curr, out_prev, out_bond]
        else:
            # Return encoded molecules
            is_end = torch.zeros(len(src)).bool().to(src.device)
            exists = torch.zeros(len(src), self.n_grows, self.n_grows).long().to(src.device)
            vals_max = torch.zeros(len(src), self.n_grows).long().to(src.device)
            frg_ids = torch.zeros(len(src), self.n_grows).long().to(src.device)
            order = torch.LongTensor(range(len(src))).to(src.device)
            curr = torch.zeros(len(src)).long().to(src.device) - 1
            blank = torch.LongTensor(len(src)).to(src.device).fill_(self.voc_trg.tk2ix['*'])
            single = torch.ones(len(src)).long().to(src.device)
            voc_mask = self.voc_trg.masks.to(src.device)

            # The part of growing
            for step in range(1, self.n_grows):
               # if step == 70:
               #     print(step)
                if is_end.all():
                    src[:, step, :] = 0
                    continue
                data = src[:, :step, :]
                triu = tri_mask(data[:, :, 0])
                emb = self.emb_word(data[:, :, 3] + data[:, :, 0] * 4)
                emb += self.emb_site(data[:, :, 1] * self.n_grows + data[:, :, 2])
                dec = self.attn(emb.transpose(0, 1), attn_mask=triu)
                dec = dec[-1, :, :]

                grow = src[:, step, 4] == 0
                mask = voc_mask.repeat(len(src), 1) < 0
                if step <= 2:
                    mask[:, -1] = True
                else:
                    judge = (vals_rom == 0) | (exists[order, curr, :] != 0)
                    judge[order, curr] = True
                    judge = judge.all(dim=1) | (vals_rom[order, curr] == 0)
                    mask[judge, -1] = True
                mask[:, 1] = True
                mask[is_end, 1:] = True
                out_atom = self.prj_atom(dec).softmax(dim=-1)
                atom = out_atom.masked_fill(mask, 0).multinomial(1).view(-1)
                src[grow, step, 0] = atom[grow]
                atom = src[:, step, 0]
                is_end |= (atom == 0) & grow
                num = (vals_max > 0).sum(dim=1)
                vals_max[order, num] = voc_mask[atom]
                vals_rom = vals_max - exists.sum(dim=1)


                bud = atom != self.voc_trg.tk2ix['*']
                curr += bud
                curr[is_end] = 0
                src[:, step, 1] = curr
                exist = exists[order, curr, :] != 0

                mask = torch.zeros(len(src), 4).bool().to(src.device)
                for i in range(1, 4):
                    judge = (vals_rom < i) | exist
                    judge[order, curr] = True
                    mask[:, i] = judge.all(dim=1) | (vals_rom[order, curr] < i)
                mask[:, 0] = False if step == 1 else True
                mask[is_end, 0] = False
                mask[is_end, 1:] = True

                atom_emb = self.emb_atom(atom)
                dec = self.rnn(atom_emb, dec)
                out_bond = self.prj_bond(dec).softmax(dim=-1)
                bond = out_bond.masked_fill(mask, 0).multinomial(1).view(-1)
                src[grow, step, 3] = bond[grow]
                bond = src[:, step, 3]

                mask = (vals_max == 0) | exist | (vals_rom < bond.unsqueeze(-1))
                mask[order, curr] = True
                if step <= 2:
                    mask[:, 0] = False
                mask[is_end, 0] = False
                mask[is_end, 1:] = True
                word_emb = self.emb_word(atom * 4 + bond)
                dec = self.rnn(word_emb, dec)
                prev_out = self.prj_loci(dec).softmax(dim=-1)
                prev = prev_out.masked_fill(mask, 0).multinomial(1).view(-1)
                src[grow, step, 2] = prev[grow]
                prev = src[:, step, 2]

                for i in range(len(src)):
                    if not grow[i]:
                        frg_ids[i, curr[i]] = src[i, step, -1]
                    elif bud[i]:
                        frg_ids[i, curr[i]] = frg_ids[i, prev[i]]
                    obj = frg_ids[i, curr[i]].clone()
                    ix = frg_ids[i, :] == frg_ids[i, prev[i]]
                    frg_ids[i, ix] = obj
                exists[order, curr, prev] = bond
                exists[order, prev, curr] = bond
                vals_rom = vals_max - exists.sum(dim=1)
                is_end |= (vals_rom == 0).all(dim=1)

            # The part of connecting
            src[:, -self.n_frags, 1:] = 0
            src[:, -self.n_frags, 0] = self.voc_trg.tk2ix['GO']
            is_end = torch.zeros(len(src)).bool().to(src.device)
            for step in range(self.n_grows + 1, self.voc_trg.max_len):
                data = src[:, :step, :]
                triu = tri_mask(data[:, :, 0])
                emb = self.emb_word(data[:, :, 3] + data[:, :, 0] * 4)
                emb += self.emb_site(data[:, :, 1] * self.n_grows + data[:, :, 2])
                dec = self.attn(emb.transpose(0, 1), attn_mask=triu)

                vals_rom = vals_max - exists.sum(dim=1)
                frgs_rom = torch.zeros(len(src), 8).long().to(src.device)
                for i in range(1, 8):
                    ix = frg_ids != i
                    rom = vals_rom.clone()
                    rom[ix] = 0
                    frgs_rom[:, i] = rom.sum(dim=1)
                is_end |= (vals_rom == 0).all(dim=1)
                is_end |= (frgs_rom != 0).sum(dim=1) <= 1
                mask = (vals_rom < 1) | (vals_max == 0)
                mask[is_end, 0] = False
                atom_emb = self.emb_word(blank * 4 + single)
                dec = self.rnn(atom_emb, dec[-1, :, :])
                out_prev = self.prj_loci(dec).softmax(dim=-1)
                prev = out_prev.masked_fill(mask, 0).multinomial(1).view(-1)

                same = frg_ids == frg_ids[order, prev].view(-1, 1)
                exist = exists[order, prev] != 0
                mask = (vals_rom < 1) | exist | (vals_max == 0) | same
                mask[is_end, 0] = False
                prev_emb = self.emb_loci(prev)
                dec = self.rnn(prev_emb, dec)
                out_curr = self.prj_loci(dec).softmax(dim=-1)
                curr = out_curr.masked_fill(mask, 0).multinomial(1).view(-1)

                src[:, step, 3] = single
                src[:, step, 2] = prev
                src[:, step, 1] = curr
                src[:, step, 0] = blank
                src[is_end, step, :] = 0

                for i in range(len(src)):
                    obj = frg_ids[i, curr[i]].clone()
                    ix = frg_ids[i, :] == frg_ids[i, prev[i]]
                    frg_ids[i, ix] = obj
                exists[order, src[:, step, 1], src[:, step, 2]] = src[:, step, 3]
                exists[order, src[:, step, 2], src[:, step, 1]] = src[:, step, 3]
            out = src
        return out
    
    def trainNet(self, loader, epoch, epochs):
        net = nn.DataParallel(self, device_ids=self.gpus)
        total_steps = len(loader)
        current_step = 0
        for src in tqdm(loader, desc='Iterating over training batches', leave=False):
            src = src.to(self.device)
            self.optim.zero_grad()
            loss = net(src, is_train=True)
            loss = sum([-l.mean() for l in loss])   
            loss.backward()
            self.optim.step()
            current_step += 1
            self.monitor.saveProgress(current_step, epoch, total_steps, epochs)
            self.monitor.savePerformanceInfo(current_step, epoch, loss.item())

        return loss.item()
                
    def validateNet(self, loader, evaluator=None, no_multifrag_smiles=True):

        valid_metrics = {}
        
        net = nn.DataParallel(self, device_ids=self.gpus)
        pbar = tqdm(loader, desc='Iterating over validation batches', leave=False)
        smiles, frags = self.sample(pbar)
        scores = self.evaluate(smiles, frags, evaluator=evaluator, no_multifrag_smiles=no_multifrag_smiles)
        valid_metrics['valid_ratio'] = scores.VALID.mean() 
        valid_metrics['accurate_ratio'] = scores.DESIRE.mean()
                
        with torch.no_grad():
            valid_metrics['loss_valid'] = sum( [ sum([-l.float().mean().item() for l in net(src, is_train=True)]) for src in loader ] )
                
        scores['Smiles'] = smiles
        scores['Frags'] = frags
        # smiles_scores = []
        # for idx, smile in enumerate(smiles):
        #     smiles_scores.append((smile, scores.VALID[idx], scores.DESIRE[idx], frags[idx]))
                
        return valid_metrics, scores
    
    def sample(self, loader, repeat=1, pbar=None):
        net = nn.DataParallel(self, device_ids=self.gpus)
        frags, smiles = [], []
        with torch.no_grad():
            for _ in range(repeat):                
                for i, src in enumerate(loader):
                    trg = net(src.to(self.device))
                    f, s = self.voc_trg.decode(trg)
                    frags += f
                    smiles += s
                        
        return smiles, frags


    def sample_smiles(self, input_smiles, num_samples=100, batch_size=32, n_proc=1, fragmenter=None, 
                      keep_frags=True, drop_duplicates=True, drop_invalid=True, progress=True, tqdm_kwargs={}):
        standardizer = Standardization(n_proc=n_proc)
        smiles = standardizer.apply(input_smiles)

        fragmenter = Fragmenter(4, 4, 'brics') if not fragmenter else fragmenter
        encoder = FragmentCorpusEncoder(
            fragmenter=fragmenter,
            encoder=GraphFragmentEncoder(
                self.voc_trg
            ),
            n_proc=n_proc
        )
        out_data = GraphFragDataSet(tempfile.NamedTemporaryFile().name)
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
                new_frags, new_smiles = self.voc_trg.decode(trg)
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

