import tempfile
import torch

import torch.nn as nn

from torch import optim
from torch.nn.init import kaiming_normal_
from tqdm.auto import tqdm

from drugex import DEFAULT_DEVICE, DEFAULT_GPUS
from drugex.data.fragments import SequenceFragmentEncoder, FragmentCorpusEncoder
from drugex.data.datasets import SmilesFragDataSet
from drugex.molecules.converters.dummy_molecules import dummyMolsFromFragments
from drugex.training.generators.utils import PositionalEmbedding, PositionwiseFeedForward, SublayerConnection, pad_mask, tri_mask
from drugex.training.generators.interfaces import FragGenerator
from drugex.utils import ScheduledOptim


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


class SequenceTransformer(FragGenerator):
    """
    Sequence Transformer for molecule generation from fragments
    """
    def __init__(self, voc_trg, d_emb=512, d_model=512, n_head=8, d_inner=1024, n_layer=12, pad_idx=0, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS):
        super(SequenceTransformer, self).__init__(device=device, use_gpus=use_gpus)
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

        self.model_name = 'SequenceTransformer'

    # def init_states(self):
    #     """
    #     Initialize model parameters
        
    #     Notes:
    #     -----
    #     Xavier initialization for all parameters except for the embedding layer
    #     """
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)
    #     self.attachToGPUs(self.gpus)

    # def attachToGPUs(self, gpus):
    #     """
    #     Attach model to GPUs

    #     Parameters:
    #     ----------
    #     gpus: `tuple`
    #         A tuple of GPU ids to use
        
    #     Returns:
    #     -------
    #     None
    #     """
    #     self.gpus = gpus
    #     self.to(self.device)

    def forward(self, src, trg=None):
        """
        Forward pass of the model
        
        Parameters:
        ----------
        src: `torch.Tensor`
            TODO: check that the shape is correct
            Source tensor of shape [batch_size, 200] 
        trg: `torch.Tensor`
            Target tensor of shape [batch_size, 200]

        Returns:
        -------
        TODO: fill outputs
        """

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

    def trainNet(self, loader, epoch, epochs):
        """
        Train the model for one epoch
        
        Parameters:
        ----------
        loader: `torch.utils.data.DataLoader`
            A dataloader object to iterate over the training data
        epoch: `int`
            Current epoch number
        epochs: `int`
            Total number of epochs
        
        Returns:
        -------
        loss: `float`
            The loss value for the current epoch
        """

        net = nn.DataParallel(self, device_ids=self.gpus)
        total_steps = len(loader)
        current_step = 0
        for src, trg in tqdm(loader, desc='Iterating over training batches', leave=False):
            src, trg = src.to(self.device), trg.to(self.device)
            self.optim.zero_grad()
            loss = net(src, trg)
            loss = -loss.mean()     
            loss.backward()
            self.optim.step()
            current_step += 1
            self.monitor.saveProgress(current_step, epoch, total_steps, epochs, loss.item())

        return loss.item()

    def validateNet(self, loader, evaluator=None, no_multifrag_smiles=True, n_samples=None):
        """
        Validate the model
        
        Parameters:
        ----------
        loader: `torch.utils.data.DataLoader`
            A dataloader object to iterate over the validation data
        evaluator: `Evaluator`
            An evaluator object to evaluate the generated SMILES
        no_multifrag_smiles: `bool`
            If `True`, only single-fragment SMILES are considered valid
        
        Returns:
        -------
        valid_metrics: `dict`
            A dictionary containing the validation metrics
        scores: `pandas.DataFrame`
            DataFrame containing Smiles, frags and the scores for each SMILES

        Notes:
        -----
        The validation metrics are:
            - valid_ratio: ratio of valid SMILES
            - accurate_ratio: ratio of SMILES that are valid and have the desired fragments
            - loss_valid: loss on the validation set
         """
        
        valid_metrics = {}

        net = nn.DataParallel(self, device_ids=self.gpus)
        pbar = tqdm(loader, desc='Iterating over validation batches', leave=False)
        smiles, frags = self.sample(pbar)
        scores = self.evaluate(smiles, frags, evaluator=evaluator, no_multifrag_smiles=no_multifrag_smiles)
        scores['SMILES'] = smiles
        scores['Frags'] = frags
        valid_metrics['valid_ratio'] = scores.Valid.mean() 
        valid_metrics['accurate_ratio'] = scores.Accurate.mean()
                
        with torch.no_grad():
            valid_metrics['loss_valid'] = sum( [ sum([-l.mean().item() for l in net(src, trg)]) for src, trg in loader ] )
            
        return valid_metrics, scores
    
    def sample(self, loader):
        """
        Sample SMILES from the model
        
        Parameters:
        ----------
        loader: `torch.utils.data.DataLoader`
            A dataloader object to iterate over the input fragments 

        Returns:
        -------
        smiles: `list`
            A list of sampled SMILES
        frags: `list`
            A list of input fragments
        """
        net = nn.DataParallel(self, device_ids=self.gpus)
        frags, smiles = [], []
        with torch.no_grad():             
            for src, _ in loader:
                trg = net(src.to(self.device))
                smiles += [self.voc_trg.decode(s, is_tk=False) for s in trg]
                frags += [self.voc_trg.decode(s, is_tk=False) for s in src]

        return smiles, frags

    def loaderFromFrags(self, frags, batch_size=32, n_proc=1):
        """
        Encode the input fragments and create a dataloader object
        
        Parameters:
        ----------
        frags: `list`
            A list of input fragments (in SMILES format)
        batch_size: `int`
            Batch size for the dataloader
        n_proc: `int`
            Number of processes to use for encoding the fragments
        
        Returns:
        -------
        loader: `torch.utils.data.DataLoader`
            A dataloader object to iterate over the input fragments 
        """
        
        # Encode the input fragments
        encoder = FragmentCorpusEncoder(
            fragmenter=dummyMolsFromFragments(),
            encoder=SequenceFragmentEncoder(
                self.voc_trg
            ),
            n_proc=n_proc
        )
        out_data = SmilesFragDataSet(tempfile.NamedTemporaryFile().name)
        encoder.apply(frags, encodingCollectors=[out_data])
        loader = out_data.asDataLoader(batch_size, n_samples=batch_size)
        
        return loader
    def decodeLoaders(self, src, trg):
        new_smiles = [self.voc_trg.decode(s, is_tk=False) for s in trg]
        new_frags = [self.voc_trg.decode(s, is_tk=False) for s in src]
        return new_frags, new_smiles

    def iterLoader(self, loader):
        for _, src in loader:
            yield src

