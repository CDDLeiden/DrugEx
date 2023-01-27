#!/usr/bin/env python
from copy import deepcopy

import random
import torch
from torch import nn
from torch.optim import Adam
from drugex import utils, DEFAULT_DEVICE, DEFAULT_GPUS
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from typing import Union, Tuple

from drugex.logs import logger
from drugex.training.explorers.interfaces import FragExplorer
from drugex.training.monitors import NullMonitor


class FragSequenceExplorer(FragExplorer):
    """
    `Explorer` to optimize a sequence-based fragment-using agent with the given `Environment`.
    """

    def __init__(self, agent, env=None, crover=None, mutate=None, batch_size=128, epsilon=0.1, beta=0.0, n_samples=-1, optim=None, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS, no_multifrag_smiles=True):
        super(FragSequenceExplorer, self).__init__(agent, env, mutate, crover, no_multifrag_smiles, batch_size, epsilon, beta, n_samples, device=device, use_gpus=use_gpus)
        """
        Parameters
        ----------
        
        agent : models.SequenceTransformer
            The agent network which is optimised to generates the desired molecules.
        env : drugex.trainining.interfaces.Environment
            The environment which provides the reward and judge if the genrated molecule is valid and desired.
        crover : models.SequenceTransformer
             The iteratively updated network which increases the exploitation of the chemical space.(Not used in this class)
        mutate : models.SequenceTransformer
            The pre-trained network which increases the exploration of the chemical space. 
        batch_size : int
            The batch size for the training.
        epsilon : float
            The probability of using the `mutate` network.
        beta : float
            The baseline for the reward.
        n_samples : int
            The number of molecules to be generated in each epoch.
        optim : torch.optim
            The optimizer to be used for the training.
        device : torch.device
            The device to be used for the training.
        use_gpus : tuple
            The GPUs to be used for the training.
        no_multifrag_smiles : bool
            If True, the molecules with multiple fragments are not considered as valid molecules.
        """
        
        self.optim = utils.ScheduledOptim(
            Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9), 1.0, 512) if not optim else optim

    def forward(self, src):
        """
        Forward pass of the agent.
        
        Parameters
        ----------
        src : torch.Tensor
            TODO: check the shape of the input tensor.
        
        Returns
        -------
        torch.Tensor
            TODO: check the shape of the input tensor.
        """
        seq_len = self.agent.voc_trg.max_len + self.agent.voc_trg.max_len
        out = torch.zeros(len(src), seq_len).long().to(self.device)
        out[:, :src.size(1)] = src
        is_end = torch.zeros(len(src)).bool().to(self.device)

        for step in range(self.agent.voc_trg.max_len):  # decode up to max length
            sub = out[:, :src.size(1) + step]
            key_mask = utils.pad_mask(sub, self.agent.pad_idx)
            atn_mask = utils.tri_mask(sub)
            rand = torch.rand(1)
            if self.epsilon < rand <= 0.5 and self.crover is not None:
                dec = self.crover.gpt2(sub.transpose(0, 1), key_mask=key_mask, atn_mask=atn_mask)
            elif rand < self.epsilon and self.mutate is not None:
                dec = self.mutate.gpt2(sub.transpose(0, 1), key_mask=key_mask, atn_mask=atn_mask)
            else:
                dec = self.agent.gpt2(sub.transpose(0, 1), key_mask=key_mask, atn_mask=atn_mask)
            proba = dec[-1,:, :].softmax(dim=-1)

            # sampling based on output probability distribution
            x = torch.multinomial(proba, 1).view(-1)

            x[is_end] = self.agent.voc_trg.tk2ix['_']
            is_end |= x == self.agent.voc_trg.tk2ix['EOS']
            out[:, src.size(1) + step] = x
            if is_end.all(): break
        return out[:, self.agent.voc_trg.max_len:].detach()
            
    def sample_input(self, loader, is_test=False):
        
        """
        Sample a batch of fragments-molecule pairs from the dataset.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            Data loader for original fragments-molecule pairs
        is_test : bool
            Whether to sample from the validation set or not
        
        Returns
        -------
        torch.utils.data.DataLoader
            Data loader for sampled fragments-molecule pairs
        """
        
        encoded_in = torch.cat([batch[0] for batch in loader], 0)
        encoded_out = torch.cat([batch[1] for batch in loader], 0)
        
        n_pairs = encoded_in.shape[0]                
        n_samples = int(self.nSamples * 0.2) if is_test else self.nSamples     
        batch_size = self.batchSize * 10 if is_test else self.batchSize * 4
                
        if n_pairs > n_samples:
        
            logger.info(f"{n_samples} fragments-molecule pairs were sampled at random from original {n_pairs} pairs for {'validation' if is_test else 'training'}")
            sample_idx = torch.tensor(random.sample([ i for i in range(n_pairs)], n_samples))
            samples = [ torch.index_select(encoded_in, 0, sample_idx), torch.index_select(encoded_out, 0, sample_idx) ]
            loader = DataLoader(samples, batch_size=batch_size, drop_last=False, shuffle=True)
            
        return loader

    def getBatchOutputs(self, net, src):
        
        """
        Outputs (frags, smiles) and loss of the agent for a batch of fragments-molecule pairs.

        Parameters
        ----------
        net : torch.nn.Module
            Agent
        src : torch.Tensor
            Fragments-molecule pairs
        
        Returns
        -------
        frags : list
            List of fragments
        smiles : list
            List of SMILES
        loss : torch.Tensor
            Loss of the agent
        """

        src, trg = src[0].to(self.device), src[1].to(self.device)
        frags = [self.agent.voc_trg.decode(s, is_tk=False) for s in src]
        smiles = [self.agent.voc_trg.decode(s, is_tk=False) for s in trg]
        loss = net(src, trg)

        return frags, smiles, loss

    def sampleEncodedPairsToLoader(self, net, loader):

        """
        Sample new fragments-molecule pairs from a data loader.

        Parameters
        ----------
        net : torch.nn.Module
            Agent
        loader : torch.utils.data.DataLoader
            Data loader for original fragments-molecule pairs

        Returns
        -------
        torch.utils.data.DataLoader
            Data loader for sampled fragments-molecule pairs
        """

        srcs, trgs = [], []
        for _, src in tqdm(loader, desc='Iterating over training batches', leave=False):
            with torch.no_grad():
                trg = net(src.to(self.device))
                trgs.append(trg.detach().cpu())
                srcs.append(src.detach().cpu())
        trgs = torch.cat(trgs, dim=0)
        srcs = torch.cat(srcs, dim=0)      

        return DataLoader(TensorDataset(srcs, trgs), batch_size=self.batchSize, shuffle=True, drop_last=False)