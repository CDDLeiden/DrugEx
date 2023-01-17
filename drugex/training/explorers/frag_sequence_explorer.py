#!/usr/bin/env python
from copy import deepcopy

import random
import torch
from torch import nn
from torch.optim import Adam
from drugex import utils, DEFAULT_DEVICE, DEFAULT_GPUS
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from drugex.logs import logger
from drugex.training.interfaces import Explorer
from drugex.training.monitors import NullMonitor


class FragSequenceExplorer(Explorer):
    """
    `Explorer` to optimize a sequence-based fragment-using agent with the given `Environment`.
    """

    def __init__(self, agent, env=None, crover=None, mutate=None, batch_size=128, epsilon=0.1, sigma=0.0, repeat=1, n_samples=-1, optim=None, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS, no_multifrag_smiles=True):
        super(FragSequenceExplorer, self).__init__(agent, env, mutate, crover, batch_size, epsilon, sigma, n_samples, repeat, device=device, use_gpus=use_gpus)
        self.optim = utils.ScheduledOptim(
            Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9), 1.0, 512) if not optim else optim
        # self.optim = optim.Adam(self.parameters(), lr=1e-5)
        self.no_multifrag_smiles = no_multifrag_smiles

    def forward(self, src):
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

    def policy_gradient(self, loader, no_multifrag_smiles=True):
        net = nn.DataParallel(self.agent, device_ids=self.gpus)
        total_steps = len(loader)
        step_idx = 0
        for src, trg in tqdm(loader, desc='Iterating over validation batches', leave=False):
            src, trg = src.to(self.device), trg.to(self.device)
            self.optim.zero_grad()
            smiles = [self.agent.voc_trg.decode(s, is_tk=False) for s in trg]
            frags = [self.agent.voc_trg.decode(s, is_tk=False) for s in src]
            reward = self.env.getRewards(smiles, frags=frags)
            if self.no_multifrag_smiles:
                reward = [r if s.count('.') == 0 else [0] for s,r in zip(smiles, reward)]
            reward = torch.Tensor(reward).to(src.device)
            loss = net(src, trg) * reward
            loss = -loss.mean()
            self.monitor.saveProgress(step_idx, None, total_steps, None)
            self.monitor.savePerformanceInfo(step_idx, None, loss.item())
            loss.backward()
            self.optim.step()
            del loss
            step_idx += 1
            
    def sample_input(self, loader, is_test=False):
        
        """
        Samples n_samples molecule-fargement pairs from the original input loader. 
        
        Arguments:
            loader                   : torch dataloader
            is_test (bool), opt      : if true, reduced sample size and increased batch size
        Returns:
            loader                   
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

    def fit(self, train_loader, valid_loader=None, epochs=1000, patience=50, criteria='desired_ratio', min_epochs=100, monitor=None):
        self.monitor = monitor if monitor else NullMonitor()
        self.bestState = deepcopy(self.agent.state_dict())
        self.monitor.saveModel(self.agent)
        n_iters = 1 if self.crover is None else 10
        net = nn.DataParallel(self, device_ids=self.gpus)
        srcs, trgs = [], []
        for it in range(n_iters):
            for epoch in tqdm(range(epochs), desc='Fitting SMILES explorer'):
                epoch += 1

                if self.nSamples > 0:
                    if epoch == 1:
                        train_loader_original = train_loader
                        valid_loader_original = valid_loader
                    train_loader = self.sample_input(train_loader_original)
                    valid_loader = self.sample_input(valid_loader_original, is_test=True)

                for i, (ix, src) in enumerate(tqdm(train_loader, desc='Iterating over training batches', leave=False)):
                    with torch.no_grad():
                        # frag = data_loader.dataset.index[ix]
                        trg = net(src.to(self.device))
                        trgs.append(trg.detach().cpu())
                        srcs.append(src.detach().cpu())

                trgs = torch.cat(trgs, dim=0)
                srcs = torch.cat(srcs, dim=0)

                dataset = TensorDataset(srcs, trgs)
                loader = DataLoader(dataset, batch_size=self.batchSize, shuffle=True, drop_last=False)
                self.policy_gradient(loader, no_multifrag_smiles=self.no_multifrag_smiles)
                srcs, trgs = [], []

                # Evaluate model
                smiles, frags = self.agent.sample(valid_loader, self.repeat)
                scores = self.agent.evaluate(smiles, frags, evaluator=self.env, no_multifrag_smiles=self.no_multifrag_smiles)
                scores['Smiles'], scores['Frags'] = smiles, frags             

                # Save evaluate criteria and save best model
                self.saveBestState(scores, criteria, epoch, it)

                # Log performance and genearated compounds
                self.logPerformanceAndCompounds(epoch, epochs, scores)
        
                # Early stopping
                if (epoch >= min_epochs) and  (epoch - self.last_save > patience) : break
            
            if self.crover is not None:
                self.agent.load_state_dict(self.bestState)
                self.crover.load_state_dict(self.bestState)
            if it - self.last_iter > 1: break
        
        self.monitor.close()
        torch.cuda.empty_cache() 