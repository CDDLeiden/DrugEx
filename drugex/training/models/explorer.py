#!/usr/bin/env python
from abc import ABC, abstractmethod
from copy import deepcopy

import random
import torch
from torch import nn
from torch.optim import Adam
from drugex import utils, DEFAULT_DEVICE, DEFAULT_GPUS
import time
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

from drugex.logs import logger
from drugex.training.interfaces import Explorer
from drugex.training.monitors import NullMonitor


class GraphExplorer(Explorer):
    """
    Graph-based `Explorer` to optimize a  graph-based agent with the given `Environment`.
    """

    def __init__(self, agent, env, mutate=None, crover=None, batch_size=128, epsilon=0.1, sigma=0.0, repeat=1, n_samples=-1, optim=None, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS, no_multifrag_smiles=True):
        super(GraphExplorer, self).__init__(agent, env, mutate, crover, batch_size, epsilon, sigma, n_samples, repeat, device=device, use_gpus=use_gpus)
        self.voc_trg = agent.voc_trg
        self.bestState = None
        self.optim = utils.ScheduledOptim(
            Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9), 1.0, 512) if not optim else optim
        # self.optim = optim.Adam(self.parameters(), lr=1e-5)
        self.no_multifrag_smiles = no_multifrag_smiles

    def forward(self, src):
        rand = torch.rand(1)
        if self.epsilon < rand <= 0.5 and self.crover is not None:
            net = self.crover
        elif rand < self.epsilon and self.mutate is not None:
            net = self.mutate
        else:
            net = self.agent
        is_end = torch.zeros(len(src)).bool().to(src.device)
        exists = torch.zeros(len(src), net.n_grows, net.n_grows).long().to(src.device)
        vals_max = torch.zeros(len(src), net.n_grows).long().to(src.device)
        frg_ids = torch.zeros(len(src), net.n_grows).long().to(src.device)
        order = torch.LongTensor(range(len(src))).to(src.device)
        curr = torch.zeros(len(src)).long().to(src.device) - 1
        blank = torch.LongTensor(len(src)).to(src.device).fill_(net.voc_trg.tk2ix['*'])
        single = torch.ones(len(src)).long().to(src.device)
        voc_mask = net.voc_trg.masks.to(src.device)
        for step in range(1, net.n_grows):
            if is_end.all():
                src[:, step, :] = 0
                continue
            data = src[:, :step, :]
            triu = utils.tri_mask(data[:, :, 0])
            emb = net.emb_word(data[:, :, 3] + data[:, :, 0] * 4)
            emb += net.emb_site(data[:, :, 1] * net.n_grows + data[:, :, 2])
            dec = net.attn(emb.transpose(0, 1), attn_mask=triu)
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
            out_atom = net.prj_atom(dec).softmax(dim=-1)
            atom = out_atom.masked_fill(mask, 0).multinomial(1).view(-1)
            src[grow, step, 0] = atom[grow]
            atom = src[:, step, 0]
            is_end |= (atom == 0) & grow
            num = (vals_max > 0).sum(dim=1)
            vals_max[order, num] = voc_mask[atom]
            vals_rom = vals_max - exists.sum(dim=1)

            bud = atom != net.voc_trg.tk2ix['*']
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

            atom_emb = net.emb_atom(atom)
            dec = net.rnn(atom_emb, dec)
            out_bond = net.prj_bond(dec).softmax(dim=-1)
            try:
                bond = out_bond.masked_fill(mask, 0).multinomial(1).view(-1)
            except Exception as e:
                raise
            src[grow, step, 3] = bond[grow]
            bond = src[:, step, 3]

            mask = (vals_max == 0) | exist | (vals_rom < bond.unsqueeze(-1))
            mask[order, curr] = True
            if step <= 2:
                mask[:, 0] = False
            mask[is_end, 0] = False
            mask[is_end, 1:] = True
            word_emb = net.emb_word(atom * 4 + bond)
            dec = net.rnn(word_emb, dec)
            prev_out = net.prj_loci(dec).softmax(dim=-1)
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
        src[:, -net.n_frags, 1:] = 0
        src[:, -net.n_frags, 0] = net.voc_trg.tk2ix['GO']
        is_end = torch.zeros(len(src)).bool().to(src.device)
        for step in range(net.n_grows + 1, net.voc_trg.max_len):
            data = src[:, :step, :]
            triu = utils.tri_mask(data[:, :, 0])
            emb = net.emb_word(data[:, :, 3] + data[:, :, 0] * 4)
            emb += net.emb_site(data[:, :, 1] * net.n_grows + data[:, :, 2])
            dec = net.attn(emb.transpose(0, 1), attn_mask=triu)

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
            atom_emb = net.emb_word(blank * 4 + single)
            dec = net.rnn(atom_emb, dec[-1, :, :])
            out_prev = net.prj_loci(dec).softmax(dim=-1)
            prev = out_prev.masked_fill(mask, 0).multinomial(1).view(-1)

            same = frg_ids == frg_ids[order, prev].view(-1, 1)
            exist = exists[order, prev] != 0
            mask = (vals_rom < 1) | exist | (vals_max == 0) | same
            mask[is_end, 0] = False
            prev_emb = net.emb_loci(prev)
            dec = net.rnn(prev_emb, dec)
            out_curr = net.prj_loci(dec).softmax(dim=-1)
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
        return src

    def policy_gradient(self, loader, monitor=None):
        monitor = monitor if monitor else NullMonitor()
        net = nn.DataParallel(self.agent, device_ids=self.gpus)
        total_steps = len(loader)
        for step_idx, src in enumerate(loader):
            monitor.saveProgress(step_idx, None, total_steps, None)
            src = src.to(self.device)
            frags, smiles = self.voc_trg.decode(src)
            reward = self.env.getRewards(smiles, frags=frags)
            if self.no_multifrag_smiles:
                reward = [r if s.count('.') == 0 else [0] for s,r in zip(smiles, reward)]
            reward = torch.Tensor(reward).to(src.device)
            self.optim.zero_grad()
            loss = net(src, is_train=True)
            loss = sum(loss).squeeze(dim=-1) * reward
            loss = -loss.mean()
            loss.backward()
            monitor.savePerformanceInfo(step_idx, None, loss.item())
            self.optim.step()
            del loss
            
    def sample_input(self, loader, is_test=False):
        
        """
        Samples n_samples molecule-fargement pairs from the original input loader. 
        
        Arguments:
            loader                   : torch dataloader
            is_test (bool), opt      : if true, reduced sample size and increased batch size
        Returns:
            loader                   
        """

        encoded_pairs = torch.cat([batch for batch in loader], 0)
        n_pairs = encoded_pairs.shape[0]                
        n_samples = int(self.nSamples * 0.2) if is_test else self.nSamples     
        batch_size = self.batchSize * 10 if is_test else self.batchSize * 4
        
        if n_pairs > n_samples:
        
            logger.info('{} fragments-molecule pairs were sampled at random from original {} pairs for {}'.format(n_samples, n_pairs, 'validation' if is_test else 'training'))
            samples = encoded_pairs[torch.randint(n_pairs, (n_samples,))]
            loader = DataLoader(samples, batch_size=batch_size, drop_last=False, shuffle=True)
            
        return loader
        

    def fit(self, train_loader, valid_loader=None, epochs=1000, patience=50, monitor=None):
        monitor = monitor if monitor else NullMonitor()
        max_desired_ratio = 0
        self.bestState = deepcopy(self.agent.state_dict())
        monitor.saveModel(self.agent)
        last_it = -1
        n_iters = 1 if self.crover is None else 10
        net = nn.DataParallel(self, device_ids=self.gpus)
        trgs = []
        logger.info(' ')
        for it in range(n_iters):
            last_save = -1
            if n_iters > 1:
                logger.info('\n----------\nITERATION %d/%d\n----------' % (it, n_iters))
            for epoch in tqdm(range(epochs)):
                epoch += 1
                t0 = time.time()
                              
                if self.nSamples > 0:
                    if epoch == 1:
                        train_loader_original = train_loader
                        valid_loader_original = valid_loader
                    train_loader = self.sample_input(train_loader_original)
                    valid_loader = self.sample_input(valid_loader_original, is_test=True)

                for i, src in enumerate(tqdm(train_loader, desc='Batch')):
                    # trgs.append(src.detach().cpu())
                    with torch.no_grad():
                        trg = net(src.to(self.device))
                        trgs.append(trg.detach().cpu())
                trgs = torch.cat(trgs, dim=0)
                loader = DataLoader(trgs, batch_size=self.batchSize, shuffle=True, drop_last=False)
                self.policy_gradient(loader, monitor=monitor)
                trgs = []

                frags, smiles, scores = self.agent.evaluate(valid_loader, repeat=self.repeat, method=self.env, no_multifrag_smiles=self.no_multifrag_smiles)
                desired_ratio = scores.DESIRE.sum() / len(smiles)
                mean_score = scores[self.env.getScorerKeys()].values.mean()
                valid_ratio = scores.VALID.sum() / len(smiles)
                unique_ratio = len(set(smiles)) / len(smiles)

                t1 = time.time()
                logger.info(f"Epoch: {epoch}  Score: {mean_score:.4f} Valid: {valid_ratio:.4f} Desire: {desired_ratio:.4f} Unique: {unique_ratio:.4f} Time: {t1-t0:.1f}s")   
        
                if max_desired_ratio < desired_ratio: 
                    monitor.saveModel(self.agent)
                    self.bestState = deepcopy(self.agent.state_dict())
                    max_desired_ratio = desired_ratio
                    last_save = epoch
                    last_it = it
                    logger.info(f"Model saved at epoch {epoch}")

                smiles_scores = []
                smiles_scores_key = ['Smiles'] + list(scores.columns) + ['Frag']
                for i, smile in enumerate(smiles):
                    smiles_scores.append((smile, *scores.values[i], frags[i]))

                monitor.savePerformanceInfo(None, epoch, None, score=mean_score, valid_ratio=valid_ratio, desire_ratio=desired_ratio, unique_ratio=unique_ratio, smiles_scores=smiles_scores, smiles_scores_key=smiles_scores_key)
                monitor.saveProgress(None, epoch, None, epochs)
                monitor.endStep(None, epoch)

                if epoch - last_save > patience: break

            if self.crover is not None:
                self.agent.load_state_dict(self.bestState)
                self.crover.load_state_dict(self.bestState)
            if it - last_it > 1: break

        torch.cuda.empty_cache()
        monitor.close()


class SmilesExplorer(Explorer):
    """
    Smiles-based `Explorer` to optimize a  graph-based agent with the given `Environment`.
    """

    def __init__(self, agent, env=None, crover=None, mutate=None, batch_size=128, epsilon=0.1, sigma=0.0, repeat=1, n_samples=-1, optim=None, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS, no_multifrag_smiles=True):
        super(SmilesExplorer, self).__init__(agent, env, mutate, crover, batch_size, epsilon, sigma, n_samples, repeat, device=device, use_gpus=use_gpus)
        self.optim = utils.ScheduledOptim(
            Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9), 1.0, 512) if not optim else optim
        self.bestState = None
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

    def policy_gradient(self, loader, no_multifrag_smiles=True, monitor=None):
        monitor = monitor if monitor else NullMonitor()
        net = nn.DataParallel(self.agent, device_ids=self.gpus)
        total_steps = len(loader)
        step_idx = 0
        for src, trg in loader:
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
            monitor.saveProgress(step_idx, None, total_steps, None)
            monitor.savePerformanceInfo(step_idx, None, loss.item())
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

    def fit(self, train_loader, valid_loader=None, epochs=1000, patience=50, monitor=None):
        self.bestState = deepcopy(self.agent.state_dict())
        monitor.saveModel(self.agent)
        max_desired_ratio = 0
        last_it = -1
        n_iters = 1 if self.crover is None else 10
        net = nn.DataParallel(self, device_ids=self.gpus)
        srcs, trgs = [], []
        for it in range(n_iters):
            last_save = -1
            for epoch in tqdm(range(epochs), desc='Epoch'):
                epoch += 1
                t0 = time.time()

                if self.nSamples > 0:
                    if epoch == 1:
                        train_loader_original = train_loader
                        valid_loader_original = valid_loader
                    train_loader = self.sample_input(train_loader_original)
                    valid_loader = self.sample_input(valid_loader_original, is_test=True)

                for i, (ix, src) in enumerate(tqdm(train_loader, desc='Batch')):
                    with torch.no_grad():
                        # frag = data_loader.dataset.index[ix]
                        trg = net(src.to(self.device))
                        trgs.append(trg.detach().cpu())
                        srcs.append(src.detach().cpu())

                trgs = torch.cat(trgs, dim=0)
                srcs = torch.cat(srcs, dim=0)

                dataset = TensorDataset(srcs, trgs)
                loader = DataLoader(dataset, batch_size=self.batchSize, shuffle=True, drop_last=False)
                self.policy_gradient(loader, no_multifrag_smiles=self.no_multifrag_smiles, monitor=monitor)
                srcs, trgs = [], []

                frags, smiles, scores = self.agent.evaluate(valid_loader, repeat=self.repeat, method=self.env, no_multifrag_smiles=self.no_multifrag_smiles)
                desired_ratio = scores.DESIRE.sum() / len(smiles)
                mean_score = scores[self.env.getScorerKeys()].values.mean()
                valid_ratio = scores.VALID.sum() / len(smiles)
                unique_ratio = len(set(smiles)) / len(smiles)

                t1 = time.time()
                logger.info(f"Epoch: {epoch}  Score: {mean_score:.4f} Valid: {valid_ratio:.4f} Desire: {desired_ratio:.4f} Unique: {unique_ratio:.4f} Time: {t1-t0:.1f}s")   

                smiles_scores = []
                smiles_scores_key = ['Smiles'] + list(scores.columns) + ['Frag']
                for i, smile in enumerate(smiles):
                    smiles_scores.append((smile, *scores.values[i], frags[i]))
        
                if max_desired_ratio < desired_ratio:
                    monitor.saveModel(self.agent)
                    self.bestState = deepcopy(self.agent.state_dict())
                    max_desired_ratio = desired_ratio
                    last_save = epoch
                    last_it = it
                    logger.info(f"Model saved at epoch {epoch}")

                monitor.savePerformanceInfo(None, epoch, None, score=mean_score, valid_ratio=valid_ratio, desire_ratio=desired_ratio, unique_ratio=unique_ratio, smiles_scores=smiles_scores, smiles_scores_key=smiles_scores_key)
                monitor.saveProgress(None, epoch, None, epochs)
                monitor.endStep(None, epoch)

                if epoch - last_save > patience: break
                
                if self.crover is not None:
                    self.agent.load_state_dict(self.bestState)
                    self.crover.load_state_dict(self.bestState)
            if it - last_it > 1: break
        
        monitor.close()
        torch.cuda.empty_cache()


class PGLearner(Explorer, ABC):
    """ Reinforcement learning framework with policy gradient. This class is the base structure for the
        drugex v1 and v2 policy gradient-based  deep reinforcement learning models.
 
    Arguments:
 
        agent (models.Generator): The agent which generates the desired molecules
 
        env (utils.Env): The environment which provides the reward and judge
                                 if the generated molecule is valid and desired.
 
        prior: The auxiliary model which is defined differently in each methods.
    """
    def __init__(self, agent, env=None, mutate=None, crover=None, memory=None, mean_func='geometric', batch_size=128, epsilon=1e-3,
                 sigma=0.0, repeat=1, n_samples=-1, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS):
        super().__init__(agent, env, mutate, crover, batch_size, epsilon, sigma, n_samples, repeat, device=device, use_gpus=use_gpus)
        self.replay = 10
        self.n_samples = 128  # * 8
        self.penalty = 0
        self.out = None
        self.memory = memory
        # mean_func: which function to use for averaging: 'arithmetic' or 'geometric'
        self.mean_func = mean_func
 
    @abstractmethod
    def policy_gradient(self, smiles=None, seqs=None, memory=None):
        pass
 
    def fit(self, train_loader, valid_loader=None, monitor=None, epochs=1000, no_multifrag_smiles=True):
        best = 0
        last_save = 0
        log = open(self.out + '.log', 'w')
        for epoch in range(1000):
            logger.info('\n----------\nEPOCH %d\n----------' % epoch)
            self.policy_gradient()
            smiles, scores = self.agent.evaluate(self.n_samples, method=self.env, drop_duplicates=True, no_multifrag_smiles=no_multifrag_smiles)
 
            desire = (scores.DESIRE).sum() / self.n_samples
            score = scores[self.env.getScorerKeys()].values.mean()
            valid = scores.VALID.mean()
 
            if best <= score:
                torch.save(self.agent.state_dict(), self.out + '.pkg')
                best = score
                last_save = epoch
 
            logger.info("Epoch: %d average: %.4f valid: %.4f desired: %.4f" %
                  (epoch, score, valid, desire), file=log)
            for i, smile in enumerate(smiles):
                score = "\t".join(['%0.3f' % s for s in scores.values[i]])
                print('%s\t%s' % (score, smile), file=log)
            if epoch - last_save > 50:
                break
        for param_group in self.agent.optim.param_groups:
            param_group['lr'] *= (1 - 0.01)
        log.close()
 
 
class SmilesExplorerNoFrag(PGLearner):
    """ DrugEx algorithm (version 2.0)
 
    Reference: Liu, X., Ye, K., van Vlijmen, H.W.T. et al. DrugEx v2: De Novo Design of Drug Molecule by
               Pareto-based Multi-Objective Reinforcement Learning in Polypharmacology.
               J Cheminform (2021). https://doi.org/10.1186/s13321-019-0355-6
 
    Arguments:
 
        agent (models.Generator): The agent network which is constructed by deep learning model
                                   and generates the desired molecules.
 
        env (utils.Env): The environment which provides the reward and judge
                                 if the genrated molecule is valid and desired.
 
        mutate (models.Generator): The pre-trained network which is constructed by deep learning model
                                   and ensure the agent to explore the approriate chemical space.
    """
    def __init__(self, agent, env, mutate=None, crover=None, mean_func='geometric', memory=None, batch_size=128, epsilon=0.1, sigma=0.0, repeat=1, n_samples=-1, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS):
        super(SmilesExplorerNoFrag, self).__init__(agent, env, mutate, crover, memory=memory, mean_func=mean_func, batch_size=batch_size, epsilon=epsilon, sigma=sigma, repeat=repeat, n_samples=n_samples, device=device, use_gpus=use_gpus)
        self.bestState = None
 
    def forward(self, crover=None, memory=None, epsilon=None):
        seqs = []
        #start = time.time()
        for _ in range(self.replay):
            seq = self.agent.evolve(self.batchSize, epsilon=epsilon, crover=crover, mutate=self.mutate)
            seqs.append(seq)
        #t1 = time.time()
        seqs = torch.cat(seqs, dim=0)
        if memory is not None:
            mems = [memory, seqs]
            seqs = torch.cat(mems)
        smiles = np.array([self.agent.voc.decode(s, is_tk = False) for s in seqs])
        # smiles = np.array(utils.canonicalize_list(smiles))
        ix = utils.unique(np.array([[s] for s in smiles]))
        smiles = smiles[ix]
        seqs = seqs[torch.LongTensor(ix).to(self.device)]
        return smiles, seqs
   
    def policy_gradient(self, smiles=None, seqs=None, memory=None, progress=None):
        # function need to get smiles
        scores = self.env.getRewards(smiles, frags=None)
        if memory is not None:
            scores[:len(memory), 0] = 1
            ix = scores[:, 0].argsort()[-self.batchSize * 4:]
            seqs, scores = seqs[ix, :], scores[ix, :]
        #t2 = time.time()
        ds = TensorDataset(seqs, torch.Tensor(scores).to(self.device))
        loader = DataLoader(ds, batch_size=self.n_samples, shuffle=True)
 
        # updating loss is done in rnn.py
        self.agent.PGLoss(loader, progress=progress)
        #t3 = time.time()
        #print(t1 - start, t2-t1, t3-t2)
 
    def fit(self, train_loader, valid_loader=None, monitor=None, epochs=1000, patience=50, no_multifrag_smiles=True):
        monitor.saveModel(self)
        self.bestState = deepcopy(self.agent.state_dict())
        max_desired_ratio = 0
        last_smiles = []
        last_scores = []
        last_save = -1
        ## add self.epoch
        for epoch in range(epochs):
            t0 = time.time()
            epoch += 1
            if epoch % 50 == 0 or epoch == 1: logger.info('\n----------\nEPOCH %d\n----------' % epoch)
            if epoch < patience and self.memory is not None:
                smiles, seqs = self.forward(crover=None, memory=self.memory, epsilon=1e-1)
                self.policy_gradient(smiles, seqs, memory=self.memory, progress=monitor)
            else:
                smiles, seqs = self.forward(crover=self.crover, epsilon=self.epsilon)
                self.policy_gradient(smiles, seqs, progress=monitor)
            smiles, scores = self.agent.evaluate(self.n_samples, method=self.env, drop_duplicates=True, no_multifrag_smiles=True)
 
            desired_ratio = (scores.DESIRE).sum() / self.n_samples
            valid_ratio = scores.VALID.sum() / self.n_samples
            unique_ratio = len(set(smiles)) / len(smiles)

            if self.mean_func == 'arithmetic':
                mean_score = scores[self.env.getScorerKeys()].values.sum() / self.n_samples / len(self.env.getScorerKeys())
            else:
                mean_score = scores[self.env.getScorerKeys()].values.prod(axis=1) ** (1.0 / len(self.env.getScorerKeys()))
                mean_score = mean_score.sum() / self.n_samples

            t1 = time.time()
            logger.info(f"Epoch: {epoch}  Score: {mean_score:.4f} Valid: {valid_ratio:.4f} Desire: {desired_ratio:.4f} Unique: {unique_ratio:.4f} Time: {t1-t0:.1f}s") 

            smiles_scores = []
            smiles_scores_key = ['Smiles'] + list(scores.columns)
            for i, smile in enumerate(smiles):
                smiles_scores.append((smile, *scores.values[i]))
 
            scores['Smiles'] = smiles
            monitor.savePerformanceInfo(None, epoch, None, score=mean_score, valid_ratio=valid_ratio, desire=desired_ratio, unique_ratio=unique_ratio, smiles_scores=smiles_scores, smiles_scores_key=smiles_scores_key)
            
            if max_desired_ratio < desired_ratio:
                monitor.saveModel(self)
                self.bestState = deepcopy(self.agent.state_dict())
                max_desired_ratio = desired_ratio
                last_save = epoch
                logger.info(f"Model saved at epoch {epoch}")
 
            if epoch % patience == 0 and epoch != 0:
                # Every nth epoch reset the agent and the crover networks to the best state
                for i, smile in enumerate(last_smiles):
                    score = "\t".join(['%.3f' % s for s in last_scores.drop(columns=['Smiles']).values[i]])
                    logger.info('%s\t%s' % (score, smile))
                self.agent.load_state_dict(self.bestState)
                self.crover.load_state_dict(self.bestState)
                logger.info('Resetting agent and crover to best state at epoch %d' % last_save)
            monitor.saveProgress(None, epoch, None, epochs)
            monitor.endStep(None, epoch)
    
            if epoch - last_save > patience: break
        
        logger.info('End time reinforcement learning: %s \n' % time.strftime('%d-%m-%y %H:%M:%S', time.localtime()))
        monitor.close()