#!/usr/bin/env python
from copy import deepcopy

import torch
from torch import nn
from torch.optim import Adam
from drugex import utils, DEFAULT_DEVICE, DEFAULT_GPUS
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from drugex.logs import logger
from drugex.training.interfaces import Explorer
from drugex.training.monitors import NullMonitor


class FragGraphExplorer(Explorer):
    """
    `Explorer` to optimize a graph-based fragment-using agent with the given `Environment`.
    """

    def __init__(self, agent, env, mutate=None, crover=None, batch_size=128, epsilon=0.1, sigma=0.0, repeat=1, n_samples=-1, optim=None, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS, no_multifrag_smiles=True):
        super(FragGraphExplorer, self).__init__(agent, env, mutate, crover, batch_size, epsilon, sigma, n_samples, repeat, device=device, use_gpus=use_gpus)
        self.voc_trg = agent.voc_trg
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

    def policy_gradient(self, loader):
        net = nn.DataParallel(self.agent, device_ids=self.gpus)
        total_steps = len(loader)
        for step_idx, src in enumerate(tqdm(loader, desc='Iterating over validation batches', leave=False)):
            self.monitor.saveProgress(step_idx, None, total_steps, None)
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
            self.monitor.savePerformanceInfo(step_idx, None, loss.item())
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
        

    def fit(self, train_loader, valid_loader=None, epochs=1000, patience=50, criteria='desired_ratio', min_epochs=100, monitor=None):
        
        self.monitor = monitor if monitor else NullMonitor()
        self.bestState = deepcopy(self.agent.state_dict())
        self.monitor.saveModel(self.agent)
        # best_value = 0

        n_iters = 1 if self.crover is None else 10
        net = nn.DataParallel(self, device_ids=self.gpus)
        trgs = []
        logger.info(' ')
        
        for it in range(n_iters):
            if n_iters > 1:
                logger.info('\n----------\nITERATION %d/%d\n----------' % (it, n_iters))
            for epoch in tqdm(range(epochs), desc='Fitting graph explorer'):
                epoch += 1
                              
                if self.nSamples > 0:
                    if epoch == 1:
                        train_loader_original = train_loader
                        valid_loader_original = valid_loader
                    train_loader = self.sample_input(train_loader_original)
                    valid_loader = self.sample_input(valid_loader_original, is_test=True)

                for i, src in enumerate(tqdm(train_loader, desc='Iterating over training batches', leave=False)):
                    # trgs.append(src.detach().cpu())
                    with torch.no_grad():
                        trg = net(src.to(self.device))
                        trgs.append(trg.detach().cpu())
                trgs = torch.cat(trgs, dim=0)
                loader = DataLoader(trgs, batch_size=self.batchSize, shuffle=True, drop_last=False)
                self.policy_gradient(loader)
                trgs = []

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

        torch.cuda.empty_cache()
        self.monitor.close()