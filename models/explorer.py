#!/usr/bin/env python
import torch
from torch import nn
from torch import optim
import utils
import time
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np


class GraphExplorer(nn.Module):
    def __init__(self, agent, crover=None, mutate=None, epsilon=1e-2, repeat=1):
        super(GraphExplorer, self).__init__()
        self.voc_trg = agent.voc_trg
        self.agent = agent
        self.crover = None if crover is None else crover
        self.mutate = None if mutate is None else mutate
        self.epsilon = epsilon
        self.repeat = repeat
        self.optim = utils.ScheduledOptim(
            optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9), 1.0, 512)
        # self.optim = optim.Adam(self.parameters(), lr=1e-5)

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
        net = nn.DataParallel(self.agent, device_ids=utils.devices)
        for src in loader:
            src = src.to(utils.dev)
            frags, smiles = self.voc_trg.decode(src)
            reward = self.env.calc_reward(smiles, self.scheme, frags=frags)
            reward = torch.Tensor(reward).to(src.device)

            self.optim.zero_grad()
            loss = net(src, is_train=True)
            loss = sum(loss).squeeze(dim=-1) * reward
            loss = -loss.mean()
            loss.backward()
            self.optim.step()
            del loss

    def fit(self, data_loader, test_loader=None, epochs=1000):
        best_score = 0
        log = open(self.out + '.log', 'w')
        last_it = -1
        n_iters = 1 if self.crover is None else 10
        net = nn.DataParallel(self, device_ids=utils.devices)
        trgs = []
        for it in range(n_iters):
            last_save = -1
            print('\n----------\nITERATION %d/ %d\n----------' % (it, n_iters))
            for epoch in tqdm(range(epochs)):
                t0 = time.time()
                t00 = t0
                #for i, src in enumerate(tqdm(data_loader)):
                for i, src in enumerate(data_loader):
                    # trgs.append(src.detach().cpu())
                    with torch.no_grad():
                        trg = net(src.to(utils.dev))
                        trgs.append(trg.detach().cpu())
                    # if len(trgs) < 10 : continue
                #t1 = time.time()
                #print('Net time:', t1-t0)
                #t0 = t1

                trgs = torch.cat(trgs, dim=0)
                loader = DataLoader(trgs, batch_size=self.batch_size, shuffle=True, drop_last=True)
                self.policy_gradient(loader)
                trgs = []
                #t1 = time.time()
                #print('PG time:', t1-t0)
                #t0 = t1

                frags, smiles, scores = self.agent.evaluate(test_loader, repeat=self.repeat, method=self.env)
                desire = scores.DESIRE.sum() / len(smiles)
                score = scores[self.env.keys].values.mean()
                valid = scores.VALID.mean()
                #t1 = time.time()
                #print('Eval time:', t1-t0)
                #t0 = t1

                t1 = time.time()
                log.write("Iteration: %s Epoch: %d Av. Clipped Score: %.4f Valid: %.4f Desire: %.4f Time: %.1fs\n" %
                          (it, epoch, score, valid, desire, t1 - t0))
                if best_score < desire:
                    torch.save(self.agent.state_dict(), self.out + '.pkg')
                    best_score = desire
                    last_save = epoch
                    last_it = it
                if epoch - last_save > 100: break

                for i, smile in enumerate(smiles):
                    score = "\t".join(['%.3f' % s for s in scores.values[i]])
                    log.write('%s\t%s\t%s\n' % (score, frags[i], smile))
                    
                t1 = time.time()
                #print('Log time:', t1-t0)
                #print('Epoch time:', t1-t00)
            if self.crover is not None:
                self.agent.load_state_dict(torch.load(self.out + '.pkg'))
                self.crover.load_state_dict(torch.load(self.out + '.pkg'))
            if it - last_it > 1: break
        log.close()


class SmilesExplorer(nn.Module):
    def __init__(self, agent, crover=None, mutate=None, epsilon=1e-2, repeat=1):
        super(SmilesExplorer, self).__init__()
        self.agent = agent
        self.crover = crover
        self.mutate = mutate
        self.epsilon = epsilon
        self.repeat = repeat
        self.optim = utils.ScheduledOptim(
            optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9), 1.0, 512)
        # self.optim = optim.Adam(self.parameters(), lr=1e-5)

    def forward(self, src):
        seq_len = self.agent.voc_trg.max_len + self.agent.voc_trg.max_len
        out = torch.zeros(len(src), seq_len).long().to(utils.dev)
        out[:, :src.size(1)] = src
        is_end = torch.zeros(len(src)).bool().to(utils.dev)

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

    def policy_gradient(self, loader):
        net = nn.DataParallel(self.agent, device_ids=utils.devices)
        for src, trg in loader:
            src, trg = src.to(utils.dev), trg.to(utils.dev)
            self.optim.zero_grad()
            smiles = [self.agent.voc_trg.decode(s, is_tk=False) for s in trg]
            frags = [self.agent.voc_trg.decode(s, is_tk=False) for s in src]
            reward = self.env.calc_reward(smiles, self.scheme, frags=frags)
            reward = torch.Tensor(reward).to(src.device)
            loss = net(src, trg) * reward
            loss = -loss.mean()
            loss.backward()
            self.optim.step()
            del loss

    def fit(self, data_loader, test_loader=None, epochs=1000):
        best_score = 0
        log = open(self.out + '.log', 'w')
        last_it = -1
        n_iters = 1 if self.crover is None else 10
        net = nn.DataParallel(self, device_ids=utils.devices)
        srcs, trgs = [], []
        for it in range(n_iters):
            last_save = -1
            for epoch in range(epochs):
                t0 = time.time()

                print('\n----------\nITERATION %d\nEPOCH %d\n----------' % (it, epoch))
                for i, (ix, src) in enumerate(tqdm(data_loader)):
                    with torch.no_grad():
                        frag = data_loader.dataset.index[ix]
                        trg = net(src.to(utils.dev))
                        trgs.append(trg.detach().cpu())
                        srcs.append(src.detach().cpu())

                trgs = torch.cat(trgs, dim=0)
                srcs = torch.cat(srcs, dim=0)

                dataset = TensorDataset(srcs, trgs)
                loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
                self.policy_gradient(loader)
                srcs, trgs = [], []

                frags, smiles, scores = self.agent.evaluate(test_loader, repeat=self.repeat, method=self.env)
                desire = scores.DESIRE.sum() / len(smiles)
                score = scores[self.env.keys].values.mean()
                valid = scores.VALID.mean()

                t1 = time.time()
                log.write("Iteration: %s Epoch: %d average: %.4f valid: %.4f desire: %.4f time: %.1fs\n" %
                          (it, epoch, score, valid, desire, t1 - t0))
                for i, smile in enumerate(smiles):
                    score = "\t".join(['%.3f' % s for s in scores.values[i]])
                    log.write('%s\t%s\t%s\n' % (score, frags[i], smile))

                if best_score < desire:
                    torch.save(self.agent.state_dict(), self.out + '.pkg')
                    best_score = desire
                    last_save = epoch
                    last_it = it
                if epoch - last_save > 100: break
                if self.crover is not None:
                    self.agent.load_state_dict(torch.load(self.out + '.pkg'))
                    self.crover.load_state_dict(torch.load(self.out + '.pkg'))
            if it - last_it > 1: break
        log.close()


class PGLearner(object):
    """ Reinforcement learning framework with policy gradient. This class is the base structure for the
        drugex v1 and v2 policy gradient-based  deep reinforcement learning models.
 
    Arguments:
 
        agent (models.Generator): The agent which generates the desired molecules
 
        env (utils.Env): The environment which provides the reward and judge
                                 if the generated molecule is valid and desired.
 
        prior: The auxiliary model which is defined differently in each methods.
    """
    def __init__(self, agent, prior=None, memory=None, mean_func='geometric'):
        self.replay = 10
        self.agent = agent
        self.prior = prior
        self.batch_size = 64  # * 4
        self.n_samples = 128  # * 8
        self.epsilon = 1e-3
        self.penalty = 0
        self.scheme = 'PR'
        self.out = None
        self.memory = memory
        # mean_func: which function to use for averaging: 'arithmetic' or 'geometric'
        self.mean_func = mean_func
 
    def policy_gradient(self):
        pass
 
    def fit(self):
        best = 0
        last_save = 0
        log = open(self.out + '.log', 'w')
        for epoch in range(1000):
            print('\n----------\nEPOCH %d\n----------' % epoch)
            self.policy_gradient()
            smiles, scores = self.agent.evaluate(self.n_samples, method=self.env, drop_duplicates=True)
 
            desire = (scores.DESIRE).sum() / self.n_samples
            score = scores[self.env.keys].values.mean()
            valid = scores.VALID.mean()
 
            if best <= score:
                torch.save(self.agent.state_dict(), self.out + '.pkg')
                best = score
                last_save = epoch
 
            print("Epoch: %d average: %.4f valid: %.4f desired: %.4f" %
                  (epoch, score, valid, desire), file=log)
            for i, smile in enumerate(smiles):
                score = "\t".join(['%0.3f' % s for s in scores.values[i]])
                print('%s\t%s' % (score, smile), file=log)
            if epoch - last_save > 100:
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
 
        prior (models.Generator): The pre-trained network which is constructed by deep learning model
                                   and ensure the agent to explore the approriate chemical space.
    """
    def __init__(self, agent, prior=None, crover=None, mean_func='geometric', memory=None):
        super(SmilesExplorerNoFrag, self).__init__(agent, prior, mean_func=mean_func, memory=memory)
        self.crover = crover
 
    def forward(self, crover=None, memory=None, epsilon=None):
        seqs = []
        #start = time.time()
        for _ in range(self.replay):
            seq = self.agent.evolve(self.batch_size, epsilon=epsilon, crover=crover, mutate=self.prior)
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
        seqs = seqs[torch.LongTensor(ix).to(utils.dev)]
        return smiles, seqs
   
    def policy_gradient(self, smiles, seqs, memory=None):
        # function need to get smiles
        scores = self.env.calc_reward(smiles, self.scheme, frags=None)
        if memory is not None:
            scores[:len(memory), 0] = 1
            ix = scores[:, 0].argsort()[-self.batch_size * 4:]
            seqs, scores = seqs[ix, :], scores[ix, :]
        #t2 = time.time()
        ds = TensorDataset(seqs, torch.Tensor(scores).to(utils.dev))
        loader = DataLoader(ds, batch_size=self.n_samples, shuffle=True)
 
        # updating loss is done in rnn.py
        self.agent.PGLoss(loader)
        #t3 = time.time()
        #print(t1 - start, t2-t1, t3-t2)
 
    def fit(self, epochs):
        best = 0
        log = open(self.out + '.log', 'a')
        last_smiles = []
        last_scores = []
        interval = 250
        last_save = -1
        ## add self.epoch
        for epoch in range(epochs):
            if epoch % 50 == 0: print('\n----------\nEPOCH %d\n----------' % epoch)
            if epoch < interval and self.memory is not None:
                smiles, seqs = self.forward(crover=None, memory=self.memory, epsilon=1e-1)
                self.policy_gradient(smiles, seqs, memory=self.memory)
            else:
                smiles, seqs = self.forward(crover=self.crover, epsilon=self.epsilon)
                self.policy_gradient(smiles, seqs)
            smiles, scores = self.agent.evaluate(self.n_samples, method=self.env, drop_duplicates=True)
 
            desire = (scores.DESIRE).sum() / self.n_samples
            if self.mean_func == 'arithmetic':
                score = scores[self.env.keys].values.sum() / self.n_samples / len(self.env.keys)
            else:
                score = scores[self.env.keys].values.prod(axis=1) ** (1.0 / len(self.env.keys))
                score = score.sum() / self.n_samples
            valid = scores.VALID.sum() / self.n_samples
 
            print("Epoch: %d average: %.4f valid: %.4f desired: %.4f" %
                  (epoch, score, valid, desire), file=log)
            if best < score:
                torch.save(self.agent.state_dict(), self.out + '.pkg')
                best = score
                last_smiles = smiles
                last_scores = scores
                last_save = epoch
 
            if epoch % interval == 0 and epoch != 0:
                for i, smile in enumerate(last_smiles):
                    score = "\t".join(['%.3f' % s for s in last_scores.values[i]])
                    print('%s\t%s' % (score, smile), file=log)
                self.agent.load_state_dict(torch.load(self.out + '.pkg'))
                self.crover.load_state_dict(torch.load(self.out + '.pkg'))
            if epoch - last_save > interval: break
        print('End time reinforcement learning: %s \n' % time.strftime('%d-%m-%y %H:%M:%S', time.localtime()), file=log)
        log.close()
