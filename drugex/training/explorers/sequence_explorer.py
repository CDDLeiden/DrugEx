#!/usr/bin/env python
from copy import deepcopy

import random
import torch
from drugex import utils, DEFAULT_DEVICE, DEFAULT_GPUS
import time
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import numpy as np

from drugex.logs import logger
from drugex.training.explorers.interfaces import Explorer
from drugex.training.monitors import NullMonitor

 
class SequenceExplorer(Explorer):

    """ 
    `Explorer` to optimize a sequence-based agent (RNN) with the given `Environment`.
 
    Reference: Liu, X., Ye, K., van Vlijmen, H.W.T. et al. DrugEx v2: De Novo Design of Drug Molecule by
               Pareto-based Multi-Objective Reinforcement Learning in Polypharmacology.
               J Cheminform (2021). https://doi.org/10.1186/s13321-019-0355-6
 
    Parameters
    ----------
    agent: drugex.training.generators.SequenceRNN
        The agent network which is optimised to generates the desired molecules.
    env : drugex.training.interfaces.Environment
        The environment which provides the reward and judge if the genrated molecule is valid and desired.
    mutate : drugex.training.generators.SequenceRNN
        The pre-trained network which increases the exploration of the chemical space.
    crover : drugex.training.generators.SequenceRNN
        The iteratively updated network which increases the exploitation of the chemical space.
    batch_size : int
        The number of molecules generated in each iteration.
    epsilon : float
        The probability of using the `mutate` network to generate molecules.
    beta : float
        The baseline for the reward.
    n_samples : int
        The number of molecules to be generated in each epoch.
    device : torch.device
        The device to run the network.
    use_gpus : tuple
        The GPU ids to run the network.
    """

    def __init__(self, agent, env, mutate=None, crover=None, memory=None, batch_size=128, epsilon=0.1, beta=0.0, n_samples=-1, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS):
        super(SequenceExplorer, self).__init__(agent, env, mutate, crover, batch_size=batch_size, epsilon=epsilon, beta=beta, n_samples=n_samples, device=device, use_gpus=use_gpus)
        self.replay = 10
        self.n_samples = 128  # * 8
        self.penalty = 0
        self.out = None
        self.memory = memory
        self.optim = torch.optim.Adam(self.agent.parameters(), lr=1e-3)

    def forward(self, crover=None, memory=None, epsilon=None):
        """
        Generate molecules with the given `agent` network.
        
        Parameters
        ----------
        crover : drugex.training.generators.SequenceRNN
            The iteratively updated network which increases the exploitation of the chemical space.
        memory : torch.Tensor
            TODO: what is this?
        epsilon : float
            The probability of using the `mutate` network to generate molecules.

        Returns
        -------
        smiles : list
            The generated SMILES.
        seqs : torch.Tensor
            The generated encoded sequences.

        TODO: why is crover, memory and epsilon passed as parameters and not called from self?
        """
    
        seqs = []
        for _ in range(self.replay):
            seq = self.agent.evolve(self.batchSize, epsilon=epsilon, crover=crover, mutate=self.mutate)
            seqs.append(seq)
        seqs = torch.cat(seqs, dim=0)
        if memory is not None:
            mems = [memory, seqs]
            seqs = torch.cat(mems)
        smiles = np.array([self.agent.voc.decode(s, is_tk = False) for s in seqs])
        ix = utils.unique(np.array([[s] for s in smiles]))
        smiles = smiles[ix]
        seqs = seqs[torch.LongTensor(ix).to(self.device)]
        return smiles, seqs
   
    def policy_gradient(self, smiles=None, seqs=None, memory=None):
        """
        Policy gradient training.
 
        Novel molecules are scored by the environment.
        The policy gradient is calculated using the REINFORCE algorithm and the agent is updated.
        
        Parameters
        ----------
        smiles : list
            The generated SMILES.
        seqs : torch.Tensor
            The generated encoded sequences. 
        memory : torch.Tensor
            TODO: what is this?
        """

        # Calculate the reward from SMILES with the environment
        reward = self.env.getRewards(smiles, frags=None)
        
        # TODO: understand what is this
        if memory is not None:
            scores[:len(memory), 0] = 1
            ix = scores[:, 0].argsort()[-self.batchSize * 4:]
            seqs, scores = seqs[ix, :], scores[ix, :]

        # Move rewards to device and create a loader containing the sequences and the rewards
        ds = TensorDataset(seqs, torch.Tensor(reward).to(self.device))
        loader = DataLoader(ds, batch_size=self.n_samples, shuffle=True)
        total_steps = len(loader)

        # Train model with policy gradient
        for step_idx, (seq, reward) in enumerate(tqdm(loader, desc='Iterating over validation batches', leave=False)):
            self.optim.zero_grad()
            loss = self.agent.likelihood(seq)
            loss = loss * (reward - self.beta) 
            loss = -loss.mean()
            loss.backward()
            self.optim.step()
            
            self.monitor.saveProgress(step_idx, None, total_steps, None)
            self.monitor.savePerformanceInfo(step_idx, None, loss.item())
            del loss
 
    def fit(self, train_loader, valid_loader=None, monitor=None, epochs=1000, patience=50, criteria='desired_ratio', min_epochs=100, no_multifrag_smiles=True):
        
        """
        Fit the graph explorer to the training data.
        
        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Data loader for training data
        valid_loader : torch.utils.data.DataLoader
            Data loader for validation data
        epochs : int
            Number of epochs to train for
        patience : int
            Number of epochs to wait for improvement before early stopping
        criteria : str
            Criteria to use for early stopping
        min_epochs : int
            Minimum number of epochs to train for
        monitor : Monitor
            Monitor to use for logging and saving model
        no_multifrag_smiles : bool
            TODO: why here?
            
        Returns
        -------
        None
        """
        
        self.monitor = monitor if monitor else NullMonitor()
        self.monitor.saveModel(self)
        self.bestState = deepcopy(self.agent.state_dict())

        for epoch in tqdm(range(epochs), desc='Fitting SMILES RNN explorer'):
            epoch += 1
            if epoch % 50 == 0 or epoch == 1: logger.info('\n----------\nEPOCH %d\n----------' % epoch)
            if epoch < patience and self.memory is not None:
                smiles, seqs = self.forward(crover=None, memory=self.memory, epsilon=1e-1)
                self.policy_gradient(smiles, seqs, memory=self.memory)
            else:
                smiles, seqs = self.forward(crover=self.crover, epsilon=self.epsilon)
                self.policy_gradient(smiles, seqs)

            # Evaluate the model on the validation set
            smiles = self.agent.sample(self.n_samples)
            scores = self.agent.evaluate(smiles, evaluator=self.env, no_multifrag_smiles=True)
            scores['Smiles'] =  smiles           

            # Save evaluate criteria and save best model
            self.saveBestState(scores, criteria, epoch, None)

            # Log performance and genearated compounds
            self.logPerformanceAndCompounds(epoch, epochs, scores)
 
            if epoch % patience == 0 and epoch != 0:
                # Every nth epoch reset the agent and the crover networks to the best state
                self.agent.load_state_dict(self.bestState)
                self.crover.load_state_dict(self.bestState)
                logger.info('Resetting agent and crover to best state at epoch %d' % self.last_save)

            # Early stopping
            if (epoch >= min_epochs) and (epoch - self.last_save > patience) : break
        
        logger.info('End time reinforcement learning: %s \n' % time.strftime('%d-%m-%y %H:%M:%S', time.localtime()))
        self.monitor.close()