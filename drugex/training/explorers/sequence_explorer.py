#!/usr/bin/env python
import time
from copy import deepcopy

import numpy as np
import torch
from drugex import DEFAULT_DEVICE, DEFAULT_GPUS, utils
from drugex.logs import logger
from drugex.training.explorers.interfaces import Explorer
from drugex.training.generators.utils import unique
from drugex.training.monitors import NullMonitor
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


class SequenceExplorer(Explorer):

    """ 
    `Explorer` to optimize a sequence-based agent (RNN) with the given `Environment`.
 
    Reference: Liu, X., Ye, K., van Vlijmen, H.W.T. et al. DrugEx v2: De Novo Design of Drug Molecule by
               Pareto-based Multi-Objective Reinforcement Learning in Polypharmacology.
               J Cheminform (2021). https://doi.org/10.1186/s13321-019-0355-6
    """

    def __init__(self, agent, env, mutate=None, crover=None, no_multifrag_smiles=True,
        batch_size=128, epsilon=0.1, beta=0.0, n_samples=1000, optim=None,
        device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS):
        super(SequenceExplorer, self).__init__(agent, env, mutate, crover, no_multifrag_smiles, batch_size, epsilon, beta, n_samples, device, use_gpus)
        """
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
        no_multifrag_smiles : bool
            If True, only single-fragment SMILES are valid.
        batch_size : int
            The batch size for the policy gradient update.
        epsilon : float
            The probability of using the `mutate` network to generate molecules.
        beta : float
            The baseline for the reward.
        n_samples : int
            The number of molecules generated in each iteration. (+ an additional 10% for evaluation)
        optim : torch.optim
            The optimizer to update the agent network.
        device : torch.device
            The device to run the network.
        use_gpus : tuple
            The GPU ids to run the network.
        """
        if self.nSamples <= 0:
            self.nSamples = 1000
        self.optim = torch.optim.Adam(self.agent.parameters(), lr=1e-3) if optim is None else optim

    def forward(self):
        """
        Generate molecules with the given `agent` network

        Returns
        -------
        smiles : list
            The generated SMILES.
        seqs : torch.Tensor
            The generated encoded sequences.
        """

        # Generate nSamples molecules
        seqs = []
        while (len(seqs)*self.batchSize) < self.nSamples:
            seq = self.agent.evolve(self.batchSize, epsilon=self.epsilon, crover=self.crover, mutate=self.mutate)
            seqs.append(seq)
        seqs = torch.cat(seqs, dim=0)[:self.nSamples, :]
            
        # Decode the sequences to SMILES
        smiles = np.array([self.agent.voc.decode(s, is_tk = False) for s in seqs])
        ix = unique(np.array([[s] for s in smiles]))
        smiles = smiles[ix]
        seqs = seqs[torch.LongTensor(ix).to(self.device)]
        return smiles, seqs
   
    def policy_gradient(self, smiles=None, seqs=None):
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

        Returns
        -------
        loss : float
            The loss of the policy gradient.
        """

        # Calculate the reward from SMILES with the environment
        reward = self.env.getRewards(smiles, frags=None)

        # Move rewards to device and create a loader containing the sequences and the rewards
        ds = TensorDataset(seqs, torch.Tensor(reward).to(self.device))
        loader = DataLoader(ds, batch_size=self.batchSize, shuffle=True)
        total_steps = len(loader)

        # Train model with policy gradient
        for step_idx, (seq, reward) in enumerate(tqdm(loader, desc='Calculating policy gradient...', leave=False)):
            self.optim.zero_grad()
            loss = self.agent.likelihood(seq)
            loss = loss * (reward - self.beta) 
            loss = -loss.mean()
            loss.backward()
            self.optim.step()
            
            self.monitor.saveProgress(step_idx, None, total_steps, None, loss=loss.item())
        
        return loss.item()
 
    def fit(self, train_loader=None, valid_loader=None, monitor=None, epochs=1000, patience=50, reload_interval = 50, criteria='desired_ratio', min_epochs=100):
        """
        Fit the graph explorer to the training data.
        
        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            ignored, for compatibility with `FragExplorer`
        valid_loader : torch.utils.data.DataLoader
            ignored, for compatibility with `FragExplorer`
        epochs : int
            Number of epochs to train for
        patience : int
            Number of epochs to wait for improvement before early stopping
        reload_interval : int
            Every nth epoch reset the agent (and the crover) network to the best state
        criteria : str
            Criteria to use for early stopping: 'desired_ratio', 'avg_amean' or 'avg_gmean'
        min_epochs : int
            Minimum number of epochs to train for
        monitor : Monitor
            Monitor to use for logging and saving model
        """
        
        self.monitor = monitor if monitor else NullMonitor()
        self.monitor.setModel(self)
        self.monitor.saveModel(self)
        self.bestState = deepcopy(self.agent.state_dict())

        for epoch in tqdm(range(epochs), desc='Fitting SMILES RNN explorer'):
            epoch += 1
            if epoch % 50 == 0 or epoch == 1: logger.info('\n----------\nEPOCH %d\n----------' % epoch)
            is_best = False

            smiles, seqs = self.forward()
            train_loss = self.policy_gradient(smiles, seqs)

            # Evaluate the model on a validation set, which is 10% of the size of training set
            smiles = self.agent.sample(int(np.round(self.nSamples)/10))
            scores = self.agent.evaluate(smiles, evaluator=self.env, no_multifrag_smiles=self.no_multifrag_smiles)
            scores['SMILES'] =  smiles       

            # Compute metrics
            metrics = self.getNovelMoleculeMetrics(scores)       
            metrics['loss_train'] = train_loss

            # Save evaluate criteria and set best model
            if metrics[criteria] > self.best_value:
                is_best = True
                self.saveBestState(metrics[criteria], epoch, None)

            # Save (intermediate) models
            save_model_option = monitor.getSaveModelOption()
            if save_model_option == 'all' or is_best == True:
                monitor.saveModel(self, epoch if save_model_option in ('all', 'best') else None)
                logger.info(f"Model saved at epoch {epoch}")

            # Log performance and generated compounds
            self.logPerformanceAndCompounds(epoch, metrics, scores)
 
            if epoch % reload_interval == 0 and epoch != 0:
                # Every nth epoch reset the agent and the crover networks to the best state
                self.agent.load_state_dict(self.bestState)
                if self.crover is not None:
                    self.crover.load_state_dict(self.bestState)
                logger.info('Resetting agent and crover to best state at epoch %d' % self.last_save)

            # Early stopping
            if (epoch >= min_epochs) and (epoch - self.last_save > patience) : break
        
        logger.info('End time reinforcement learning: %s \n' % time.strftime('%d-%m-%y %H:%M:%S', time.localtime()))
        self.monitor.close()