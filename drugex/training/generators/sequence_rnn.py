from copy import deepcopy

import numpy as np
import torch
from tqdm.auto import tqdm
from torch import nn
from torch import optim
from drugex import utils, DEFAULT_DEVICE, DEFAULT_GPUS

from drugex.logs import logger
from drugex.training.interfaces import Generator
from drugex.training.scorers.smiles import SmilesChecker
from drugex.training.monitors import NullMonitor


class SequenceRNN(Generator):
    def __init__(self, voc, embed_size=128, hidden_size=512, is_lstm=True, lr=1e-3, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS):
        super(SequenceRNN, self).__init__(device=device, use_gpus=use_gpus)
        self.voc = voc
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = voc.size

        self.embed = nn.Embedding(voc.size, embed_size)
        self.is_lstm = is_lstm
        rnn_layer = nn.LSTM if is_lstm else nn.GRU
        self.rnn = rnn_layer(embed_size, hidden_size, num_layers=3, batch_first=True)
        self.linear = nn.Linear(hidden_size, voc.size)
        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.attachToGPUs(self.gpus)

        self.model_name = 'SequenceRNN'

    def attachToGPUs(self, gpus):
        """
        This model currently uses only one GPU. Therefore, only the first one from the list will be used.

        Args:
            gpus: a `tuple` of GPU IDs to attach this model to (only the first one will be used)

        Returns:

        """
        self.device = torch.device(f'cuda:{gpus[0]}')
        self.to(self.device)
        self.gpus = (gpus[0],)


    def forward(self, input, h):
        output = self.embed(input.unsqueeze(-1))
        output, h_out = self.rnn(output, h)
        output = self.linear(output).squeeze(1)
        return output, h_out

    def init_h(self, batch_size, labels=None):
        h = torch.rand(3, batch_size, 512).to(self.device)
        if labels is not None:
            h[0, batch_size, 0] = labels
        if self.is_lstm:
            c = torch.rand(3, batch_size, self.hidden_size).to(self.device)
        return (h, c) if self.is_lstm else h

    def likelihood(self, target):
        batch_size, seq_len = target.size()
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(self.device)
        h = self.init_h(batch_size)
        scores = torch.zeros(batch_size, seq_len).to(self.device)
        for step in range(seq_len):
            logits, h = self(x, h)
            logits = logits.log_softmax(dim=-1)
            score = logits.gather(1, target[:, step:step+1]).squeeze()
            scores[:, step] = score
            x = target[:, step]
        return scores

    def PGLoss(self, loader, progress=None):
        total_steps = len(loader)
        step_idx = 0
        for seq, reward in loader:
            self.zero_grad()
            score = self.likelihood(seq)
            loss = score * reward
            loss = -loss.mean()
            if progress:
                progress.saveProgress(step_idx, None, total_steps, None)
                progress.savePerformanceInfo(step_idx, None, loss.item())
            loss.backward()
            self.optim.step()
            step_idx += 1
    
    def sample(self, batch_size):
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(self.device)
        h = self.init_h(batch_size)
        sequences = torch.zeros(batch_size, self.voc.max_len).long().to(self.device)
        isEnd = torch.zeros(batch_size).bool().to(self.device)

        for step in range(self.voc.max_len):
            logit, h = self(x, h)
            proba = logit.softmax(dim=-1)
            x = torch.multinomial(proba, 1).view(-1)
            x[isEnd] = self.voc.tk2ix['EOS']
            sequences[:, step] = x
            end_token = (x == self.voc.tk2ix['EOS'])
            isEnd = torch.ge(isEnd + end_token, 1)
            if (isEnd == 1).all(): break

        return [self.voc.decode(s, is_tk = False) for s in sequences]

    def evolve(self, batch_size, epsilon=0.01, crover=None, mutate=None):
        # Start tokens
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(self.device)
        # Hidden states initialization for exploitation network (agent)
        hA = self.init_h(batch_size)
        # Hidden states initialization for exploration networks (mutate and crover)
        hM = self.init_h(batch_size)
        hC = self.init_h(batch_size)
        # Initialization of output matrix
        sequences = torch.zeros(batch_size, self.voc.max_len).long().to(self.device)
        # labels to judge and record which sample is ended
        is_end = torch.zeros(batch_size).bool().to(self.device)

        for step in range(self.voc.max_len):
            # Get unscaled logits and hidden states from agent network and convert them to probabilities with softmax
            logitA, hA = self(x, hA)
            proba = logitA.softmax(dim=-1)

            # If crover combine probablities from agent network and crover network
            if crover is not None:
                ratio = torch.rand(batch_size, 1).to(self.device)
                logitC, hC = crover(x, hC)
                proba = proba * ratio + logitC.softmax(dim=-1) * (1 - ratio)
            
            # If mutate replace with the epsilon-rate the probabilities with the ones from the mutation network
            if mutate is not None:
                logitM, hM = mutate(x, hM)
                is_mutate = (torch.rand(batch_size) < epsilon).to(self.device)
                proba[is_mutate, :] = logitM.softmax(dim=-1)[is_mutate, :]
            

            # sampling based on output probability distribution
            x = torch.multinomial(proba, 1).view(-1)

            is_end |= x == self.voc.tk2ix['EOS']
            x[is_end] = self.voc.tk2ix['EOS']
            sequences[:, step] = x
            if is_end.all(): break
        return sequences

    def trainNet(self, loader, epoch, epochs):

        """Train the RNN network"""

        total_steps = len(loader)
        for i, batch in enumerate(loader):
            self.optim.zero_grad()
            loss = self.likelihood(batch.to(self.device))
            loss = -loss.mean()
            loss.backward()
            self.optim.step()
            self.monitor.saveProgress(i, epoch, total_steps, epochs)
            self.monitor.savePerformanceInfo(i, epoch, loss.item())


        return loss.item()

    def validateNet(self, batch_size, loader_valid=None, evaluator=None, no_multifrag_smiles=True):

        """Validate the RNN network by sampling SMILES and evaluate them or check their validity
        
        Parameters:
        -----------
        batch: int
            Number of SMILES to sample
        loader (opt): DataLoader
            DataLoader used to calculate the validation loss
        evaluator (opt): Evaluator
            Evaluator to evaluate the SMILES

        Returns:
        --------
        valid_metrics: dict
            Dictionary containing the validation metrics
        scores: pd.DataFrame
            DataFrame containing the SMILES and their scores
        """ 

        valid_metrics = {}
        smiles = self.sample(batch_size)
        scores = self.evaluate(smiles, evaluator=evaluator)
        valid_metrics['valid_ratio'] = scores.VALID.mean()

        # If a separate validation set is provided, use it to compute the validation loss 
        if loader_valid is not None:
            loss_valid, size = 0, 0
            for j, batch in enumerate(loader_valid):
                size += batch.size(0)
                loss_valid += -self.likelihood(batch.to(self.device)).sum().item()
            valid_metrics['loss_valid'] = loss_valid / size / self.voc.max_len

        scores['Smiles'] = smiles

        return valid_metrics, scores       
    

    # def fit(self, loader_train, loader_valid=None, epochs=100, monitor=None, lr=1e-3, patience=50):
        
    #     self.monitor = monitor if monitor else NullMonitor() 
    #     best_value = float('inf')
    #     last_save = -1

    #     self.optim = optim.Adam(self.parameters(), lr=lr)
        
    #     for epoch in tqdm(range(epochs), desc='Fitting model'):
    #         epoch += 1
            
    #         # Train the model
    #         train_loss = self.trainNet(loader_train, epoch, epochs)
            
    #         # Validate the model
    #         batch_size = loader_train.batch_size * 2 # Sohvi: why training batch size *2?
    #         valid_ratio, valid_loss, smiles_scores = self.validateNet(batch_size, evaluator=None)
    #         self.monitor.saveProgress(None, epoch, None, epochs)

    #         # If validation loss given, save the model if the validation loss is the best so far
    #         # Otherwise, save the model if the validation error rate is the best so far
    #         if valid_loss is not None: value = valid_loss
    #         else : value = 1 - valid_ratio

    #         if value < best_value:
    #             self.monitor.saveModel(self)
    #             best_value = value
    #             last_save = epoch
    #             logger.info(f"Model was saved at epoch {epoch}") 

    #         self.monitor.savePerformanceInfo(None, epoch, train_loss.item(), loss_valid=valid_loss, smiles_scores=smiles_scores, smiles_scores_key=['Smiles', 'VALID'], valid_ratio=valid_ratio, error = 1 - valid_ratio)     
            
    #         del train_loss, valid_loss
    #         monitor.endStep(None, epoch)

    #         if epoch - last_save > patience:
    #             break

    #         torch.cuda.empty_cache()
    #         monitor.close()
            
        #     # each epoch sample SMILES generated by model to assess error rate    
        #     seqs = self.sample(len(batch * 2))
        #     ix = utils.unique(seqs)
        #     seqs = seqs[ix]
        #     smiles = [self.voc.decode(s, is_tk = False) for s in seqs]
        #     valids = SmilesChecker.checkSmiles(smiles, frags=None)
        #     error = (1 - sum(valids) / len(seqs))[0]
        #     info = "Epoch: %d error_rate: %.3f loss_train: %.3f" % (epoch, error, loss_train.item())
        #     loss_valid = None
        #     if loader_valid is not None:
        #         loss_valid, size = 0, 0
        #         for j, batch in enumerate(loader_valid):
        #             size += batch.size(0)
        #             loss_valid += -self.likelihood(batch.to(self.device)).sum().item()
        #         loss_valid = loss_valid / size / self.voc.max_len
        #         if loss_valid < best_error:
        #             monitor.saveModel(self)
        #             best_error = loss_valid
        #             last_save = epoch
        #         info += ' loss_valid: %.3f' % loss_valid
        #     elif error < best_error:
        #         monitor.saveModel(self)
        #         best_error = error
        #         last_save = epoch
        #     logger.info(info)
        #     smiles_scores = []
        #     for i, smile in enumerate(smiles):
        #         smiles_scores.append((smile, valids[i][0]))
        #         logger.debug('%d\t%s' % (valids[i][0], smile))
        #     monitor.savePerformanceInfo(None, epoch, loss_train.item(), loss_valid=loss_valid, smiles_scores=smiles_scores, smiles_scores_key=['Smiles', 'VALID'], error=error, valid_ratio=1 - error)
        #     logger.info(info)
        #     monitor.endStep(None, epoch)
        #     if epoch - last_save > max_interval: break
        # torch.cuda.empty_cache()
        # monitor.close()

    def sample_smiles(self, num_samples, batch_size=100, drop_duplicates=True, drop_invalid=True, progress=True, tqdm_kwargs={}):
        if progress:
            tqdm_kwargs.update({'total': num_samples, 'desc': 'Generating molecules'})
            pbar = tqdm(**tqdm_kwargs)
        smiles = []
        while len(smiles) < num_samples:
            # sample SMILES
            smiles = self.sample(batch_size)
            # decode according to vocabulary
            new_smiles = utils.canonicalize_list(smiles)
            # drop duplicates
            if drop_duplicates:
                new_smiles = np.array(new_smiles)
                new_smiles = new_smiles[np.logical_not(np.isin(new_smiles, smiles))]
                new_smiles = new_smiles.tolist()
            # drop invalid smiles
            if drop_invalid:
                scores = SmilesChecker.checkSmiles(new_smiles, frags=None).ravel()
                new_smiles = np.array(new_smiles)[scores > 0].tolist()
            smiles += new_smiles
            # Update progress bar
            if progress:
                pbar.update(len(new_smiles) if pbar.n + len(new_smiles) <= num_samples else num_samples - pbar.n)
        smiles = smiles[:num_samples]
        if progress:
            pbar.close()
        return smiles
