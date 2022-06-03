import torch
from drugex import utils
from torch import nn

from drugex.training.interfaces import Generator
from .attention import DecoderAttn
import time

from drugex.logs import logger
from ..scorers import SmilesChecker


class Base(Generator):

    def attachToDevice(self, device):
        super().attachToDevice(device)
        self.to(self.device)

    def fit(self, pair_loader, ind_loader=None, epochs=100, evaluator=None, monitor=None):

        best = 0.
        net = nn.DataParallel(self, device_ids=self.devices)
        last_save = -1
        # threshold for number of epochs without change that will trigger early stopping
        max_interval = 50
        for epoch in range(epochs):
            t0 = time.time()
            total_steps = len(pair_loader)
            for i, (src, trg) in enumerate(pair_loader):
                src, trg = src.to(self.device), trg.to(self.device)

                self.optim.zero_grad()
                loss = net(src, trg)
                loss = -loss.mean()
                print(epoch, i, loss)
                loss.backward()
                self.optim.step()

                if i % 1000 != 0: continue
                frags, smiles, scores = self.evaluate(ind_loader, method=evaluator)
                valid = scores.VALID.sum() / len(scores)
                desire = scores.DESIRE.sum() / len(scores)
                t1 = time.time()
                logger.info("Epoch: %d step: %d loss: %.3f valid: %.3f desire: %.3f time: %d\n" %
                      (epoch, i, loss.item(), valid, desire, t1-t0))
                logger.info('%.3f' % (t1-t0))
                t0 = t1
                monitor.saveProgress(i, epoch, total_steps, epochs)

                smiles_scores = []
                for i, smile in enumerate(smiles):
                    logger.debug('%d\t%.3f\t%s\t%s\n' % (scores.VALID[i], scores.DESIRE[i], frags[i], smile))
                    smiles_scores.append((smiles, scores.VALID[i], scores.DESIRE[i], frags[i]))
                monitor.savePerformanceInfo(i, epoch, loss.item(), valid=valid, desire=desire, best=best, smiles_scores=smiles_scores)
                del loss
                if best <= desire:
                    monitor.saveModel(self)
                    best = desire
                    last_save = epoch
                monitor.endStep(i, epoch)
            if epoch - last_save > max_interval: break
        monitor.close()

    def evaluate(self, loader, repeat=1, method=None):
        net = nn.DataParallel(self, device_ids=self.devices)
        frags, smiles = [], []
        with torch.no_grad():
            for _ in range(repeat):
                for ix, src in loader:
                    trg = net(src.to(self.device))
                    ix = loader.dataset.index[ix]
                    smiles += [self.voc_trg.decode(s, is_tk=False) for s in trg]
                    frags += ix.tolist()
                    break
        if method is None:
            method = SmilesChecker()
        scores = method(smiles, frags=frags)
        return frags, smiles, scores

    def init_states(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.to(self.device)


class Seq2Seq(Base):
    def __init__(self, voc_src, voc_trg, emb_sharing=True):
        super(Seq2Seq, self).__init__()
        self.voc_size = 128
        self.hidden_size = 512
        self.voc_src = voc_src
        self.voc_trg = voc_trg
        self.encoder = EncoderRNN(voc_src, self.voc_size, self.hidden_size)
        self.decoder = DecoderAttn(voc_trg, self.voc_size, self.hidden_size)
        if emb_sharing:
            self.encoder.embed.weight = self.decoder.embed.weight
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, input, output=None):
        batch_size = input.size(0)
        memory, hc = self.encoder(input)

        output_ = torch.zeros(batch_size, self.voc_trg.max_len).to(self.device)
        if output is None:
            output_ = output_.long()
        # Start token
        x = torch.LongTensor([self.voc_trg.tk2ix['GO']] * batch_size).to(self.device)
        isEnd = torch.zeros(batch_size).bool().to(self.device)

        for step in range(self.voc_trg.max_len):
            logit, hc = self.decoder(x, hc, memory)
            if output is not None:
                score = logit.log_softmax(dim=-1)
                score = score.gather(1, output[:, step:step + 1]).squeeze()
                output_[:, step] = score
                x = output[:, step]
            else:
                proba = logit.softmax(dim=-1)
                x = torch.multinomial(proba, 1).view(-1)
                x[isEnd] = self.voc_trg.tk2ix['_']
                output_[:, step] = x
                isEnd |= x == self.voc_trg.tk2ix['EOS']
                if isEnd.all(): break
        return output_


class EncDec(Base):
    def __init__(self, voc_src, voc_trg, emb_sharing=True):
        super(EncDec, self).__init__()
        self.voc_size = 128
        self.hidden_size = 512
        self.voc_src = voc_src
        self.voc_trg = voc_trg
        self.encoder = EncoderRNN(voc_src, self.voc_size, self.hidden_size)
        self.decoder = DecoderRNN(voc_trg, self.voc_size, self.hidden_size)
        if emb_sharing:
            self.encoder.embed.weight = self.decoder.embed.weight
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, input, output=None):
        batch_size = input.size(0)
        _, hc = self.encoder(input)

        output_ = torch.zeros(batch_size, self.voc_trg.max_len).to(self.device)
        if output is None:
            output_ = output_.long()

        x = torch.LongTensor([self.voc_trg.tk2ix['GO']] * batch_size).to(self.device)
        isEnd = torch.zeros(batch_size).bool().to(self.device)
        for step in range(self.voc_trg.max_len):
            logit, hc = self.decoder(x, hc)
            if output is not None:
                score = logit.log_softmax(dim=-1)
                score = score.gather(1, output[:, step:step + 1]).squeeze()
                output_[:, step] = score
                x = output[:, step]
            else:
                proba = logit.softmax(dim=-1)
                x = torch.multinomial(proba, 1).view(-1)
                x[isEnd] = self.voc_trg.tk2ix['_']
                output_[:, step] = x
                isEnd |= x == self.voc_trg.tk2ix['EOS']
                if isEnd.all(): break
        return output_


class EncoderRNN(nn.Module):
    def __init__(self, voc, d_emb=128, d_hid=512, is_lstm=True, n_layers=3):
        super(EncoderRNN, self).__init__()
        self.voc = voc
        self.hidden_size = d_hid
        self.embed_size = d_emb
        self.n_layers = n_layers
        self.embed = nn.Embedding(voc.size, d_emb)
        rnn_layer = nn.LSTM if is_lstm else nn.GRU
        self.rnn = rnn_layer(d_emb, d_hid, batch_first=True, num_layers=n_layers)

    def forward(self, input):
        if not hasattr(self, '_flattened'):
            self.rnn.flatten_parameters()
            setattr(self, '_flattened', True)

        output = self.embed(input)
        output, (h_out, c_out) = self.rnn(output, None)
        return output, (h_out, c_out)


class DecoderRNN(nn.Module):
    def __init__(self, voc, embed_size, hidden_size, n_layers=3, is_lstm=True):
        super(DecoderRNN, self).__init__()
        self.voc = voc
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = voc.size
        self.n_layers = n_layers
        self.is_lstm = is_lstm

        rnn_layer = nn.LSTM if is_lstm else nn.GRU
        self.embed = nn.Embedding(voc.size, embed_size)
        self.rnn = rnn_layer(embed_size, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, voc.size)

    def forward(self, input, hc):
        if not hasattr(self, '_flattened'):
            self.rnn.flatten_parameters()
            setattr(self, '_flattened', True)
        output = self.embed(input.unsqueeze(-1))
        output, hc = self.rnn(output, hc)
        output = self.linear(output).squeeze(1)
        return output, hc


class ValueNet(nn.Module):
    def __init__(self, voc, embed_size=128, hidden_size=512, max_value=1, min_value=0, n_objs=1, is_lstm=True):
        super(ValueNet, self).__init__()
        self.voc = voc
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = voc.size
        self.max_value = max_value
        self.min_value = min_value
        self.n_objs = n_objs

        self.embed = nn.Embedding(voc.size, embed_size)
        self.is_lstm = is_lstm
        rnn_layer = nn.LSTM if is_lstm else nn.GRU
        self.rnn = rnn_layer(embed_size, hidden_size, num_layers=3, batch_first=True)
        self.linear = nn.Linear(hidden_size, voc.size * n_objs)
        self.optim = torch.optim.Adam(self.parameters())
        self.to(self.device)

    def forward(self, input, h):
        output = self.embed(input.unsqueeze(-1))
        output, h_out = self.rnn(output, h)
        output = self.linear(output).view(len(input), self.n_objs, self.voc.size)
        # output: n_batch * n_obj * voc.size
        return output, h_out

    def init_h(self, batch_size):
        if self.is_lstm:
            return (torch.zeros(3, batch_size, self.hidden_size).to(self.device),
                    torch.zeros(3, batch_size, self.hidden_size).to(self.device))
        else:
            return torch.zeros(3, batch_size, 512).to(self.device)

    def sample(self, batch_size, is_pareto=False):
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(self.device)
        h = self.init_h(batch_size)

        isEnd = torch.zeros(batch_size).bool().to(self.device)
        outputs = []
        for job in range(self.n_objs):
            seqs = torch.zeros(batch_size, self.voc.max_len).long().to(self.device)
            for step in range(self.voc.max_len):
                logit, h = self(x, h)
                logit = logit.view(batch_size, self.voc.size, self.n_objs)
                if is_pareto:
                    proba = torch.zeros(batch_size, self.voc.size).to(self.device)
                    for i in range(batch_size):
                        preds = logit[i, :, :]
                        fronts, ranks = utils.nsgaii_sort(preds)
                        for front in fronts:
                            low, high = preds[front, :].mean(axis=1).min(), preds[front, :].mean(axis=1).max()
                            low = (low - self.min_value) / (self.max_value - self.min_value)
                            high = (high - self.min_value) / (self.max_value - self.min_value)
                            for j, ix in enumerate(front):
                                scale = len(front) - 1 if len(front) > 1 else 1
                                proba[i, ix] = (high - low) * j / scale + low
                else:
                    proba = logit[:, :, job].softmax(dim=-1)
                x = torch.multinomial(proba, 1).view(-1)
                x[isEnd] = self.voc.tk2ix['EOS']
                seqs[:, step] = x

                end_token = (x == self.voc.tk2ix['EOS'])
                isEnd = torch.ge(isEnd + end_token, 1)
                if (isEnd == 1).all(): break
            outputs.append(seqs)
        return torch.cat(outputs, dim=0)