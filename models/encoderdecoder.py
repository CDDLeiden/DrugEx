import torch
import utils
from torch import nn
from .attention import DecoderAttn
import time
import pandas as pd
from tqdm import tqdm

from drugex.logs import logger

class Base(nn.Module):
    
    def train(self, loader, net):
        
        if self.mol_type == 'smiles':
            for src, trg in loader:
                src, trg = src.to(utils.dev), trg.to(utils.dev)
                self.optim.zero_grad()
                loss = net(src, trg)
                loss = -loss.mean()     
                loss.backward()
                self.optim.step()
                del loss
                
        elif self.mol_type == 'graph':
            for src in loader:
                src = src.to(utils.dev)
                self.optim.zero_grad()
                loss = net(src, is_train=True)
                loss = sum([-l.mean() for l in loss])   
                loss.backward()
                self.optim.step()
                del loss              
                
    def validate(self, loader, net):
        
        frags, smiles, scores = self.evaluate(loader)
        valid = scores.VALID.mean() 
        desired = scores.DESIRE.mean()
        
        with torch.no_grad():
            if self.mol_type == 'smiles':
                loss_valid = sum( [ sum([-l.mean().item() for l in net(src, trg)]) for src, trg in loader ] )
            elif self.mol_type == 'graph':
                loss_valid = sum( [ sum([-l.mean().item() for l in net(src, is_train=False)]) for src in loader ] )
                
        # Should only be done if debug is on
        for j, smile in enumerate(smiles):
            logger.debug('%s\t%s\n' % (frags[j], smile))   
                
        return valid, desired, loss_valid

    
    def fit(self, train_loader, valid_loader, epochs=100, method=None, out=None):
        log = open(out + '.log', 'w')
        best = float('inf')
        net = nn.DataParallel(self, device_ids=utils.devices)
        last_save = -1
        max_interval = 50 # threshold for number of epochs without change that will trigger early stopping
         
        for epoch in tqdm(range(epochs)):
            
            t0 = time.time()
            self.train(train_loader, net)
            valid, _, loss_valid = self.validate(valid_loader, net)
            t1 = time.time()
            
            logger.info(f"Epoch: {epoch} Validation loss: {loss_valid:.3f} Valid: {valid:.3f} Time: {int(t1-t0)}s")
            
            if loss_valid < best:
                torch.save(self.state_dict(), out + '.pkg')
                best = loss_valid
                last_save = epoch
                logger.info(f"Model was saved at epoch {epoch}")         
                
            if epoch - last_save > max_interval : break

    def evaluate(self, loader, repeat=1, method=None):
        net = nn.DataParallel(self, device_ids=utils.devices)
        frags, smiles = [], []
        with torch.no_grad():
            for _ in range(repeat):
                if self.mol_type == 'graph':
                   # Molecules and fragments encoded togther >> graph
                    for src in loader:
                        trg = net(src.to(utils.dev)) 
                        f, s = self.voc_trg.decode(trg)
                        frags += f
                        smiles += s  
                elif self.mol_type == 'smiles':
                    # Molecules and fragments encoded separtetly >> smiles
                    for src, _ in loader:
                        trg = net(src.to(utils.dev))
                        smiles += [self.voc_trg.decode(s, is_tk=False) for s in trg]
                        frags += [self.voc_trg.decode(s, is_tk=False, is_smiles=False) for s in src]
                        break
        if method is None:
            scores = utils.Env.check_smiles(smiles, frags=frags)
            scores = pd.DataFrame(scores, columns=['VALID', 'DESIRE'])
        else:
            scores = method(smiles, frags=frags)
        return frags, smiles, scores

    def init_states(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.to(utils.dev)


class Seq2Seq(Base):
    def __init__(self, voc_src, voc_trg, emb_sharing=True):
        super(Seq2Seq, self).__init__()
        self.mol_type = 'smiles'
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

        output_ = torch.zeros(batch_size, self.voc_trg.max_len).to(utils.dev)
        if output is None:
            output_ = output_.long()
        # Start token
        x = torch.LongTensor([self.voc_trg.tk2ix['GO']] * batch_size).to(utils.dev)
        isEnd = torch.zeros(batch_size).bool().to(utils.dev)

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
        self.mol_type = 'smiles'
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

        output_ = torch.zeros(batch_size, self.voc_trg.max_len).to(utils.dev)
        if output is None:
            output_ = output_.long()

        x = torch.LongTensor([self.voc_trg.tk2ix['GO']] * batch_size).to(utils.dev)
        isEnd = torch.zeros(batch_size).bool().to(utils.dev)
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
        self.to(utils.dev)

    def forward(self, input, h):
        output = self.embed(input.unsqueeze(-1))
        output, h_out = self.rnn(output, h)
        output = self.linear(output).view(len(input), self.n_objs, self.voc.size)
        # output: n_batch * n_obj * voc.size
        return output, h_out

    def init_h(self, batch_size):
        if self.is_lstm:
            return (torch.zeros(3, batch_size, self.hidden_size).to(utils.dev),
                    torch.zeros(3, batch_size, self.hidden_size).to(utils.dev))
        else:
            return torch.zeros(3, batch_size, 512).to(utils.dev)

    def sample(self, batch_size, is_pareto=False):
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(utils.dev)
        h = self.init_h(batch_size)

        isEnd = torch.zeros(batch_size).bool().to(utils.dev)
        outputs = []
        for job in range(self.n_objs):
            seqs = torch.zeros(batch_size, self.voc.max_len).long().to(utils.dev)
            for step in range(self.voc.max_len):
                logit, h = self(x, h)
                logit = logit.view(batch_size, self.voc.size, self.n_objs)
                if is_pareto:
                    proba = torch.zeros(batch_size, self.voc.size).to(utils.dev)
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