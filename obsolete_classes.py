import torch
import torch.nn as nn
import torch.nn.functional as F
from drugex import utils

class ScaledDotProduct(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, dropout=0.):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -float('Inf'))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, n_head, d_k=64, d_v=64, dropout=0.):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProduct(temperature=d_k ** 0.5, dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        q = self.attention(q, k, v, mask=mask)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        return q


class Attn(nn.Module):
    def __init__(self, h_dim):
        super(Attn, self).__init__()
        self.h_dim = h_dim
        self.main = nn.Sequential(
            nn.Linear(h_dim, 100),
            nn.ReLU(True),
            nn.Linear(100, 1)
        )

    def forward(self, encoder_outputs):
        b_size = encoder_outputs.size(0)
        attn = self.main(encoder_outputs.contiguous().view(-1, self.h_dim))  # (b, s, h) -> (b * s, 1)
        attn = F.softmax(attn.contiguous().view(b_size, -1), dim=1).unsqueeze(2)  # (b*s, 1) -> (b, s, 1)
        return attn

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