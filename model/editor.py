import torch
from torch import nn
from torch.nn import functional as F
from model.entail import Module as Base
from torch.nn.utils.rnn import pad_sequence
from preprocess_sharc import CLASSES, detokenize


class Decoder(nn.Module):

    def __init__(self, denc, emb, dropout=0):
        super().__init__()
        dhid = denc
        self.demb = emb.size(1)
        self.vocab_size = emb.size(0)
        self.emb = nn.Embedding(self.vocab_size, self.demb)
        self.emb.weight.data = emb

        self.dropout = nn.Dropout(dropout)

        self.attn_scorer = nn.Linear(denc, 1)

        self.rnn = nn.LSTMCell(denc+self.demb, dhid)

        self.proj = nn.Linear(denc+dhid, self.demb)

        self.emb0 = nn.Parameter(torch.Tensor(self.demb))
        self.h0 = nn.Parameter(torch.Tensor(dhid))
        self.c0 = nn.Parameter(torch.Tensor(dhid))

        for p in [self.emb0, self.h0, self.c0]:
            nn.init.uniform_(p, -0.1, 0.1)

    def forward(self, enc, inp_mask, label, max_decode_len=30):
        max_t = label.size(1) if self.training else max_decode_len
        batch = enc.size(0)
        h_t = self.h0.repeat(batch, 1)
        c_t = self.c0.repeat(batch, 1)
        emb_t = self.emb0.repeat(batch, 1)

        outs = []
        for t in range(max_t):
            h_t = self.dropout(h_t)
            # attend to input
            inp_score = enc.bmm(h_t.unsqueeze(2)).squeeze(2) - (1-inp_mask) * 1e20
            inp_score_norm = F.softmax(inp_score, dim=1)
            inp_attn = inp_score_norm.unsqueeze(2).expand_as(enc).mul(enc).sum(1)

            rnn_inp = self.dropout(torch.cat([inp_attn, emb_t], dim=1))

            h_t, c_t = self.rnn(rnn_inp, (h_t, c_t))

            proj_inp = self.dropout(torch.cat([inp_attn, h_t], dim=1))
            proj = self.proj(proj_inp)

            out_t = proj.mm(self.emb.weight.t().detach())
            outs.append(out_t)
            word_t = label[:, t] if self.training else out_t.max(1)[1]
            # get rid of -1's from unchosen spans
            word_t = torch.clamp(word_t, 0, self.vocab_size)
            emb_t = self.emb(word_t)
        outs = torch.stack(outs, dim=1)
        return outs


class Module(Base):

    def __init__(self, args):
        super().__init__(args)
        vocab = torch.load('{}/vocab.pt'.format(args.data))
        self.vocab = vocab['vocab']
        self.decoder = Decoder(
            emb=vocab['emb'],
            denc=self.args.bert_hidden_size,
            dropout=self.args.dropout,
        )

    def forward(self, batch):
        out = super().forward(batch)

        out['edit_scores'] = decs = []
        out['edit_labels'] = labels = []
        for ex, enc, spans in zip(batch, out['bert_enc'], out['spans']):
            inp = [enc[s:e+1] for s, e in spans]
            lens = [t.size(0) for t in inp]
            max_len = max(lens)
            mask = torch.tensor([[1] * l + [0] * (max_len-l) for l in lens], device=self.device, dtype=torch.float)
            inp = pad_sequence(inp, batch_first=True, padding_value=0)
            label = pad_sequence([torch.tensor(o, dtype=torch.long) for o in ex['edit_num']['out_vocab_id']], batch_first=True, padding_value=-1).to(self.device) if self.training else None
            dec = self.decoder(inp, mask, label)
            decs.append(dec)
            labels.append(label)
        return out

    def compute_loss(self, out, batch):
        loss = super().compute_loss(out, batch)
        edit_loss = 0
        for ex, dec in zip(batch, out['edit_scores']):
            label = pad_sequence([torch.tensor(o, dtype=torch.long) for o in ex['edit_num']['out_vocab_id']], batch_first=True, padding_value=-1).to(self.device)
            edit_loss += F.cross_entropy(dec.view(-1, dec.size(-1)), label.view(-1), ignore_index=-1)
        loss['edit'] = edit_loss / len(batch) * self.args.loss_editor_weight
        return loss

    def extract_preds(self, out, batch, top_k=20):
        preds = []
        for ex, clf_i, retrieve_i, spans_i, edit_scores_i in zip(batch, out['clf_scores'].max(1)[1].tolist(), out['retrieve_scores'].max(1)[1].tolist(), out['spans'], out['edit_scores']):
            a = CLASSES[clf_i]
            edit_ids = edit_scores_i.max(2)[1].tolist()
            edits = []
            for ids in edit_ids:
                words = self.vocab.index2word(ids)
                if 'eos' in words:
                    words = words[:words.index('eos')]
                edits.append(' '.join(words))
            r = None
            if a == 'more':
                s, e = spans_i[retrieve_i]
                r = detokenize(ex['feat']['inp'][s:e+1])
                a = edits[retrieve_i]
            preds.append({
                'utterance_id': ex['utterance_id'],
                'retrieval': r,
                'answer': a,
                'spans': spans_i,
            })
        return preds
