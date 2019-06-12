import torch
from model.base import Module as Base
from torch import nn
from torch.nn import functional as F
from preprocess_sharc import detokenize
from metric import compute_f1


class Module(Base):

    def __init__(self, args):
        super().__init__(args)
        self.span_scorer = nn.Linear(self.args.bert_hidden_size, 2)

    def forward(self, batch):
        out = super().forward(batch)
        span_scores = self.span_scorer(self.dropout(out['bert_enc']))
        out['span_scores'] = self.mask_scores(span_scores, out['pointer_mask']).sigmoid()
        return out

    def extract_spans(self, span_scores, batch):
        pstart, pend = span_scores.split(1, dim=-1)
        spans = []
        for pstart_i, pend_i, ex in zip(pstart.squeeze(-1), pend.squeeze(-1), batch):
            spans_i = []
            sthresh = min(pstart_i.max(), self.args.thresh)
            start = pstart_i.ge(sthresh).tolist()
            for si, strig in enumerate(start):
                if strig:
                    ethresh = min(pend_i[si:].max(), self.args.thresh)
                    end = pend_i[si:].ge(ethresh).tolist()
                    for ei, etrig in enumerate(end):
                        ei += si
                        if etrig:
                            spans_i.append((si, ei, detokenize(ex['feat']['inp'][si:ei+1]), pstart_i[si].item(), pend_i[ei].item()))
                            break
            spans.append(spans_i)
        return spans

    def extract_preds(self, out, batch, top_k=20):
        preds = super().extract_preds(out, batch, top_k=top_k)
        for p, s in zip(preds, self.extract_spans(out['span_scores'], batch)):
            p['spans'] = s
        return preds

    def compute_metrics(self, preds, data):
        metrics = super().compute_metrics(preds, data)
        f1s = []
        for p, ex in zip(preds, data):
            pspans = [gloss for s, e, gloss, ps, pe in p['spans']]
            gspans = [detokenize(ex['feat']['inp'][s:e+1]) for s, e in ex['feat']['spans']]
            f1s.append(compute_f1('\n'.join(gspans), '\n'.join(pspans)))
        metrics['span_f1'] = sum(f1s) / len(f1s)
        return metrics

    def get_span_loss(self, out, batch):
        span_scores = out['span_scores']
        ystart, yend = span_scores.split(1, dim=-1)

        gstart = []
        gend = []
        for ex in batch:
            gstart_i = [0] * len(ex['feat']['inp'])
            gend_i = [0] * len(ex['feat']['inp'])
            for s, e in ex['feat']['spans']:
                gstart_i[s] = 1
                gend_i[e] = 1
            gstart.append(gstart_i)
            gend.append(gend_i)
        gstart = torch.tensor(gstart, dtype=torch.float, device=self.device)
        gend = torch.tensor(gend, dtype=torch.float, device=self.device)

        lstart = F.binary_cross_entropy(ystart.squeeze(-1), gstart)
        lend = F.binary_cross_entropy(yend.squeeze(-1), gend)
        return lstart, lend

    def compute_loss(self, out, batch):
        loss = super().compute_loss(out, batch)
        loss['span_start'], loss['span_end'] = self.get_span_loss(out, batch)
        loss['span_start'] *= self.args.loss_span_weight
        loss['span_end'] *= self.args.loss_span_weight
        return loss
