import torch
from model.span import Module as Base
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from preprocess_sharc import detokenize, CLASSES, compute_metrics
from metric import compute_f1


class Module(Base):

    def __init__(self, args):
        super().__init__(args)
        self.span_attn_scorer = nn.Linear(self.args.bert_hidden_size, 1)
        self.span_retrieval_scorer = nn.Linear(self.args.bert_hidden_size, 1)
        self.inp_attn_scorer = nn.Linear(self.args.bert_hidden_size, 1)
        self.class_clf = nn.Linear(self.args.bert_hidden_size, len(CLASSES))

    def extract_bullets(self, spans, ex):
        mask = ex['feat']['pointer_mask'].tolist()
        classes_start = mask.index(1)
        snippet_start = classes_start + 5
        snippet_end = snippet_start + mask[snippet_start:].index(0)
        bullet_inds = [i for i in range(snippet_start, snippet_end) if ex['feat']['inp'][i]['sub'] == '*']
        if bullet_inds:
            bullets = [(s+1, e-1) for s, e in zip(bullet_inds, bullet_inds[1:] + [snippet_end]) if e-1 >= s+1]
            non_bullet_spans = []
            for s, e in spans:
                gloss = detokenize(ex['feat']['inp'])
                if '*' not in gloss and '\n' not in gloss:
                    non_bullet_spans.append((s, e))
            all_spans = bullets + non_bullet_spans
            all_spans.sort(key=lambda tup: tup[1]-tup[0], reverse=True)
            covered = [False] * len(ex['feat']['inp'])
            keep = []
            for s, e in all_spans:
                if not all(covered[s:e+1]):
                    for i in range(s, e+1):
                        covered[i] = True
                    keep.append((s, e))
            return keep
        else:
            return spans

    def forward(self, batch):
        out = super().forward(batch)
        if self.training:
            spans = out['spans'] = [ex['feat']['spans'] for ex in batch]
        else:
            spans = [[span[:2] for span in spans_i] for spans_i in self.extract_spans(out['span_scores'], batch)]
            spans = out['spans'] = [self.extract_bullets(s, ex) for s, ex in zip(spans, batch)]

        span_enc = []
        for h_i, spans_i in zip(out['bert_enc'], spans):
            span_h = [h_i[s:e+1] for s, e in spans_i]
            max_len = max([h.size(0) for h in span_h])
            span_mask = torch.tensor([[1] * h.size(0) + [0] * (max_len-h.size(0)) for h in span_h], device=self.device, dtype=torch.float)
            span_h = pad_sequence(span_h, batch_first=True, padding_value=0)
            span_attn_mask = pad_sequence(span_mask, batch_first=True, padding_value=0)
            span_attn_score = self.span_attn_scorer(self.dropout(span_h)).squeeze(2) - (1-span_attn_mask).mul(1e20)
            span_attn = F.softmax(span_attn_score, dim=1).unsqueeze(2).expand_as(span_h).mul(self.dropout(span_h)).sum(1)
            span_enc.append(span_attn)
        max_len = max([h.size(0) for h in span_enc])
        span_mask = torch.tensor([[1] * h.size(0) + [0] * (max_len-h.size(0)) for h in span_enc], device=self.device, dtype=torch.float)
        span_enc = pad_sequence(span_enc, batch_first=True, padding_value=0)
        out['retrieve_scores'] = self.span_retrieval_scorer(self.dropout(span_enc)).squeeze(2) - (1-span_mask).mul(1e20)

        inp_attn_score = self.inp_attn_scorer(self.dropout(out['bert_enc'])).squeeze(2) - (1-out['input_mask'].float()).mul(1e20)
        inp_attn = F.softmax(inp_attn_score, dim=1).unsqueeze(2).expand_as(out['bert_enc']).mul(self.dropout(out['bert_enc'])).sum(1)
        out['clf_scores'] = self.class_clf(self.dropout(inp_attn))
        return out

    def extract_preds(self, out, batch, top_k=20):
        preds = []
        for ex, clf_i, retrieve_i, span_i in zip(batch, out['clf_scores'].max(1)[1].tolist(), out['retrieve_scores'].max(1)[1].tolist(), out['spans']):
            a = CLASSES[clf_i]
            if a == 'more':
                s, e = span_i[retrieve_i]
                a = detokenize(ex['feat']['inp'][s:e+1])
            preds.append({
                'utterance_id': ex['utterance_id'],
                'answer': a,
                'spans': span_i,
                'retrieve_span': retrieve_i,
            })
        return preds

    def compute_metrics(self, preds, data):
        metrics = compute_metrics(preds, data)
        f1s = []
        for p, ex in zip(preds, data):
            pspans = [detokenize(ex['feat']['inp'][s:e+1]) for s, e in p['spans']]
            gspans = [detokenize(ex['feat']['inp'][s:e+1]) for s, e in ex['feat']['spans']]
            f1s.append(compute_f1('\n'.join(gspans), '\n'.join(pspans)))
        metrics['span_f1'] = sum(f1s) / len(f1s)
        return metrics

    def compute_loss(self, out, batch):
        gclf = torch.tensor([ex['feat']['answer_class'] for ex in batch], device=self.device, dtype=torch.long)
        gretrieve = torch.tensor([ex['feat']['answer_span'] for ex in batch], device=self.device, dtype=torch.long)
        loss = {
            'clf': F.cross_entropy(out['clf_scores'], gclf),
            'retrieve': F.cross_entropy(out['retrieve_scores'], gretrieve, ignore_index=-1),
        }
        loss['span_start'], loss['span_end'] = self.get_span_loss(out, batch)
        loss['span_start'] *= self.args.loss_span_weight
        loss['span_end'] *= self.args.loss_span_weight
        return loss
