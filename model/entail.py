import torch
from model.retrieve import Module as Base
from model.span import Module as SpanModule
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from preprocess_sharc import detokenize, CLASSES
from metric import compute_f1
from tqdm import trange


class Module(Base):

    def __init__(self, args):
        super().__init__(args)
        self.span_attn_scorer = nn.Linear(self.args.bert_hidden_size, 1)
        self.span_retrieval_scorer = nn.Linear(self.args.bert_hidden_size+2, 1)
        self.inp_attn_scorer = nn.Linear(self.args.bert_hidden_size, 1)
        self.class_clf = nn.Linear(self.args.bert_hidden_size, len(CLASSES))

    def compute_entailment(self, spans, ex):
        chunks = [detokenize(ex['feat']['inp'][s:e+1]) for s, e in spans]
        history = [0] * len(chunks)
        scenario = [0] * len(chunks)
        # history
        for i, c in enumerate(chunks):
            for q in ex['ann']['hquestion']:
                history[i] = max(history[i], compute_f1(c, detokenize(q)))
            scenario[i] = max(scenario[i], compute_f1(c, detokenize(ex['ann']['scenario'])))
        entail = torch.tensor([history, scenario], dtype=torch.float, device=self.device).t()
        return entail

    def forward(self, batch):
        out = SpanModule.forward(self, batch)
        if self.training:
            spans = out['spans'] = [ex['feat']['spans'] for ex in batch]
        else:
            spans = [[span[:2] for span in spans_i] for spans_i in self.extract_spans(out['span_scores'], batch)]
            spans = out['spans'] = [self.extract_bullets(s, ex) for s, ex in zip(spans, batch)]

        span_enc = []
        out['entail'] = []
        for h_i, spans_i, ex_i in zip(out['bert_enc'], spans, batch):
            span_h = [h_i[s:e+1] for s, e in spans_i]
            max_len = max([h.size(0) for h in span_h])
            span_mask = torch.tensor([[1] * h.size(0) + [0] * (max_len-h.size(0)) for h in span_h], device=self.device, dtype=torch.float)
            span_h = pad_sequence(span_h, batch_first=True, padding_value=0)
            span_attn_mask = pad_sequence(span_mask, batch_first=True, padding_value=0)
            span_attn_score = self.span_attn_scorer(self.dropout(span_h)).squeeze(2) - (1-span_attn_mask).mul(1e20)
            span_attn = F.softmax(span_attn_score, dim=1).unsqueeze(2).expand_as(span_h).mul(self.dropout(span_h)).sum(1)
            span_entail = self.compute_entailment(spans_i, ex_i)
            out['entail'].append(span_entail)
            span_enc.append(torch.cat([span_attn, span_entail], dim=1))
        max_len = max([h.size(0) for h in span_enc])
        span_mask = torch.tensor([[1] * h.size(0) + [0] * (max_len-h.size(0)) for h in span_enc], device=self.device, dtype=torch.float)
        span_enc = pad_sequence(span_enc, batch_first=True, padding_value=0)
        out['retrieve_scores'] = self.span_retrieval_scorer(self.dropout(span_enc)).squeeze(2) - (1-span_mask).mul(1e20)

        inp_attn_score = self.inp_attn_scorer(self.dropout(out['bert_enc'])).squeeze(2) - (1-out['input_mask'].float()).mul(1e20)
        inp_attn = F.softmax(inp_attn_score, dim=1).unsqueeze(2).expand_as(out['bert_enc']).mul(self.dropout(out['bert_enc'])).sum(1)
        out['clf_scores'] = self.class_clf(self.dropout(inp_attn))
        return out

    def extract_preds(self, out, batch, top_k=20):
        preds = super().extract_preds(out, batch, top_k=top_k)
        for ex, p, span_i, clf_i, retrieve_i, entail_i in zip(batch, preds, out['span_scores'], out['clf_scores'], out['retrieve_scores'], out['entail']):
            p['clf_scores'] = dict(list(zip(CLASSES, F.softmax(clf_i, dim=0).tolist())))
            spans = [detokenize(ex['feat']['inp'][s:e+1]) for s, e in p['spans']]
            p['span_scores'] = dict(list(zip(spans, F.softmax(retrieve_i, dim=0).tolist())))
            p['words'] = [w['sub'] for w in ex['feat']['inp'] if w['orig'] != 'pad']
            p['og'] = {k: v for k, v in ex.items() if k in ['snippet', 'scenario', 'question', 'history', 'answer']}
            p['start_scores'] = span_i[:, 0].tolist()
            p['end_scores'] = span_i[:, 1].tolist()
            p['entail_hist_scores'] = dict(list(zip(spans, entail_i[:, 0].tolist())))
            p['entail_scen_scores'] = dict(list(zip(spans, entail_i[:, 1].tolist())))
        return preds
