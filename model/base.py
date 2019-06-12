import os
import shutil
import torch
import logging
import importlib
import numpy as np
import json
from tqdm import trange
from pprint import pformat
from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from preprocess_sharc import detokenize, compute_metrics, BERT_MODEL
from pytorch_pretrained_bert import BertModel, BertAdam
from argparse import Namespace


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


DEVICE = torch.device('cpu')
if torch.cuda.is_available() and torch.cuda.device_count():
    DEVICE = torch.device('cuda')
    torch.cuda.manual_seed_all(0)


class Module(nn.Module):

    def __init__(self, args, device=DEVICE):
        super().__init__()
        self.args = args
        self.device = device
        self.bert = BertModel.from_pretrained(BERT_MODEL, cache_dir=None)
        self.dropout = nn.Dropout(self.args.dropout)
        self.ans_scorer = nn.Linear(self.args.bert_hidden_size, 2)
        self.epoch = 0

    @classmethod
    def load_module(cls, name):
        return importlib.import_module('model.{}'.format(name)).Module

    @classmethod
    def load(cls, fname, override_args=None):
        load = torch.load(fname, map_location=lambda storage, loc: storage)
        args = vars(load['args'])
        if override_args:
            args.update(override_args)
        args = Namespace(**args)
        model = cls.load_module(args.model)(args)
        model.load_state_dict(load['state'])
        return model

    def save(self, metrics, dsave, early_stop):
        files = [os.path.join(dsave, f) for f in os.listdir(dsave) if f.endswith('.pt') and f != 'best.pt']
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        if len(files) > self.args.keep-1:
            for f in files[self.args.keep-1:]:
                os.remove(f)

        fsave = os.path.join(dsave, 'ep{}-{}.pt'.format(metrics['epoch'], metrics[early_stop]))
        torch.save({
            'args': self.args,
            'state': self.state_dict(),
            'metrics': metrics,
        }, fsave)
        fbest = os.path.join(dsave, 'best.pt')
        if os.path.isfile(fbest):
            os.remove(fbest)
        shutil.copy(fsave, fbest)

    def create_input_tensors(self, batch):
        feat = {
            k: torch.stack([e['feat'][k] for e in batch], dim=0).to(self.device)
            for k in ['input_ids', 'type_ids', 'input_mask', 'pointer_mask']
        }
        # for ex in batch:
        #     s = ex['feat']['answer_start']
        #     e = ex['feat']['answer_end']
        #     print(s, ex['feat']['pointer_mask'][s])
        #     print(e, ex['feat']['pointer_mask'][e])
        #     import pdb; pdb.set_trace()
        return feat

    def score(self, enc):
        return self.ans_scorer(enc)

    def forward(self, batch):
        out = self.create_input_tensors(batch)
        out['bert_enc'], _ = bert_enc, _ = self.bert(out['input_ids'], out['type_ids'], out['input_mask'], output_all_encoded_layers=False)
        scores = self.score(self.dropout(bert_enc))
        out['scores'] = self.mask_scores(scores, out['pointer_mask'])
        return out

    def mask_scores(self, scores, mask):
        invalid = 1 - mask
        scores -= invalid.unsqueeze(2).expand_as(scores).float().mul(1e20)
        return scores

    def get_top_k(self, probs, k):
        p = list(enumerate(probs.tolist()))
        p.sort(key=lambda tup: tup[1], reverse=True)
        return p[:k]

    def extract_preds(self, out, batch, top_k=20):
        scores = out['scores']
        ystart, yend = scores.split(1, dim=-1)
        pstart = F.softmax(ystart.squeeze(-1), dim=1)
        pend = F.softmax(yend.squeeze(-1), dim=1)

        preds = []
        for pstart_i, pend_i, ex in zip(pstart, pend, batch):
            top_start = self.get_top_k(pstart_i, top_k)
            top_end = self.get_top_k(pend_i, top_k)
            top_preds = []
            for s, ps in top_start:
                for e, pe in top_end:
                    if e >= s:
                        top_preds.append((s, e, ps*pe))
            top_preds = sorted(top_preds, key=lambda tup: tup[-1], reverse=True)[:top_k]
            top_answers = [(detokenize(ex['feat']['inp'][s:e+1]), s, e, p) for s, e, p in top_preds]
            top_ans, top_s, top_e, top_p = top_answers[0]
            preds.append({
                'utterance_id': ex['utterance_id'],
                'top_k': top_answers,
                'answer': top_ans,
                'spans': [(top_s, top_e)],
                'retrieve_span': 0,
            })
        return preds

    def compute_loss(self, out, batch):
        scores = out['scores']
        ystart, yend = scores.split(1, dim=-1)

        gstart = torch.tensor([e['feat']['answer_start'] for e in batch], dtype=torch.long, device=self.device)
        lstart = F.cross_entropy(ystart.squeeze(-1), gstart)

        gend = torch.tensor([e['feat']['answer_end'] for e in batch], dtype=torch.long, device=self.device)
        lend = F.cross_entropy(yend.squeeze(-1), gend)
        return {'start': lstart, 'end': lend}

    def compute_metrics(self, preds, data):
        preds = [{'utterance_id': p['utterance_id'], 'answer': p['top_k'][0][0]} for p in preds]
        return compute_metrics(preds, data)

    def run_pred(self, dev):
        preds = []
        self.eval()
        for i in trange(0, len(dev), self.args.dev_batch, desc='batch'):
            batch = dev[i:i+self.args.dev_batch]
            out = self(batch)
            preds += self.extract_preds(out, batch)
        return preds

    def run_train(self, train, dev):
        if not os.path.isdir(self.args.dsave):
            os.makedirs(self.args.dsave)

        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(self.args.dsave, 'train.log'))
        fh.setLevel(logging.CRITICAL)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.CRITICAL)
        logger.addHandler(ch)

        num_train_steps = int(len(train) / self.args.train_batch * self.args.epoch)

        # remove pooler
        param_optimizer = list(self.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

        optimizer = BertAdam(optimizer_grouped_parameters, lr=self.args.learning_rate, warmup=self.args.warmup, t_total=num_train_steps)

        print('num_train', len(train))
        print('num_dev', len(dev))

        global_step = 0
        best_metrics = {self.args.early_stop: -float('inf')}
        for epoch in trange(self.args.epoch, desc='epoch'):
            self.epoch = epoch
            train = train[:]
            np.random.shuffle(train)

            stats = defaultdict(list)
            preds = []
            self.train()
            for i in trange(0, len(train), self.args.train_batch, desc='batch'):
                batch = train[i:i+self.args.train_batch]
                out = self(batch)
                pred = self.extract_preds(out, batch)
                loss = self.compute_loss(out, batch)

                sum(loss.values()).backward()
                lr_this_step = self.args.learning_rate * warmup_linear(global_step/num_train_steps, self.args.warmup)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                for k, v in loss.items():
                    stats['loss_' + k].append(v.item())
                preds += pred
            train_metrics = {k: sum(v) / len(v) for k, v in stats.items()}
            train_metrics.update(self.compute_metrics(preds, train))

            stats = defaultdict(list)
            preds = self.run_pred(dev)
            dev_metrics = {k: sum(v) / len(v) for k, v in stats.items()}
            dev_metrics.update(self.compute_metrics(preds, dev))

            metrics = {'epoch': epoch}
            metrics.update({'train_' + k: v for k, v in train_metrics.items()})
            metrics.update({'dev_' + k: v for k, v in dev_metrics.items()})
            logger.critical(pformat(metrics))

            if metrics[self.args.early_stop] > best_metrics[self.args.early_stop]:
                logger.critical('Found new best! Saving to ' + self.args.dsave)
                best_metrics = metrics
                self.save(best_metrics, self.args.dsave, self.args.early_stop)
                with open(os.path.join(self.args.dsave, 'dev.preds.json'), 'wt') as f:
                    json.dump(preds, f, indent=2)

        logger.critical('Best dev')
        logger.critical(pformat(best_metrics))
