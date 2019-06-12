import os
import torch
import importlib
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from model.base import Module as Base
from model.editor import Decoder
from metric import compute_f1
from preprocess_sharc import detokenize


class Module(Base):

    def __init__(self, args, vocab=None):
        super().__init__(args)
        self.denc = self.args.bert_hidden_size
        vocab = vocab or torch.load(os.path.join(args.data, 'vocab.pt'))
        self.emb = vocab['emb']
        self.vocab = vocab['vocab']
        self.decoder = Decoder(self.denc, self.emb, dropout=self.args.dropout)

    @classmethod
    def load_module(cls, name):
        return importlib.import_module('editor_model.{}'.format(name)).Module

    def create_input_tensors(self, batch):
        feat = {
            k: torch.stack([e[k] for e in batch], dim=0).to(self.device)
            for k in ['inp_ids', 'type_ids', 'inp_mask']
        }
        feat['inp_mask'] = feat['inp_mask'].float()
        feat['out_vids'] = pad_sequence([e['out_vids'] for e in batch], batch_first=True, padding_value=-1).to(self.device) if self.training else None
        return feat

    def forward(self, batch):
        out = self.create_input_tensors(batch)
        out['bert_enc'], _ = bert_enc, _ = self.bert(out['inp_ids'], out['type_ids'], out['inp_mask'], output_all_encoded_layers=False)
        out['dec'] = self.decoder.forward(bert_enc, out['inp_mask'], out['out_vids'], max_decode_len=30)
        return out

    def extract_preds(self, out, batch):
        preds = []
        for pred, ex in zip(out['dec'].max(2)[1].tolist(), batch):
            pred = self.vocab.index2word(pred)
            if 'eos' in pred:
                pred = pred[:pred.index('eos')]
            preds.append({
                'utterance_id': ex['utterance_id'],
                'answer': ' '.join(pred),
            })
        return preds

    def compute_loss(self, out, batch):
        return {'dec': F.cross_entropy(out['dec'].view(-1, len(self.vocab)), out['out_vids'].view(-1), ignore_index=-1)}

    def compute_metrics(self, preds, batch):
        f1s = [compute_f1(p['answer'], detokenize(e['question'])) for p, e in zip(preds, batch)]
        return {'f1': sum(f1s) / len(f1s)}
