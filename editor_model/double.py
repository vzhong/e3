from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from editor_model.base import Module as Base
from model.editor import Decoder
from preprocess_sharc import detokenize


class Module(Base):

    def __init__(self, args, vocab=None):
        super().__init__(args, vocab=vocab)
        self.decoder_after = Decoder(self.denc, self.emb, dropout=self.args.dropout)

    def create_input_tensors(self, batch):
        feat = super().create_input_tensors(batch)
        feat['before_vids'] = pad_sequence([e['before_vids'] for e in batch], batch_first=True, padding_value=-1).to(self.device) if self.training else None
        feat['after_vids'] = pad_sequence([e['after_vids'] for e in batch], batch_first=True, padding_value=-1).to(self.device) if self.training else None
        return feat

    def forward(self, batch):
        out = self.create_input_tensors(batch)
        out['bert_enc'], _ = bert_enc, _ = self.bert(out['inp_ids'], out['type_ids'], out['inp_mask'], output_all_encoded_layers=False)
        out['before'] = self.decoder.forward(bert_enc, out['inp_mask'], out['before_vids'], max_decode_len=10)
        out['after'] = self.decoder_after.forward(bert_enc, out['inp_mask'], out['after_vids'], max_decode_len=10)
        return out

    def extract_preds(self, out, batch):
        preds = []
        for before, after, ex in zip(
                out['before'].max(2)[1].tolist(),
                out['after'].max(2)[1].tolist(),
                batch):
            before = self.vocab.index2word(before)
            if 'eos' in before:
                before = before[:before.index('eos')]
            after = self.vocab.index2word(after)
            if 'eos' in after:
                after = after[:after.index('eos')]
            s, e = ex['span']
            middle = detokenize(ex['inp'][s:e+1])
            preds.append({
                'utterance_id': ex['utterance_id'],
                'answer': '{} {} {}'.format(' '.join(before), middle, ' '.join(after)),
            })
        return preds

    def compute_loss(self, out, batch):
        return {
            'before': F.cross_entropy(out['before'].view(-1, len(self.vocab)), out['before_vids'].view(-1), ignore_index=-1),
            'after': F.cross_entropy(out['after'].view(-1, len(self.vocab)), out['after_vids'].view(-1), ignore_index=-1),
        }
