#!/usr/bin/env python
import os
import json
import torch
import embeddings
import stanfordnlp
from metric import compute_f1
from vocab import Vocab
from tqdm import tqdm
from preprocess_sharc import detokenize, tokenizer, make_tag


def get_orig(tokens):
    words = []
    for i, t in enumerate(tokens):
        if t['orig_id'] is None or (i and t['orig_id'] == tokens[i-1]['orig_id']):
            continue
        else:
            words.append(t['orig'].strip().lower())
    return words


nlp = None


def trim_span(snippet, span):
    global nlp
    if nlp is None:
        nlp = stanfordnlp.Pipeline(processors='tokenize,pos,lemma', models_dir='cache')
    bad_pos = {'DET', 'ADP', '#', 'AUX', 'SCONJ', 'CCONJ', 'PUNCT'}
    s, e = span
    words = nlp(' '.join([t['orig'] for t in snippet[s:e+1]])).sentences[0].words
    while words and words[0].upos in bad_pos:
        words = words[1:]
        s += 1
    while words and words[-1].upos in bad_pos:
        words.pop()
        e -= 1
    return s, e


def create_split(trees, vocab, max_len=300, train=True):
    split = []
    keys = sorted(list(trees.keys()))
    for k in tqdm(keys):
        v = trees[k]
        snippet = v['t_snippet']
        for q_str, q_tok in v['questions'].items():
            span = v['spans'][v['match'][q_str]]
            # trim the span a bit to facilitate editing
            s, e = trim_span(snippet, span)
            if e >= s:
                inp = [make_tag('[CLS]')] + snippet[s:e+1] + [make_tag('[SEP]')]
                # account for prepended tokens
                new_s, new_e = s + len(inp), e + len(inp)
                inp += snippet + [make_tag('[SEP]')]
                type_ids = [0] + [0] * (e+1-s) + [1] * (len(snippet) + 2)
                inp_ids = tokenizer.convert_tokens_to_ids([t['sub'] for t in inp])
                inp_mask = [1] * len(inp)

                assert len(type_ids) == len(inp) == len(inp_ids)

                while len(inp_ids) < max_len:
                    inp.append(make_tag('pad'))
                    inp_ids.append(0)
                    inp_mask.append(0)
                    type_ids.append(0)

                if len(inp_ids) > max_len:
                    inp = inp[:max_len]
                    inp_ids = inp_ids[:max_len]
                    inp_mask = inp_mask[:max_len]
                    inp_mask[-1] = make_tag('[SEP]')
                    type_ids = type_ids[:max_len]

                out = get_orig(q_tok)
                if train:
                    out_vids = torch.tensor(vocab.word2index(out + ['eos'], train=train), dtype=torch.long)
                else:
                    out_vids = None

                ex = {
                    'utterance_id': len(split),
                    'question': q_tok,
                    'span': (new_s, new_e),
                    'inp': inp,
                    'type_ids': torch.tensor(type_ids, dtype=torch.long),
                    'inp_ids': torch.tensor(inp_ids, dtype=torch.long),
                    'inp_mask': torch.tensor(inp_mask, dtype=torch.long),
                    'out': out,
                    'out_vids': out_vids,
                }
                split.append(ex)
    return split


def segment(ex, vocab, threshold=0.25):
    s, e = ex['span']
    span = ex['inp'][s:e+1]
    span_str = detokenize(span)
    ques = ex['question']

    best_i, best_j, best_score = None, None, -1
    for i in range(len(ques)):
        for j in range(i, len(ques)):
            chunk = detokenize(ques[i:j+1])
            score = compute_f1(span_str, chunk)
            if score > best_score:
                best_score, best_i, best_j = score, i, j
    if best_score > threshold:
        before = ex['question'][:best_i]
        after = ex['question'][best_j+1:]
        ret = {
            'before': get_orig(before),
            'after': get_orig(after),
        }
        ret.update({
            k + '_vids': torch.tensor(vocab.word2index(v + ['eos']), dtype=torch.long)
            for k, v in ret.items()
        })
        return ret
    else:
        return None


if __name__ == '__main__':
    import joblib
    vocab = Vocab()
    with open('sharc/trees_train.json') as f:
        train_trees = json.load(f)
    with open('sharc/trees_dev.json') as f:
        dev_trees = json.load(f)
    dout = 'sharc/editor_disjoint'
    if not os.path.isdir(dout):
        os.makedirs(dout)

    print('Flattening train')
    train = create_split(train_trees, vocab)
    print('Flattening dev')
    dev = create_split(dev_trees, vocab)

    par = joblib.Parallel(12)
    print('Segmenting train')
    train_ba = par(joblib.delayed(segment)(ex, vocab) for ex in tqdm(train))

    train_filtered = []
    for ex, ba in zip(train, train_ba):
        if ba:
            ex.update(ba)
            train_filtered.append(ex)

    print('filtered train from {} to {}'.format(len(train), len(train_filtered)))
    print('vocab size {}'.format(len(vocab)))

    emb = embeddings.ConcatEmbedding([embeddings.GloveEmbedding(), embeddings.KazumaCharEmbedding()], default='zero')
    mat = torch.Tensor([emb.emb(w) for w in vocab._index2word])
    torch.save({'vocab': vocab, 'emb': mat}, dout + '/vocab.pt')
    torch.save(train_filtered, dout + '/proc_train.pt')
    torch.save(dev, dout + '/proc_dev.pt')
