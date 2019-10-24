#!/usr/bin/env python
import os
import editdistance
import torch
import string
import revtok
import json
from tempfile import NamedTemporaryFile
from tqdm import tqdm
from pprint import pprint
from collections import defaultdict
from pytorch_pretrained_bert.tokenization import BertTokenizer


FORCE = True
MAX_LEN = 300
BERT_MODEL = 'cache/bert-base-uncased.tar.gz'
BERT_VOCAB = 'cache/bert-base-uncased-vocab.txt'
LOWERCASE = True
tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=LOWERCASE, cache_dir=None)
MATCH_IGNORE = {'do', 'have', '?'}
SPAN_IGNORE = set(string.punctuation)
CLASSES = ['yes', 'no', 'irrelevant', 'more']


nlp = None


def tokenize(doc):
    if not doc.strip():
        return []
    tokens = []
    for i, t in enumerate(revtok.tokenize(doc)):
        subtokens = tokenizer.tokenize(t.strip())
        for st in subtokens:
            tokens.append({
                'orig': t,
                'sub': st,
                'orig_id': i,
            })
    return tokens


def convert_to_ids(tokens):
    return tokenizer.convert_tokens_to_ids([t['sub'] for t in tokens])


def filter_answer(answer):
    return detokenize([a for a in answer if a['orig'] not in MATCH_IGNORE])


def filter_chunk(answer):
    return detokenize([a for a in answer if a['orig'] not in MATCH_IGNORE])


def detokenize(tokens):
    words = []
    for i, t in enumerate(tokens):
        if t['orig_id'] is None or (i and t['orig_id'] == tokens[i-1]['orig_id']):
            continue
        else:
            words.append(t['orig'])
    return revtok.detokenize(words)


def make_tag(tag):
    return {'orig': tag, 'sub': tag, 'orig_id': tag}


def compute_metrics(preds, data):
    import evaluator
    with NamedTemporaryFile('w') as fp, NamedTemporaryFile('w') as fg:
        json.dump(preds, fp)
        fp.flush()
        json.dump([{'utterance_id': e['utterance_id'], 'answer': e['answer']} for e in data], fg)
        fg.flush()
        results = evaluator.evaluate(fg.name, fp.name, mode='combined')
        results['combined'] = results['macro_accuracy'] * results['bleu_4']
        return results


def get_span(context, answer):
    answer = filter_answer(answer)
    best, best_score = None, float('inf')
    stop = False
    for i in range(len(context)):
        if stop:
            break
        for j in range(i, len(context)):
            chunk = filter_chunk(context[i:j+1])
            if '\n' in chunk or '*' in chunk:
                continue
            score = editdistance.eval(answer, chunk)
            if score < best_score or (score == best_score and j-i < best[1]-best[0]):
                best, best_score = (i, j), score
            if chunk == answer:
                stop = True
                break
    s, e = best
    while not context[s]['orig'].strip() or context[s]['orig'] in SPAN_IGNORE:
        s += 1
    while not context[e]['orig'].strip() or context[s]['orig'] in SPAN_IGNORE:
        e -= 1
    return s, e


def get_bullets(context):
    indices = [i for i, c in enumerate(context) if c['sub'] == '*']
    pairs = list(zip(indices, indices[1:] + [len(context)]))
    cleaned = []
    for s, e in pairs:
        while not context[e-1]['sub'].strip():
            e -= 1
        while not context[s]['sub'].strip() or context[s]['sub'] == '*':
            s += 1
        if e - s > 2 and e - 2 < 45:
            cleaned.append((s, e-1))
    return cleaned


def extract_clauses(data, tokenizer):
    snippet = data['snippet']
    t_snippet = tokenize(snippet)
    questions = data['questions']
    t_questions = [tokenize(q) for q in questions]

    spans = [get_span(t_snippet, q) for q in t_questions]
    bullets = get_bullets(t_snippet)
    all_spans = spans + bullets
    coverage = [False] * len(t_snippet)
    sorted_by_len = sorted(all_spans,  key=lambda tup: tup[1] - tup[0], reverse=True)

    ok = []
    for s, e in sorted_by_len:
        if not all(coverage[s:e+1]):
            for i in range(s, e+1):
                coverage[i] = True
            ok.append((s, e))
    ok.sort(key=lambda tup: tup[0])

    match = {}
    match_text = {}
    clauses = [None] * len(ok)
    for q, tq in zip(questions, t_questions):
        best_score = float('inf')
        best = None
        for i, (s, e) in enumerate(ok):
            score = editdistance.eval(detokenize(tq), detokenize(t_snippet[s:e+1]))
            if score < best_score:
                best_score, best = score, i
                clauses[i] = tq
        match[q] = best
        s, e = ok[best]
        match_text[q] = detokenize(t_snippet[s:e+1])

    return {'questions': {q: tq for q, tq in zip(questions, t_questions)}, 'snippet': snippet, 't_snippet': t_snippet, 'spans': ok, 'match': match, 'match_text': match_text, 'clauses': clauses}


if __name__ == '__main__':
    for split in ['dev', 'train']:
        fsplit = 'sharc_train' if split == 'train' else 'sharc_dev'
        with open('sharc/json/{}.json'.format(fsplit)) as f:
            data = json.load(f)
            ftree = 'sharc/trees_{}.json'.format(split)
            if not os.path.isfile(ftree) or FORCE:
                tasks = {}
                for ex in data:
                    for h in ex['evidence']:
                        if 'followup_question' in h:
                            h['follow_up_question'] = h['followup_question']
                            h['follow_up_answer'] = h['followup_answer']
                            del h['followup_question']
                            del h['followup_answer']
                    if ex['tree_id'] in tasks:
                        task = tasks[ex['tree_id']]
                    else:
                        task = tasks[ex['tree_id']] = {'snippet': ex['snippet'], 'questions': set()}
                    for h in ex['history'] + ex['evidence']:
                        task['questions'].add(h['follow_up_question'])
                    if ex['answer'].lower() not in {'yes', 'no', 'irrelevant'}:
                        task['questions'].add(ex['answer'])
                keys = sorted(list(tasks.keys()))
                vals = [extract_clauses(tasks[k], tokenizer) for k in tqdm(keys)]
                mapping = {k: v for k, v in zip(keys, vals)}
                with open(ftree, 'wt') as f:
                    json.dump(mapping, f, indent=2)
            else:
                with open(ftree) as f:
                    mapping = json.load(f)
            fproc = 'sharc/proc_{}.pt'.format(split)
            if not os.path.isfile(fproc) or FORCE:
                stats = defaultdict(list)
                for ex in data:
                    ex_answer = ex['answer'].lower()
                    m = mapping[ex['tree_id']]
                    ex['ann'] = a = {
                        'snippet': m['t_snippet'],
                        'clauses': m['clauses'],
                        'question': tokenize(ex['question']),
                        'scenario': tokenize(ex['scenario']),
                        'answer': tokenize(ex['answer']),
                        'hanswer': [{'yes': 1, 'no': 0}[h['follow_up_answer'].lower()] for h in ex['history']],
                        'hquestion': [m['questions'][h['follow_up_question']] for h in ex['history']],
                        'hquestion_span': [m['match'][h['follow_up_question']] for h in ex['history']],
                        'hquestion_span_text': [m['match_text'][h['follow_up_question']] for h in ex['history']],
                        'sentailed': [m['questions'][h['follow_up_question']] for h in ex['evidence']],
                        'sentailed_span': [m['match'][h['follow_up_question']] for h in ex['evidence']],
                        'sentailed_span_text': [m['match_text'][h['follow_up_question']] for h in ex['evidence']],
                        'spans': m['spans'],
                    }
                    if ex_answer not in CLASSES:
                        a['answer_span'] = m['match'][ex['answer']]
                        a['answer_span_text'] = m['match_text'][ex['answer']]
                    else:
                        a['answer_span'] = None
                        a['answer_span_text'] = None

                    inp = [make_tag('[CLS]')] + a['question']
                    type_ids = [0] * len(inp)
                    clf_indices = {
                        'yes': len(inp) + 2,
                        'no': len(inp) + 3,
                        'irrelevant': len(inp) + 4,
                    }
                    sep = make_tag('[SEP]')
                    pointer_mask = [0] * len(inp)
                    inp += [
                        sep,
                        make_tag('classes'),
                        make_tag('yes'),
                        make_tag('no'),
                        make_tag('irrelevant'),
                        sep,
                        make_tag('document'),
                    ]
                    pointer_mask += [0, 0, 1, 1, 1, 0, 0]
                    offset = len(inp)
                    spans = [(s+offset, e+offset) for s, e in a['spans']]
                    inp += a['snippet']
                    pointer_mask += [1] * len(a['snippet'])  # where can the answer pointer land
                    inp += [sep]
                    start = len(inp)
                    inp += [make_tag('scenario')] + a['scenario'] + [sep]
                    end = len(inp)
                    scen_offsets = start, end
                    inp += [make_tag('history')]
                    hist_offsets = []
                    for hq, ha in zip(a['hquestion'], a['hanswer']):
                        start = len(inp)
                        inp += [make_tag('question')] + hq + [make_tag('answer'), [make_tag('yes'), make_tag('no')][ha]]
                        end = len(inp)
                        hist_offsets.append((start, end))
                    inp += [sep]
                    type_ids += [1] * (len(inp) - len(type_ids))
                    input_ids = convert_to_ids(inp)
                    input_mask = [1] * len(inp)
                    pointer_mask += [0] * (len(inp) - len(pointer_mask))

                    if ex_answer in CLASSES:
                        start = clf_indices[ex_answer]
                        end = start
                        clf = CLASSES.index(ex_answer)
                        answer_span = -1
                    else:
                        answer_span = a['answer_span']
                        start, end = spans[answer_span]
                        clf = CLASSES.index('more')

                    # for s, e in spans:
                    #     print(detokenize(inp[s:e+1]))
                    # print(detokenize(inp[start:end+1]))
                    # print(ex_answer)
                    # import pdb; pdb.set_trace()

                    if len(inp) > MAX_LEN:
                        inp = inp[:MAX_LEN]
                        input_mask = input_mask[:MAX_LEN]
                        type_ids = type_ids[:MAX_LEN]
                        input_ids = input_ids[:MAX_LEN]
                        pointer_mask = pointer_mask[:MAX_LEN]
                    pad = make_tag('pad')
                    while len(inp) < MAX_LEN:
                        inp.append(pad)
                        input_mask.append(0)
                        type_ids.append(0)
                        input_ids.append(0)
                        pointer_mask.append(0)

                    assert len(inp) == len(input_mask) == len(type_ids) == len(input_ids)

                    ex['feat'] = {
                        'inp': inp,
                        'input_ids': torch.LongTensor(input_ids),
                        'type_ids': torch.LongTensor(type_ids),
                        'input_mask': torch.LongTensor(input_mask),
                        'pointer_mask': torch.LongTensor(pointer_mask),
                        'spans': spans,
                        'hanswer': a['hanswer'],
                        'hquestion_span': torch.LongTensor(a['hquestion_span']),
                        'sentailed_span': torch.LongTensor(a['sentailed_span']),
                        'answer_start': start,
                        'answer_end': end,
                        'answer_class': clf,
                        'answer_span': answer_span,
                        'snippet_offset': offset,
                        'scen_offsets': scen_offsets,
                        'hist_offsets': hist_offsets,
                    }

                    stats['snippet_len'].append(len(ex['ann']['snippet']))
                    stats['scenario_len'].append(len(ex['ann']['scenario']))
                    stats['history_len'].append(sum([len(q) + 3 for q in ex['ann']['hquestion']]))
                    stats['question_len'].append(len(ex['ann']['question']))
                    stats['inp_len'].append(sum(input_mask))
                for k, v in sorted(list(stats.items()), key=lambda tup: tup[0]):
                    print(k)
                    print('mean: {}'.format(sum(v) / len(v)))
                    print('min: {}'.format(min(v)))
                    print('max: {}'.format(max(v)))
                preds = [{'utterance_id': e['utterance_id'], 'answer': detokenize(e['feat']['inp'][e['feat']['answer_start']:e['feat']['answer_end']+1])} for e in data]
                pprint(compute_metrics(preds, data))
                torch.save(data, fproc)
