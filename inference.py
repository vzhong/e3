import os
import torch
import json
from pprint import pprint
from argparse import ArgumentParser
from model.base import Module
from preprocess_sharc import tokenize, make_tag, convert_to_ids, MAX_LEN, compute_metrics
from editor_model.base import Module as EditorModule
from preprocess_editor import trim_span


def preprocess(data):
    for ex in data:
        ex['ann'] = a = {
            'snippet': tokenize(ex['snippet']),
            'question': tokenize(ex['question']),
            'scenario': tokenize(ex['scenario']),
            'hanswer': [{'yes': 1, 'no': 0}[h['follow_up_answer'].lower()] for h in ex['history']],
            'hquestion': [tokenize(h['follow_up_question']) for h in ex['history']],
        }
        inp = [make_tag('[CLS]')] + a['question']
        type_ids = [0] * len(inp)
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
        snippet_start = len(inp)
        offset = len(inp)
        inp += a['snippet']
        snippet_end = len(inp)
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
            'snippet_start': snippet_start,
            'snippet_end': snippet_end,
            'input_ids': torch.LongTensor(input_ids),
            'type_ids': torch.LongTensor(type_ids),
            'input_mask': torch.LongTensor(input_mask),
            'pointer_mask': torch.LongTensor(pointer_mask),
            'hanswer': a['hanswer'],
            'snippet_offset': offset,
            'scen_offsets': scen_offsets,
            'hist_offsets': hist_offsets,
        }
    return data


def preprocess_editor(orig_data, preds):
    data = []
    for orig_ex, pred in zip(orig_data, preds):
        if pred['answer'].lower() not in {'yes', 'no', 'irrelevant'}:
            s, e = pred['spans'][pred['retrieve_span']]
            sstart = orig_ex['feat']['snippet_start']
            send = orig_ex['feat']['snippet_end']
            s -= sstart
            e -= sstart
            if s < 0 or e < 0:
                continue
            snippet = orig_ex['feat']['inp'][sstart:send]
            s, e = trim_span(snippet, (s, e))
            if e >= s:
                inp = [make_tag('[CLS]')] + snippet[s:e+1] + [make_tag('[SEP]')]
                # account for prepended tokens
                new_s, new_e = s + len(inp), e + len(inp)
                inp += snippet + [make_tag('[SEP]')]
                type_ids = [0] + [0] * (e+1-s) + [1] * (len(snippet) + 2)
                inp_ids = convert_to_ids(inp)
                inp_mask = [1] * len(inp)

                assert len(type_ids) == len(inp) == len(inp_ids)

                while len(inp_ids) < MAX_LEN:
                    inp.append(make_tag('pad'))
                    inp_ids.append(0)
                    inp_mask.append(0)
                    type_ids.append(0)

                if len(inp_ids) > MAX_LEN:
                    inp = inp[:MAX_LEN]
                    inp_ids = inp_ids[:MAX_LEN]
                    inp_mask = inp_mask[:MAX_LEN]
                    inp_mask[-1] = make_tag('[SEP]')
                    type_ids = type_ids[:MAX_LEN]

                ex = {
                    'utterance_id': orig_ex['utterance_id'],
                    'span': (new_s, new_e),
                    'inp': inp,
                    'type_ids': torch.tensor(type_ids, dtype=torch.long),
                    'inp_ids': torch.tensor(inp_ids, dtype=torch.long),
                    'inp_mask': torch.tensor(inp_mask, dtype=torch.long),
                }
                data.append(ex)
    return data


def merge_edits(preds, edits):
    # note: this happens in place
    edits = {p['utterance_id']: p for p in edits}
    for p in preds:
        p['orig_answer'] = p['answer']
        if p['utterance_id'] in edits:
            p['answer'] = p['edit_answer'] = edits[p['utterance_id']]['answer']
    return preds


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--retrieval', required=True, help='retrieval model to use')
    parser.add_argument('--editor', help='editor model to use (optional)')
    parser.add_argument('--fin', default='sharc/json/sharc_dev.json', help='input data file')
    parser.add_argument('--dout', default=os.getcwd(), help='directory to store output files')
    parser.add_argument('--data', default='sharc/editor_disjoint', help='editor data')
    parser.add_argument('--verify', action='store_true', help='run evaluation')
    parser.add_argument('--force', action='store_true', help='overwrite retrieval predictions')
    args = parser.parse_args()

    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    with open(args.fin) as f:
        raw = json.load(f)

    print('preprocessing data')
    data = preprocess(raw)

    fretrieval = os.path.join(args.dout, 'retrieval_preds.json')
    if os.path.isfile(fretrieval) and not args.force:
        print('loading {}'.format(fretrieval))
        with open(fretrieval) as f:
            retrieval_preds = json.load(f)
    else:
        print('resuming retrieval from ' + args.retrieval)
        retrieval = Module.load(args.retrieval)
        retrieval.to(retrieval.device)
        retrieval_preds = retrieval.run_pred(data)
        with open(fretrieval, 'wt') as f:
            json.dump(retrieval_preds, f, indent=2)

    if args.verify:
        pprint(compute_metrics(retrieval_preds, raw))

    if args.editor:
        editor_data = preprocess_editor(data, retrieval_preds)
        editor = EditorModule.load(args.editor, override_args={'data': args.data})
        editor.to(editor.device)
        raw_editor_preds = editor.run_pred(editor_data)
        editor_preds = merge_edits(retrieval_preds, raw_editor_preds)

        with open(os.path.join(args.dout, 'editor_preds.json'), 'wt') as f:
            json.dump(editor_preds, f, indent=2)

        if args.verify:
            pprint(compute_metrics(editor_preds, raw))
