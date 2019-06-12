#!/usr/bin/env python
import json
import os
import torch
import tabulate
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--editor', action='store_true')
    parser.add_argument('--dsave', default='save')
    parser.add_argument('--force', '-f', action='store_true')
    args = parser.parse_args()

    rows = []
    keys = ['epoch', 'dev_combined', 'dev_macro_accuracy', 'dev_micro_accuracy', 'dev_bleu_1', 'dev_bleu_4', 'dev_span_f1']
    columns = ['name', 'epoch', 'combined', 'macro', 'micro', 'bleu1', 'bleu4', 'span_f1']
    early = 'combined'
    if args.editor:
        keys = ['epoch', 'dev_f1']
        columns = ['name', 'epoch', 'f1']
        early = 'f1'
    for root, dirs, files in os.walk(args.dsave):
        if 'best.pt' in files:
            fbest = os.path.join(root, 'best.pt')
            fbest_json = os.path.join(root, 'best.json')
            if not os.path.isfile(fbest_json) or args.force:
                with open(fbest_json, 'wt') as f:
                    metrics = torch.load(fbest, map_location='cpu')['metrics']
                    json.dump(metrics, f, indent=2)
            with open(fbest_json) as f:
                metrics = json.load(f)

            rows.append([root] + [metrics.get(k, 0) for k in keys])
    rows.sort(key=lambda r: r[columns.index(early)])
    print(tabulate.tabulate(rows, headers=columns))
