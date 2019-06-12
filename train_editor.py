import torch
import random
import numpy as np
from argparse import ArgumentParser
from editor_model.base import Module
from pprint import pprint


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_batch', default=10, type=int)
    parser.add_argument('--dev_batch', default=5, type=int)
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--keep', default=2, type=int)
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--dropout', default=0.4, type=float)
    parser.add_argument('--warmup', default=0.1, type=float)
    parser.add_argument('--thresh', default=0.5, type=float)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dsave', default='editor_save/{}')
    parser.add_argument('--model', default='double')
    parser.add_argument('--prefix', default='default')
    parser.add_argument('--early_stop', default='dev_f1')
    parser.add_argument('--bert_hidden_size', default=768, type=int)
    parser.add_argument('--bert_model', default='bert-base-uncased')
    parser.add_argument('--data', default='sharc/editor_disjoint')
    parser.add_argument('--resume', default='')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    args.dsave = args.dsave.format(args.prefix + '-' + args.model)
    # if args.model != 'base':
    #     args.dsave += '/{}/{}'.format(args.loss_span_weight, args.loss_editor_weight)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    limit = 10 if args.debug else None
    data = {k: torch.load('{}/proc_{}.pt'.format(args.data, k))[:limit] for k in ['dev', 'train']}

    if args.resume:
        print('resuming model from ' + args.resume)
        model = Module.load(args.resume)
    else:
        print('instanting model')
        model = Module.load_module(args.model)(args)

    model.to(model.device)

    if args.test:
        preds = model.run_pred(data['dev'])
        metrics = model.compute_metrics(preds, data['dev'])
        pprint(metrics)
    else:
        model.run_train(data['train'], data['dev'])
