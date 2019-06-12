import torch
import random
import numpy as np
from argparse import ArgumentParser
from model.base import Module
from pprint import pprint


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_batch', default=10, type=int, help='training batch size')
    parser.add_argument('--dev_batch', default=5, type=int, help='dev batch size')
    parser.add_argument('--epoch', default=5, type=int, help='number of epochs')
    parser.add_argument('--keep', default=2, type=int, help='number of model saves to keep')
    parser.add_argument('--seed', default=3, type=int, help='random seed')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0.35, type=float, help='dropout rate')
    parser.add_argument('--warmup', default=0.1, type=float, help='optimizer warmup')
    parser.add_argument('--thresh', default=0.5, type=float, help='rule extraction threshold')
    parser.add_argument('--loss_span_weight', default=400., type=float, help='span loss weight')
    parser.add_argument('--loss_editor_weight', default=1., type=float, help='editor loss weight')
    parser.add_argument('--debug', action='store_true', help='debug flag to load less data')
    parser.add_argument('--dsave', default='save/{}', help='save directory')
    parser.add_argument('--model', default='entail', help='model to use')
    parser.add_argument('--early_stop', default='dev_combined', help='early stopping metric')
    parser.add_argument('--bert_hidden_size', default=768, type=int, help='hidden size for the bert model')
    parser.add_argument('--data', default='sharc', help='directory for data')
    parser.add_argument('--prefix', default='default', help='prefix for experiment name')
    parser.add_argument('--resume', default='', help='model .pt file')
    parser.add_argument('--test', action='store_true', help='only run evaluation')

    args = parser.parse_args()
    args.dsave = args.dsave.format(args.prefix+'-'+args.model)

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
