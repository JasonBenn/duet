# coding: utf-8
"""
git clone https://github.com/cybertronai/transformer-xl.git
cd transformer-xl/pytorch
source activate pytorch_p36

# Match eval parameters from paper.
# https://github.com/kimiyoung/transformer-xl/blob/master/tf/scripts/wt103_large_tpu.sh
python eval.py --data=data/wikitext-103 --dataset=wt103 --batch_size=8 --tgt_len=128 --clamp_len=1000 --mem_len=1600 --work_dir=/ncluster/runs/ben-batch-sched-slow2.01/
# new dataset
python eval.py --data=data/wikiextracted/ --dataset=wiki --batch_size=8 --tgt_len=128 --clamp_len=1000 --mem_len=1600 --work_dir=/ncluster/runs.new/ben-txl-large-adam.05 --bpe

"""
import argparse
import math
import os

import torch
import tqdm

from data_utils import get_lm_corpus
from utils.exp_utils import get_logger

def main():
    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
    parser.add_argument('--data', type=str, default='../data/wikitext-103',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='wt103',
                        choices=['wt103', 'lm1b', 'enwik8', 'text8', 'wt2', 'wiki'],
                        help='dataset name')
    parser.add_argument('--split', type=str, default='all',
                        choices=['all', 'valid', 'test'],
                        help='which split to evaluate')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size')
    parser.add_argument('--tgt_len', type=int, default=5,
                        help='number of tokens to predict')
    parser.add_argument('--ext_len', type=int, default=0,
                        help='length of the extended context')
    parser.add_argument('--mem_len', type=int, default=0,
                        help='length of the retained previous heads')
    parser.add_argument('--clamp_len', type=int, default=-1,
                        help='max positional embedding index')
    parser.add_argument('--work_dir', type=str, required=True,
                        help='path to the work_dir')
    parser.add_argument('--no_log', action='store_true',
                        help='do not log the eval result')
    parser.add_argument('--same_length', action='store_true',
                        help='set same length attention with masking')
    parser.add_argument('--bpe', action='store_true', default=False,
                        help='Use BPE instead of traditional vocabulary.')

    args = parser.parse_args()
    assert args.ext_len >= 0, 'extended context length must be non-negative'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get logger
    logging = get_logger(os.path.join(args.work_dir, 'eval-log.txt'),
                         log_=not args.no_log)

    # Load dataset
    corpus = get_lm_corpus(args.data, args.dataset, use_bpe=args.bpe)
    ntokens = len(corpus.vocab)

    # Load the best saved model.
    with open(os.path.join(args.work_dir, 'model-best.pt'), 'rb') as f:
        model = torch.load(f)

    model_tokens = model.n_token if hasattr(model, 'n_token') else model.module.n_token
    assert model_tokens == ntokens, 'vocab size mismatch, did you mean `--bpe`?'
    model = model.to(device)

    logging('Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {}'.format(
        args.batch_size, args.tgt_len, args.ext_len, args.mem_len, args.clamp_len))

    if hasattr(model, 'reset_length'):
        model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    else:
        model.module.reset_length(args.tgt_len, args.ext_len, args.mem_len)

    if args.clamp_len > 0:
        model.clamp_len = args.clamp_len
    if args.same_length:
        model.same_length = True

    # Run on test data.
    for split in ('valid', 'test'):
        if args.split in (split, 'all'):
            it = corpus.get_iterator(split, args.batch_size, args.tgt_len,
                device=device, ext_len=args.ext_len)
            logging(format_log(args, *evaluate(model, it, split), split))


def evaluate(model, eval_iter, label: str, max_eval_steps: int = 0):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_len, total_loss = 0, 0.
    with torch.no_grad():
        mems = tuple()
        bar = tqdm.tqdm(eval_iter, leave=False)
        for i, (data, target, seq_len) in enumerate(bar):
            if max_eval_steps > 0 and i >= max_eval_steps:
                break
            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            total_loss += seq_len * loss.item()
            total_len += seq_len
            bar.set_description(f'{label} loss: {total_loss / total_len:.2f}')
    return total_loss, total_len


def format_log(args, loss, total, split):
    if args.dataset in ['enwik8', 'text8']:
        special = f'bpc {loss / math.log(2):9.5f}'
    else:
        special = f'ppl {math.exp(loss/total):9.3f}'
    return f'| {split} loss\t{loss/total:5.4f} | {split}\t{special}\tloss {loss:.1f}\ttokens {total}\n'



if __name__ == '__main__':
    main()
