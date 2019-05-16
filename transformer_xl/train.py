# coding: utf-8
#
import argparse
import collections
import datetime
import itertools
import logging
import math
import os
import sys
import time
import warnings
from collections import OrderedDict

from fp16_opt import FP16_Module, FP16_Optimizer

import numpy as np
import pytz
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import tqdm
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from lr_finder import LRFinder
from pytorch_lamb import Lamb, log_lamb_rs
from eval import evaluate

import util

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--logdir', type=str, default='/tmp/default', help="where logs and events go")
parser.add_argument('--run_name', type=str, default='txl', help="name of run")

parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8', 'wt2', 'wiki', 'wt103-normal'],
                    help='dataset name')
parser.add_argument('--n_layer', type=int, default=12,
                    help='number of total layers')
parser.add_argument('--n_head', type=int, default=10,
                    help='number of heads')
parser.add_argument('--d_head', type=int, default=50,
                    help='head dimension')
parser.add_argument('--d_embed', type=int, default=-1,
                    help='embedding dimension')
parser.add_argument('--d_model', type=int, default=500,
                    help='model dimension')
parser.add_argument('--d_inner', type=int, default=1000,
                    help='inner dimension in FF')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='global dropout rate')
parser.add_argument('--dropatt', type=float, default=0.0,
                    help='attention probability dropout rate')
parser.add_argument('--init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--emb_init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--init_range', type=float, default=0.1,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--emb_init_range', type=float, default=0.01,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--init_std', type=float, default=0.02,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--proj_init_std', type=float, default=0.01,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--optim', default='adam', type=str,
                    choices=['adam', 'sgd', 'adagrad', 'lamb'],
                    help='optimizer to use.')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument('--mom', type=float, default=0.0,
                    help='momentum for sgd')
parser.add_argument('--wd', type=float, default=0,
                    help='weight decay for adam|lamb)')
parser.add_argument('--scheduler', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant', 'finder'],
                    help='lr scheduler to use.')
parser.add_argument('--warmup_tokens', type=int, default=0,
                    help='upper epoch limit')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min', type=float, default=0.0,
                    help='minimum learning rate during annealing')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--clip_nonemb', action='store_true',
                    help='only clip the gradient of non-embedding params')
parser.add_argument('--max_tokens', type=int, default=1.8e9, help='upper epoch limit affecting LR schedule')
parser.add_argument('--batch_size', type=int, default=60,
                    help='batch size')
parser.add_argument('--tgt_len', type=int, default=70,
                    help='number of tokens to predict')
parser.add_argument('--eval_tgt_len', type=int, default=50,
                    help='number of tokens to predict for evaluation')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--not_tied', action='store_true',
                    help='do not tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--adaptive', action='store_true',
                    help='use adaptive softmax')
parser.add_argument('--div_val', type=int, default=1,
                    help='divident value for adapative input and softmax')
parser.add_argument('--pre_lnorm', action='store_true',
                    help='apply LayerNorm to the input instead of the output')
parser.add_argument('--log_interval', type=int, default=200,
                    help='logging interval in number of steps')
parser.add_argument('--retune_interval', type=int, default=5,
                    help='how often to retune parameters')
parser.add_argument('--verbose_log_steps', type=int, default=60,
                    help='do logging at every step for this many steps at the start of training')
parser.add_argument('--eval_interval', type=int, default=4000,
                    help='evaluation interval in number of steps')

parser.add_argument('--checkpoint_each_epoch', type=int, default=0,
                    help='whether to save checkpoint at each epoch')
parser.add_argument('--checkpoint', type=str, default='',
                    help='checkpoint file to use to restore training')

parser.add_argument('--restart', action='store_true',
                    help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir', type=str, default='',
                    help='restart dir')
parser.add_argument('--same_length', action='store_true',
                    help='use the same attn length for all tokens')
parser.add_argument('--attn_type', type=int, default=0,
                    help='attention type. 0 for ours, 1 for Shaw et al,'
                         '2 for Vaswani et al, 3 for Al Rfou et al.')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='use the same pos embeddings after clamp_len')
parser.add_argument('--eta_min', type=float, default=0.0,
                    help='min learning rate for cosine scheduler')
parser.add_argument('--gpu0_bsz', type=int, default=-1,
                    help='batch size on gpu 0')
parser.add_argument('--max_eval_steps', type=int, default=-1,
                    help='max eval steps')
parser.add_argument('--sample_softmax', type=int, default=-1,
                    help='number of samples in sampled softmax')
parser.add_argument('--patience', type=int, default=0,
                    help='patience')
parser.add_argument('--finetune_v2', action='store_true',
                    help='finetune v2')
parser.add_argument('--finetune_v3', action='store_true',
                    help='finetune v3')
parser.add_argument('--num_gpu', type=int, default=1,
                    help="number of gpus (used to make sure # tokens is correct)")
parser.add_argument('--bpe', action='store_true', default=False,
                    help="Use BPE instead of traditional vocabulary.")
parser.add_argument('--fp16', action='store_true',
                    help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--static_loss_scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can '
                         'improve fp16 convergence.')
parser.add_argument('--dynamic_loss_scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument'
                         ' supersedes --static-loss-scale.')

# distributed training flags
parser.add_argument('--local', action='store_true', help='Run local training instead of distrbuted.')
parser.add_argument('--dist_url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--local_rank', default=0, type=int,
                    help='Used for multi-process training. Can either be manually set ' +
                         'or automatically set by using \'python -m multiproc\'.')

# infra flags
parser.add_argument('--skip_auto_shutdown', action='store_true',
                    help='skip shutdown at the end of training or failure')
parser.add_argument('--auto_shutdown_success_delay_mins', default=10, type=int,
                    help='how long to wait until shutting down on success')
parser.add_argument('--auto_shutdown_failure_delay_mins', default=60, type=int,
                    help='how long to wait before shutting down on error')

args = parser.parse_args()
args.tied = not args.not_tied

# global variables
global_timeit_dict = OrderedDict()
global_token_count = 0
event_writer = util.NoOp()
epoch = 0
train_step = 0

local_rank = args.local_rank
global_rank = util.get_global_rank()
max_rank = util.get_world_size()

class FileLogger:
    def __init__(self, output_dir: str, global_rank: int, local_rank: int):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.logger = FileLogger.get_logger(output_dir, global_rank=global_rank, local_rank=local_rank)

    def exception(self, *args_, **kwargs):
        return self.logger.exception(*args_, **kwargs)

    @staticmethod
    def get_logger(output_dir: str, global_rank: int, local_rank: int):
        logger_ = logging.getLogger('txl training')
        logger_.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')

        vlog = logging.FileHandler(output_dir + f'/info-{global_rank}.log')
        vlog.setLevel(logging.INFO)
        vlog.setFormatter(formatter)
        logger_.addHandler(vlog)

        eventlog = logging.FileHandler(output_dir +  f'/warn-{global_rank}.log')
        eventlog.setLevel(logging.WARN)
        eventlog.setFormatter(formatter)
        logger_.addHandler(eventlog)

        time_formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(message)s')
        debuglog = logging.FileHandler(output_dir + f'/debug-{global_rank}.log')
        debuglog.setLevel(logging.DEBUG)
        debuglog.setFormatter(time_formatter)
        logger_.addHandler(debuglog)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG if local_rank == 0 else logging.WARN)
        logger_.addHandler(console)
        return logger_

    def debug(self, *args_):
        self.logger.debug(*args_)

    def warn(self, *args_):
        self.logger.warn(*args_)

    def info(self, *args_):
        self.logger.info(*args_)


class timeit:
    """Decorator to measure length of time spent in the block in millis and log
  it to TensorBoard."""

    def __init__(self, tag="", noop=False):
        self.tag = tag
        self.noop = noop

    def __enter__(self):
        if self.noop:
            return self
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_args):
        if self.noop:
            return
        self.end = time.perf_counter()
        interval_ms = 1000 * (self.end - self.start)
        global_timeit_dict.setdefault(self.tag, []).append(interval_ms)
        newtag = 'times/' + self.tag
        log_tb(newtag, interval_ms)


def log_tb(tag, val):
    """Log value to tensorboard (relies on global_token_count rather than step count to give comparable graphs across
    batch sizes)"""
    global global_token_count, event_writer
    event_writer.add_scalar(tag, val, global_token_count)


PT_TZ = pytz.timezone('America/Los_Angeles')


def current_timestamp() -> str:
    # timestamp format like 2019-04-15_11-29-51
    # correct to local timezone (PDT) if running on AWS (which is UTC)
    localtime = pytz.utc.localize(datetime.datetime.now(), is_dst=None).astimezone(PT_TZ)
    return localtime.strftime('%Y-%m-%d_%H-%M-%S')


if args.d_embed < 0:
    args.d_embed = args.d_model

assert args.ext_len >= 0, 'extended context length must be non-negative'

logger = FileLogger(args.logdir, global_rank=global_rank, local_rank=local_rank)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(args.local_rank)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###############################################################################
# Load data
###############################################################################
corpus = get_lm_corpus(args.data, args.dataset, use_bpe=args.bpe)
ntokens = len(corpus.vocab)
args.n_token = ntokens

# adaptive softmax / embedding
cutoffs, tie_projs = [], [False]
if args.adaptive:
    assert args.dataset in ['wt103', 'lm1b', 'wt2', 'wiki']
    if args.dataset in ('wt103', 'wt2', 'wiki'):
        if args.bpe:
            cutoffs = [5000, 10000, 40000]
        else:
            cutoffs = [20000, 40000, 200000]
        tie_projs += [True] * len(cutoffs)
    elif args.dataset == 'lm1b':
        cutoffs = [60000, 100000, 640000]
        tie_projs += [False] * len(cutoffs)


###############################################################################
# Build the model
###############################################################################
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)


logger.info(f"Torch version: {torch.__version__}")
logger.info('=' * 100)
for k, v in args.__dict__.items():
    logger.info(f'    - {k} : {v}')
logger.info('=' * 100)


###############################################################################
# Training code
###############################################################################


def evaluate_and_log(optimizer, eval_iter, split, train_step=-1):
    global best_val_loss
    eval_start_time = time.time()

    # Have to unwrap DDP & FP16, if using.
    def unwrap(module):
        if isinstance(module, MemTransformerLM):
            return module
        return unwrap(module.module)

    model_to_reset = unwrap(model)
    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.mem_len == 0:
        model_to_reset.reset_length(
            args.eval_tgt_len, args.ext_len + args.tgt_len - args.eval_tgt_len, args.mem_len)
    else:
        model_to_reset.reset_length(
            args.eval_tgt_len, args.ext_len, args.mem_len + args.tgt_len - args.eval_tgt_len)

    total_loss, total_len = evaluate(model, eval_iter, split, args.max_eval_steps)

    # Switch back to the training mode
    model_to_reset.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    # Log all the things.
    mean_loss = total_loss / total_len
    logger.info('-' * 100)
    log_str = (f'| Eval {train_step // args.eval_interval:3d} at step {train_step:>8d} | ' +
               f'time: {time.time() - eval_start_time:5.2f}s ' +
               f'| {split} loss {mean_loss:5.2f}')
    if args.dataset in ['enwik8', 'text8']:
        log_str += f' | bpc {mean_loss / math.log(2):9.5f}'
    else:
        log_str += f' | {split} ppl {math.exp(mean_loss):9.3f}'
    logger.info(log_str)
    logger.info('-' * 100)
    log_tb(f'learning/{split}_loss', mean_loss)
    log_tb(f'learning/{split}_ppl', math.exp(mean_loss))

    # Update checkpoint if validation loss improved.
    if split == 'val' and (not best_val_loss or mean_loss < best_val_loss):
        logger.info('Saving checkpoint for new best loss')
        util.dist_save_checkpoint(model, optimizer, args.logdir, suffix='best')
        best_val_loss = mean_loss


def train(va_iter, optimizer, scheduler):
    global global_token_count, event_writer, train_loss, best_val_loss, \
        train_step, last_log_step, epoch
    # Turn on training mode which enables dropout.
    model.train()

    log_tb('sizes/batch_size', args.batch_size)
    log_tb('sizes/seq_size', args.tgt_len)

    tr_iter = corpus.get_dist_iterator(
        'train', global_rank, max_rank, args.batch_size, args.tgt_len,
        device=device, ext_len=args.ext_len)
    mems = tuple()
    log_start_time = time.time()
    for batch, (data, target, seq_len) in enumerate(tr_iter):
        assert seq_len == data.shape[0]
        for i in range(1, data.shape[0]):
            assert torch.all(torch.eq(data[i], target[i - 1]))
            break

        batch_total = torch.tensor(data.shape[1]).to(device)
        batch_total = batch_total.to(device)  # needed for NCCL sync
        if args.local:
            batch_total = batch_total.sum()
        else:
            batch_total = util.dist_sum_tensor(batch_total)  # global batch size
            batch_total = util.toscalar(batch_total)

        total_tokens = batch_total * seq_len
        should_log = train_step < args.verbose_log_steps or train_step % args.log_interval == 0

        global_token_count += total_tokens
        model.zero_grad()
        ret = model(data, target, *mems)
        loss, mems = ret[0], ret[1:]
        loss = loss.float().mean().type_as(loss)
        with timeit('backwards', noop=not should_log):
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
        train_loss += loss.float().item()

        if args.fp16:
            optimizer.clip_master_grads(args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        train_step += 1

        # step-wise learning rate annealing
        if args.fp16 and optimizer.overflow:
            logger.info("skipped iteration")
        else:
            if args.scheduler in ['cosine', 'constant', 'dev_perf']:
                # linear warmup stage
                if global_token_count < args.warmup_tokens:
                    curr_lr = args.lr * global_token_count / args.warmup_tokens
                    optimizer.param_groups[0]['lr'] = curr_lr
                elif args.scheduler == 'cosine':
                    # Divide by 1e6 for numerical stability.
                    scheduler.step(global_token_count // 1e6)
            else:
                scheduler.step(global_token_count)


        if should_log:
            elapsed_time = time.time() - log_start_time
            elapsed_steps = train_step - last_log_step

            # compute average loss over last logging interval
            cur_loss = train_loss / elapsed_steps
            log_str = f'| epoch {epoch:3d} step {train_step:>8d} | {batch:>6d} batches | lr {optimizer.param_groups[0]["lr"]:.3g} ' \
                      f'| ms/batch {elapsed_time * 1000 / elapsed_steps:5.2f} | loss {cur_loss:5.2f}'
            if args.dataset in ['enwik8', 'text8']:
                log_str += f' | bpc {cur_loss / math.log(2):9.5f}'
            else:
                log_str += f' | ppl {math.exp(cur_loss):9.3f}'
            logger.info(log_str)
            log_tb('learning/epoch', epoch)
            log_tb('_loss', cur_loss)  # the most important thing
            log_tb('learning/loss', cur_loss)
            log_tb('learning/ppl', math.exp(cur_loss))

            # currently step timings are not synchronized in multi-machine
            # case (see #4). Can add torch.distributed.barrier() to get
            # more accurate timings, but this may add slowness.
            log_tb('times/step', 1000 * elapsed_time / elapsed_steps)
            current_lr = optimizer.param_groups[0]['lr']

            log_tb('learning/lr', current_lr)

            # 32 is the "canonical" batch size
            linear_scaling_factor = batch_total / 32
            log_tb('learning/base_lr', current_lr / linear_scaling_factor)
            if args.optim == 'lamb':
                log_lamb_rs(optimizer, event_writer, global_token_count)

            time_per_batch = elapsed_time / elapsed_steps
            time_per_sample = time_per_batch / args.batch_size
            time_per_token = time_per_sample / args.tgt_len

            log_tb('times/batches_per_sec', 1 / time_per_batch)
            log_tb('times/samples_per_sec', 1 / time_per_sample)
            log_tb('times/tokens_per_sec', 1 / time_per_token)

            if str(device) == 'cuda':
                log_tb("memory/allocated_gb", torch.cuda.memory_allocated() / 1e9)
                log_tb("memory/max_allocated_gb", torch.cuda.max_memory_allocated() / 1e9)
                log_tb("memory/cached_gb", torch.cuda.memory_cached() / 1e9)
                log_tb("memory/max_cached_gb", torch.cuda.max_memory_cached() / 1e9)

            train_loss = 0
            log_start_time = time.time()
            last_log_step = train_step

        if train_step % args.eval_interval == 0:
            evaluate_and_log(optimizer, va_iter, 'val', train_step)

        if global_token_count >= args.max_tokens:
            if args.eta_min == 0:
                raise StopIteration
            logger.info('End of schedule, staying at current LR')
            args.scheduler = 'constant'

    if args.checkpoint_each_epoch:
        logger.info(f'Saving checkpoint for epoch {epoch}')
        util.dist_save_checkpoint(model, optimizer, args.logdir, suffix=f'{epoch}')


def main():
    global global_token_count, event_writer, train_step, train_loss, last_log_step, \
        best_val_loss, epoch, model

    if args.local_rank > 0:
        pass  # skip shutdown when rank is explicitly set + not zero rank
    else:
        os.system('shutdown -c')

    if not args.local:
        logger.info(
            f'Distributed initializing process group with {args.dist_backend}, {args.dist_url}, {util.get_world_size()}')
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=util.get_world_size())
        assert (util.get_world_size() == dist.get_world_size())
        logger.info(f"Distributed: success ({args.local_rank}/{dist.get_world_size()})")

    model = MemTransformerLM(ntokens, args.n_layer, args.n_head, args.d_model,
                             args.d_head, args.d_inner, args.dropout, args.dropatt,
                             tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
                             tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
                             ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
                             same_length=args.same_length, attn_type=args.attn_type,
                             clamp_len=args.clamp_len, sample_softmax=args.sample_softmax)

    # log model info
    n_all_param = sum([p.nelement() for p in model.parameters()])
    log_tb('sizes/params', n_all_param)
    n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])
    log_tb('sizes/non_emb_params', n_nonemb_param)
    logger.info('params %s non_emb_params %s', n_all_param, n_nonemb_param)

    # optimizer
    if args.optim.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
    elif args.optim.lower() == 'lamb':
        optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        assert args.optim.lower() == 'adam'
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # scheduler
    if args.scheduler == 'cosine':
        # Divide by 1e6 for numerical stability.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_tokens // 1e6, eta_min=args.eta_min)
    elif args.scheduler == 'finder':
        scheduler = LRFinder(optimizer, args.max_tokens, init_value=args.lr / 1e3)
    elif args.scheduler == 'constant':
        pass

    model.apply(weights_init)
    model.word_emb.apply(weights_init)  # ensure embedding init is not overridden by out_layer in case of weight sharing

    if args.checkpoint:
        if global_rank == 0:
            util.restore_from_checkpoint(model=model, checkpoint_fn=args.checkpoint)

    model = model.to(device)
    if args.fp16:
        model = FP16_Module(model)
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={'init_scale': 2 ** 16},
                                   verbose=False)

    if args.local:
        model = nn.DataParallel(model, dim=1)
    else:
        # Uncomment find_unused_parameters and upgrade to torch 1.1 for adaptive embedding.
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank) #, find_unused_parameters=True)

    if global_rank == 0:
        event_writer = SummaryWriter(args.logdir)

    event_writer.add_text('args', str(args))

    # test checkpoint writing
    if args.checkpoint_each_epoch:
        logger.info(f'Saving checkpoint for epoch {epoch}')
        util.dist_save_checkpoint(model, optimizer, args.logdir, suffix=f'{0}')

    # Loop over epochs.
    train_step = 0
    train_loss = 0
    last_log_step = 0
    best_val_loss = None
    va_iter, te_iter = [
         corpus.get_dist_iterator(
            split, global_rank, max_rank, args.batch_size * 2, args.tgt_len,
            device=device, ext_len=args.ext_len)
        for split in ('valid', 'test')
    ]

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in itertools.count(start=1):
            train(va_iter, optimizer, scheduler)
    except KeyboardInterrupt:
        logger.info('-' * 100)
        logger.info('Exiting from training early')
    except StopIteration:
        pass

    # Eval one more time.
    evaluate_and_log(optimizer, va_iter, 'val', train_step=-1)

    # Load the best saved model.
    logger.info("Loading best checkpoint")
    model_file = os.path.join(args.logdir, 'model-best.pt')
    if os.path.exists(model_file):
        with open(model_file, 'rb') as model_f:
            with timeit('load'):
                if args.local:
                    model = torch.load(model_f)
                else:
                    model = torch.load(model_f, map_location=lambda storage, loc: storage.cuda(args.local_rank))
                    model = DistributedDataParallel(
                        model,
                        device_ids=[args.local_rank],
                        output_device=args.local_rank)
    else:
        logger.warn('no model file, using current model for loss')

    # Run on test data.
    evaluate_and_log(optimizer, te_iter, 'test', -1)


if __name__ == '__main__':
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            main()
        if not args.skip_auto_shutdown and args.local_rank == 0:
            os.system(f'sudo shutdown -h -P +{args.auto_shutdown_success_delay_mins}')
    except Exception as e:
        import traceback

        traceback.print_exc(file=sys.stdout)
        # Logger automatically picks up exc info from context.
        logger.exception('Failed')
        # in case of exception, wait 2 hours before shutting down
        if not args.skip_auto_shutdown:
            os.system(f'sudo shutdown -h -P +{args.auto_shutdown_failure_delay_mins}')
