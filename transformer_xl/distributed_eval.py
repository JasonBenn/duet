# coding: utf-8
#
# To run locally:
#
# download/untar s3://yaroslavvb2/data/txl-wikitext-2.tar to /ncluster/data/wikitext-2, then
#
# python train.py --log-interval=1 --eval-interval=5 --max_tokens=500 --batch_size=1 --work_dir=/tmp/checkpoints --dataset=wt2 --data=../data/wikitext-2 --n_layer=1 --n_head=1 --d_head=1 --d_model=2 --d_inner=2  --dataset wt2 --max_eval_steps 1 --data=/ncluster/data/wikitext-2 --lr 0.025
#
# Tensorboard results go to /ncluster/runs
#
# To run remotely:
# cp -R /ncluster/data/transformer-xl-data ../data
# bash run_wt103_base.sh train --work_dir ~/workdir
import argparse
import datetime
import logging
import os
import sys
import time
import warnings
from collections import OrderedDict

import numpy as np
import pytz
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from pytorch_lamb import Lamb
from torch.nn.parallel import DistributedDataParallel

from data_utils import get_lm_corpus
from lr_finder import LRFinder
from mem_transformer import MemTransformerLM

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--logdir', type=str, default='/tmp/default', help="where logs and events go")
parser.add_argument('--run_name', type=str, default='txl', help="name of run")

parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8', 'wt2'],
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
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='split batch into chunks to save memory')
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
parser.add_argument('--varlen', action='store_true',
                    help='use variable length')
parser.add_argument('--log-interval', type=int, default=200,
                    help='logging interval in number of steps')
parser.add_argument('--retune-interval', type=int, default=5,
                    help='how often to retune parameters')
parser.add_argument('--verbose-log-steps', type=int, default=60,
                    help='do logging at every step for this many steps at the start of training')

parser.add_argument('--eval-interval', type=int, default=4000,
                    help='evaluation interval in number of steps')

parser.add_argument('--checkpoint-each-epoch', type=int, default=0,
                    help='whether to save checkpoint at each epoch')

parser.add_argument('--work_dir', default=None, type=str,
                    help='Experiment directory. Defaults to logdir')
parser.add_argument('--restart', action='store_true',
                    help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir', type=str, default='',
                    help='restart dir')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
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
parser.add_argument('--fp16', action='store_true',
                    help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can '
                    'improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument'
                    ' supersedes --static-loss-scale.')

# distributed training flags
parser.add_argument('--distributed', action='store_true', help='Run distributed training. Default True')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--local_rank', default=0, type=int,
                    help='Used for multi-process training. Can either be manually set ' +
                    'or automatically set by using \'python -m multiproc\'.')

# infra flags
parser.add_argument('--skip-auto-shutdown', action='store_true',
                    help='skip shutdown at the end of training or failure')
parser.add_argument('--auto-shutdown-success-delay-mins', default=10, type=int,
                    help='how long to wait until shutting down on success')
parser.add_argument('--auto-shutdown-failure-delay-mins', default=60, type=int,
                    help='how long to wait before shutting down on error')

parser.add_argument('--checkpoint', '/ncluster/runs.new/yaro-one.05/model-1.pt')


parser.add_argument('--role', type=str, default='worker',
                    help='internal flag, launcher or worker')

args = parser.parse_args()
args.tied = not args.not_tied


class DDP(DistributedDataParallel):

    def __init__(self, module, *args, **kwargs):
        super(DistributedDataParallel, self, *args, **kwargs).__init__()
        self.module = module

        for p in self.module.parameters():
            if torch.is_tensor(p):
                dist.broadcast(p, 0)

    def forward(self, *args, **kwargs):
        # DDP has a sync point on forward. No need to do this for eval. This allows us to have different batch sizes
        if self.training: return super().forward(*args, **kwargs)
        else:             return self.module(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.module.load_state_dict(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)
    

def env_world_size(): return int(os.environ.get('WORLD_SIZE', 1))


def env_rank(): return int(os.environ.get('RANK', 0))



def sum_tensor(tensor):
    if not args.distributed:
        return tensor
    rt = tensor.clone()
    # TODO(y): fix UserWarning: torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead
    #  warnings.warn("torch.distributed.reduce_op is deprecated, please use "
    # /home/ubuntu/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/torch/distributed/distributed_c10d.py:86: U

    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    return rt

# no_op method/object that accept every signature
class NoOp:
  def __getattr__(self, *args):
    def no_op(*args, **kwargs): pass
    return no_op


# todo(y): optimnizer/model are also global variables, fix
def save_checkpoint(model_1, optimizer_1, suffix=''):
    if not is_master:
        return
    with timeit('save'):
        with open(args.work_dir+f'/model-{suffix}.pt', 'wb') as f_1:
            torch.save(model_1, f_1)
        with open(args.work_dir+f'/optimizer-{suffix}.pt', 'wb') as f_1:
            torch.save(optimizer_1.state_dict(), f_1)


def toscalar(t):  # use on python scalars/pytorch scalars
    if isinstance(t, (float, int)): return t
    if hasattr(t, 'item'): return t.item()
    else: return t[0]


# install pdb handler on error
if global_rank == 0:
    def info(type, value, tb):
        if hasattr(sys, 'ps1') or not sys.stderr.isatty():
            # we are in interactive mode or we don't have a tty-like
            # device, so we call the default hook
            sys.__excepthook__(type, value, tb)
        else:
            import traceback, pdb
            # we are NOT in interactive mode, print the exception...
            traceback.print_exception(type, value, tb)
            print()
            # ...then start the debugger in post-mortem mode.
            # pdb.pm() # deprecated
            pdb.post_mortem(tb) # more "modern"

    sys.excepthook = info


class FileLogger:
  def __init__(self, output_dir, is_master=False, is_rank0=False):
    self.output_dir = output_dir
    if not os.path.exists(self.output_dir):
        if is_master:
            os.makedirs(self.output_dir)
    # only log on one process per node
    if is_rank0:
      self.logger = self.get_logger(output_dir, log_to_file=is_master)
    else:
      self.logger = NoOp()
    
  def exception(self, *args, **kwargs):
    return self.logger.exception(*args, **kwargs)

  def get_logger(self, output_dir, log_to_file=True):
    logger = logging.getLogger('txl training')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')

    if log_to_file:
      vlog = logging.FileHandler(output_dir+'/info.log')
      vlog.setLevel(logging.INFO)
      vlog.setFormatter(formatter)
      logger.addHandler(vlog)

      eventlog = logging.FileHandler(output_dir+'/warn.log')
      eventlog.setLevel(logging.WARN)
      eventlog.setFormatter(formatter)
      logger.addHandler(eventlog)

      time_formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(message)s')
      debuglog = logging.FileHandler(output_dir+'/debug.log')
      debuglog.setLevel(logging.DEBUG)
      debuglog.setFormatter(time_formatter)
      logger.addHandler(debuglog)
      
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)
    return logger

  def debug(self, *args):
    self.logger.debug(*args)

  def warn(self, *args):
    self.logger.warn(*args)

  def info(self, *args):
    self.logger.info(*args)


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

    def __exit__(self, *args):
        if self.noop:
            return
        self.end = time.perf_counter()
        interval_ms = 1000 * (self.end - self.start)
        global_timeit_dict.setdefault(self.tag, []).append(interval_ms)
        newtag = 'times/' + self.tag
        log_tb(newtag, interval_ms)


def log_tb(tag, val):
    """Log value to tensorboard (relies on global_example_count rather than step count to give comparable graphs across
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
assert args.batch_size % args.batch_chunk == 0

if not args.work_dir:
    args.work_dir = args.logdir
#args.work_dir = '{}-{}'.format(args.work_dir, args.dataset)
#args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
#logging = create_exp_dir(args.work_dir,
#    scripts_to_save=['train.py', 'mem_transformer.py'], debug=args.debug)

is_master = (not args.distributed) or (global_rank==0)
is_rank0 = args.local_rank == 0
logger = FileLogger(args.logdir, is_master=is_master, is_rank0=is_rank0)


# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Validate `--fp16` option
# if args.fp16:
#     if not args.cuda:
#         print('WARNING: --fp16 requires --cuda, ignoring --fp16 option')
#         args.fp16 = False
#     else:
#         try:
#             from apex.fp16_utils import FP16_Optimizer
#         except:
#             print('WARNING: apex not installed, ignoring --fp16 option')
#             args.fp16 = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###############################################################################
# Load data
###############################################################################
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)
args.n_token = ntokens

eval_batch_size = 10
tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
    device=device, ext_len=args.ext_len)
va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len,
    device=device, ext_len=args.ext_len)
te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len,
    device=device, ext_len=args.ext_len)

# adaptive softmax / embedding
cutoffs, tie_projs = [], [False]
if args.adaptive:
    assert args.dataset in ['wt103', 'lm1b']
    if args.dataset == 'wt103' or args.dataset == 'wt2':
        cutoffs = [20000, 40000, 200000]
        tie_projs += [True] * len(cutoffs)
    elif args.dataset == 'lm1b':
        cutoffs = [60000, 100000, 640000]
        tie_projs += [False] * len(cutoffs)


# todo(y): move into main()
logger.info("Torch version: {}".format(torch.__version__))
logger.info('=' * 100)
for k, v in args.__dict__.items():
    logger.info('    - {} : {}'.format(k, v))
logger.info('=' * 100)
logger.info('#params = {}'.format(args.n_all_param))
logger.info('#non emb params = {}'.format(args.n_nonemb_param))
#print('model')
#print(model)


def worker():
   print(f'Distributed initializing process group with {args.dist_backend}, {args.dist_url}, {env_world_size()}')

    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=env_world_size())
    assert(env_world_size() == dist.get_world_size())
    print("Distributed: success (%d/%d)"%(args.local_rank, dist.get_world_size()))

    if is_master:
      model = torch.load(f)
      model.eval()

    model = DDP(model, device_ids=[args.local_rank],
                output_device=args.local_rank)

    train_iter = tr_iter.get_dist_iter(global_rank, max_rank)

    total_loss = 0
    total_examples = 0
    for batch, (data, target, seq_len) in enumerate(train_iter):	
        # TODO(y): batch is dimension 1, why?
        assert seq_len == data.shape[0]

        batch_total = torch.tensor(data.shape[1]).to(device)
        batch_total = batch_total.to(device)       # needed for NCCL sync
        batch_total = sum_tensor(batch_total)      # global batch size
        total_tokens = batch_total.item()*seq_len

        ret = model(data, target, *mems)
        loss, mems = ret[0], ret[1:]
        global_loss = dist.all_reduce(loss, op=dist.reduce_op.SUM)

        total_loss += global_loss.item()
        total_examples += batch_total.item()
        if total_examples % 1000 == 0 and args.local_rank == 0:
            print(total_loss / total_examples)

    print('=' * 100)

    if args.local_rank == 0:
        print("Final loss", total_loss / total_examples)

def launcher():
  
  training_params = default_params + training_params

    # pass through command-line launcher arguments to the worker
    user_params = ['--checkpoint-each-epoch', args.checkpoint_each_epoch]

    training_params.extend(user_params)
    
    training_params = ' '.join(str(p) for p in training_params)
    nccl_params = get_nccl_params(args.machines, num_gpus)

    for i, task in enumerate(job.tasks):
        dist_params = \
            f'--nproc_per_node={num_gpus} ' \
            f'--nnodes={args.machines} --node_rank={i} ' \
            f'--master_addr={job.tasks[0].ip} --master_port={6016}'
        cmd = f'{nccl_params} python -m torch.distributed.launch {dist_params} train.py {training_params}'
        task.run(f'echo {cmd} > {job.logdir}/task-{i}.cmd')  # save command-line
        task.run(cmd, non_blocking=True)

    print(f"Logging to {job.logdir}")


def main():
    if args.role == "launcher":
        launcher()
    elif args.role == "worker":

        torch.cuda.set_device(args.local_rank)

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        log = util.FileLogger(args.logdir + f'/worker-{util.get_global_rank()}', mirror=(args.local_rank == 0))

        torch.cuda.set_device(args.local_rank)
        #      test_p2p()
        if args.method == 'optimize':
            test_optimize()
        elif args.method == 'allreduce':
            test_allreduce()
        else:
            assert False, 'unknown arg'

if __name__ == '__main__':
  main()

