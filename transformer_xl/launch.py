#!/usr/bin/env python
"""Launch training on AWS with 8 GPUs."""

import argparse
from attrdict import AttrDict, AttrDefault
import re
import util

import ncluster

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='txl',
                    help="name of the current run, used for machine naming and rundir name")
parser.add_argument('--config', type=str, default='',
                    help='which training config to use')
parser.add_argument('--nospot', action='store_true',
                    help='Use more expensive on-demand instance')

parser.add_argument('--skip_setup', action='store_true',
                    help='Make startup slightly faster by skiping various initialization tasks, like '
                         'tmux/efs setup. Only use on reruns.')

parser.add_argument('--wiki', action='store_true',
                    help='Train on all of wikipedia.')
parser.add_argument('--bpe', action='store_true',
                    help='Use BPE to reduce vocab instead of adaptive softmax div')

# network settings
parser.add_argument('--num_rings', type=int, default=16)

# config flags for trying configurations outside of existing configs
parser.add_argument('--machines', type=int, default=0,
                    help="how many machines to use")
parser.add_argument('--instance_type', type=str, default='',
                    help="how many machines to use")
parser.add_argument('--checkpoint_each_epoch', type=int, default=0,
                    help='whether to save checkpoint at each epoch')
parser.add_argument('--image_name', type=str, default='',
                    help="use custom AMI ")
parser.add_argument('--conda_env', type=str, default='',
                    help='use custom conda env')
parser.add_argument('--checkpoint', type=str, default='',
                    help='restore from this checkpoint')

args = parser.parse_args()

# default environment settings, should change rarely since they affect
# all configs
IMAGE_NAME = 'cybertronai00'
CONDA_ENV = 'pytorch_p36'

# 'base_lr': learning rate for BASE_LR_BATCHSIZE, linear lr scaling will grow this rate proportionally to final global
# batch size
# local_batch_size: per-GPU batch size
BASE_LR_BATCHSIZE = 32

# logs: yaro-1gpu
one_gpu = {
    # 24x smaller batch than ben-big-lr.09, use 5x more agressive learning rate
    'base_lr': 0.000125 * 5 / 3 * 5,
    'local_batch_size': 32,
    'instance_type': 'p3.2xlarge',
    'machines': 1,
}

# logs: yaro-one.08
one_machine = {
    'base_lr': 0.000125 * 5 / 3,  # from ben-big-lr.09
    'instance_type': 'p3dn.24xlarge',
    'local_batch_size': 96,
    'machines': 1,
}

# logs: yaro-fp16
one_machine_fp16 = {
    'base_lr': 0.000125 * 5 / 3,  # from ben-big-lr.09
    'instance_type': 'p3dn.24xlarge',
    'local_batch_size': 96,
    'machines': 1,
    'extra_worker_params': {
        'fp16': True,
        'dynamic_loss_scale': True,
    }
}

one_small_machine = {
    'base_lr': 0.001 / 4, # Divide by 4 to counteract batch adjustment
    'instance_type': 'p3.16xlarge',
    'local_batch_size': 6,
    'machines': 1,
    'large': True,
    'extra_worker_params': {
        'fp16': True,
        'dynamic_loss_scale': True,
    }
}

# Match https://github.com/kimiyoung/transformer-xl/blob/master/tf/scripts/wt103_large_tpu.sh
# Differences: fp16, lamb, 0 warmup, untie_r (doesn't exist in pytorch)
# logs: ben-large-lamb-slow
one_machine_fp16_large = {
    'base_lr': 0.001 / 4, # Divide by 4 to counteract batch adjustment
    'instance_type': 'p3dn.24xlarge',
    'local_batch_size': 16,
    'machines': 1,
    'large': True,
}

# fork of one_machine_fp16_large
four_machine_fp16_large = {
    'base_lr': 0.001 / 4, # Divide by 4 to counteract batch adjustment
    'instance_type': 'p3dn.24xlarge',
    'local_batch_size': 16,
    'machines': 4,
    'large': True,
}

# logs: ben-eight.01
eight_machine_fp16_large = {
    'base_lr': 0.001 / 4, # Divide by 4 to counteract batch adjustment
    'instance_type': 'p3dn.24xlarge',
    'local_batch_size': 16,
    'machines': 8,
    'large': True,
}

# /ncluster/runs.new/yaro-fp16.09
one_machine_fp16_2xlr = {
    'base_lr': 0.000125 * 5 / 3 * 2,  # from ben-big-lr.09
    'instance_type': 'p3dn.24xlarge',
    'local_batch_size': 96,
    'machines': 1,
    'extra_worker_params': {
        'fp16': True,
        'dynamic_loss_scale': True,
    }
}

# /ncluster/runs.new/yaro-fp16.10
one_machine_fp16_4xlr = {
    'base_lr': 0.000125 * 5 / 3 * 4,  # from ben-big-lr.09
    'instance_type': 'p3dn.24xlarge',
    'local_batch_size': 96,
    'machines': 1,
    'extra_worker_params': {
        'fp16': True,
        'dynamic_loss_scale': True,
    }
}

# /ncluster/runs.new/yaro-fp16.11
# nans after 30 seconds
one_machine_fp16_8xlr = {
    'base_lr': 0.000125 * 5 / 3 * 8,  # from ben-big-lr.09
    'instance_type': 'p3dn.24xlarge',
    'local_batch_size': 96,
    'machines': 1,
    'extra_worker_params': {
        'fp16': True,
        'dynamic_loss_scale': True,
    }
}

# smaller p3.16 machine, logs: ben-bpe
one_machine_fp16_small = {
    'base_lr': 0.000125 * 5 / 3 / 2,  # from ben-big-lr.09
    'instance_type': 'p3.16xlarge',
    'local_batch_size': 96 // 2,
    'machines': 1,
    'extra_worker_params': {
        'fp16': True,
        'dynamic_loss_scale': True,
    }
}

one_machine_fp16_checkpoint = {
    'base_lr': 0.000125 * 5 / 3,  # from ben-big-lr.09
    'instance_type': 'p3dn.24xlarge',
    'local_batch_size': 96,
    'machines': 1,
    'checkpoint': '/ncluster/runs.new/yaro-one.08/model-1.pt',
    'extra_worker_params': {
        'fp16': True,
        'dynamic_loss_scale': True,
    }
}


# /ncluster/runs.new/yaro-two-fp16.04 (with checkpoints)
two_machines_fp16 = {
    'base_lr': 0.000125 * 5 / 3,  # from ben-big-lr.09
    'instance_type': 'p3dn.24xlarge',
    'local_batch_size': 96,
    'machines': 2,
    'extra_worker_params': {
        'fp16': True,
        'dynamic_loss_scale': True,
    }
}

two_machines = {
    'base_lr': 0.000125 * 5 / 3,  # yaro-two.07
    'instance_type': 'p3dn.24xlarge',
    'local_batch_size': 96,
    'machines': 2,
}


# yaro-four
four_machines = {
    'base_lr': 0.000125,  # remove ben's 5/3 tweak, and additional penalty of 2x
    'instance_type': 'p3dn.24xlarge',
    'local_batch_size': 96,
    'machines': 4,
    'extra_worker_params': {
        'fp16': True,
        'dynamic_loss_scale': True,
    }
}

eight_machines = {
    'base_lr': 0.000125/2,  # remove ben's 5/3 tweak, and additional penalty of 2x
    'instance_type': 'p3dn.24xlarge',
    'local_batch_size': 96,
    'machines': 8,
    'checkpoint': '/ncluster/runs.new/yaro-one.08/model-1.pt',
    'extra_worker_params': {
        'fp16': True,
        'dynamic_loss_scale': True,
        'warmup_tokens': 50e6,
    }
}

def dict_to_args(dict_: dict):
    def item_to_arg(item: tuple):
        k, v = item
        if v is False or v is None:
            return ''
        if v is True:
            return f'--{k}'
        return f'--{k} {v}'
        
    return ' '.join([item_to_arg(item) for item in dict_.items()])

# Match https://github.com/kimiyoung/transformer-xl/blob/master/tf/scripts/wt103_large_tpu.sh
LARGE_ARGS = {
    'n_layer': 18,
    'd_model': 1024,
    'n_head': 16,
    'd_head': 64,
    'd_inner': 4096,
    'dropout': 0.2,
    'dropatt': 0.2,
    'optim': 'lamb',
    'warmup_tokens': 0,
    'tgt_len': 384,
    'mem_len': 384,
    'eval_tgt_len': 128,
    'fp16': True,
    'dynamic_loss_scale': True,
    'init_std': 0.005,
    'div_val': 4,
}

# Roughly match https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/run_wt103_base.sh
# Divisible by 8 for fp16 compatibility.
SMALL_ARGS = {
    'n_layer': 16,
    'd_model': 512,
    'n_head': 8,
    'd_head': 48,
    'd_inner': 2048,
    'dropout': 0.1,
    'dropatt': 0.0,
    'optim': 'lamb',
    'tgt_len': 128,
    'mem_len': 128,
    'eval_tgt_len': 128,
}

def _get_nccl_params():
    params = f'NCCL_DEBUG=VERSION '

    params += f'NCCL_MIN_NRINGS={args.num_rings} ' \
        f'NCCL_MAX_NRINGS={args.num_rings} '
    return params


def main():
    ncluster.set_backend('aws')

    if args.config:
        assert not args.instance_type, "specify instance_type as part of config"
        assert not args.machines, "specify number of machines as part of config"
        assert re.match('\\w+', args.config)
        assert args.config in globals(), f'no config called {args.config}'
        config = eval(args.config)

    else:  # setting config vars through command-line flags
        assert args.instance_type
        assert args.machines
        config = {'base_lr': 0.000125 * 5 / 3,
                  'local_batch_size': 96,
                  'instance_type': args.instance_type,
                  'machines': args.machines}

    config = AttrDefault(str, config)  # easier access to dictionary entries
    config.image_name = IMAGE_NAME
    config.conda_env = CONDA_ENV

    if args.conda_env:
        config.conda_env = args.conda_env
        print("Using non-standard conda env ", config.conda_env)
    if args.image_name:
        config.image_name = args.image_name
        print("Using non-standard image ", config.image_name)

    instance_info = ncluster.aws_backend.INSTANCE_INFO[config.instance_type]
    num_gpus_per_machine = instance_info['gpus']

    job = ncluster.make_job(name=args.name,
                            run_name=f"{args.name}",
                            num_tasks=config.machines,
                            image_name=config.image_name,
                            instance_type=config.instance_type,
                            spot=not args.nospot,
                            skip_setup=args.skip_setup)

    job.rsync('.')
    job.run(f'killall python || echo failed && '  # kill previous run
            f'source activate {config.conda_env} && ' +
            f'pip install -r requirements.txt')

    local_batch_size = config.local_batch_size
    base_lr = config.base_lr

    num_workers = num_gpus_per_machine * config.machines
    global_batch_size = local_batch_size * num_workers
    print("using global batch ", global_batch_size)  # 512=8*32*2*1

    # linear LR scaling (https://arxiv.org/abs/1706.02677)
    lr = base_lr * (global_batch_size / BASE_LR_BATCHSIZE)

    # worker parameters with training setup
    worker_params = {
        'seed': 1111,
        'data': 'data/wikitext-103',
        'dataset': 'wt103',
        'adaptive': True,
        'log_interval': 100,
        'eval_interval': 500,
        'max_tokens': int(1.5e9),
        'logdir': job.logdir,
        'lr': lr,
        'batch_size': local_batch_size,
        'eta_min': lr / 10,
    }
    
    worker_params.update(LARGE_ARGS if config.large else SMALL_ARGS)

    user_params = {}
    # pass through some user-provided settings that were arguments to the launcher script
    if args.checkpoint_each_epoch:
        user_params['checkpoint_each_epoch'] = args.checkpoint_each_epoch
    if config.warmup_tokens:
        user_params['warmup_tokens'] = config.warmup_tokens

    if args.checkpoint or config.checkpoint:
        user_params['checkpoint'] = util.one_of([args.checkpoint, config.checkpoint])

    if args.wiki:
        worker_params.update({
            'data': 'data/wikiextracted',
            'dataset': 'wiki',
            'dropatt': 0.1,
            'dropout': 0.1,
        })

    if args.bpe:
        worker_params.update({
            'div_val': 1,
            'bpe': True,
            'adaptive': False,
        })

    worker_params.update(user_params)

    if config.extra_worker_params:
        worker_params.update(config.extra_worker_params)

    nccl_params = _get_nccl_params()

    for i, task in enumerate(job.tasks):
        dist_params = \
            f'--nproc_per_node={num_gpus_per_machine} ' \
            f'--nnodes={config.machines} --node_rank={i} ' \
            f'--master_addr={job.tasks[0].ip} --master_port={6016}'
        cmd = f'{nccl_params} python -m torch.distributed.launch {dist_params} train.py {dict_to_args(worker_params)}'
        task.run(f'echo {cmd} > {job.logdir}/task-{i}.cmd')  # save command-line
        task.run(cmd, non_blocking=True)

    print(f"Logging to {job.logdir}")

if __name__ == '__main__':
    main()
