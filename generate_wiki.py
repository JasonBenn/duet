"""Generate samples from a model.

Note: only works for BPE-based models.
Based on https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from pytorch_pretrained_bert import GPT2Tokenizer


def init():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dir = os.environ.get('PYTORCH_PRETRAINED_BERT_CACHE', 'cache/')
    # Load the best saved model.
    with open(os.path.join(cache_dir, 'model-best.pt'), 'rb') as f:
        sys.path.insert(0,f'{os.getcwd()}/transformer_xl/')
        sys.path.insert(0,f'{os.getcwd()}/transformer_xl/utils')
        model = torch.load(f, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    
    if not torch.cuda.is_available():
        model = model.float()

    model = model.to(device)
    model.eval()
    return model

def main(model, tokenizer, context, length=20, batch_size=1, top_p=0, temperature=.7, top_k=40):
    max_context = 384
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NL = tokenizer.encode('\n')

    ## Init
    data = torch.tensor(NL*4 + tokenizer.encode(context.strip())).to(device)
    initial_length = len(data)
    # Turn into a batch.
    data.unsqueeze_(1)
    data = data.repeat_interleave(batch_size, dim=1)

    if not hasattr(model, 'init_mems'):
        model = model.module
    mems = model.init_mems()

    for i in range(length):
        ## Grab a sample from the last frame, append to result list, append to `data`
        pred_hid, mems = predict(model, data[-max_context:] if i == 0 else data[-1:], mems)
        softmax = hidden_to_softmax(model, pred_hid[-1], top_k=top_k, temperature=temperature, top_p=top_p)

        new_sample = torch.multinomial(softmax, num_samples=1).unsqueeze(-1).squeeze(2)
        data = torch.cat((data, new_sample.t()), dim=0)

    for i in range(data.size(1)):
        # Chop off the context
        return tokenizer.decode(data[initial_length:, i].tolist())

def predict(model, data, mems):
    tgt_len = data.size(0)
    with torch.no_grad():
        hidden, new_mems = model._forward(data, mems=mems)
    pred_hid = hidden[-tgt_len:]
    return pred_hid, new_mems

def hidden_to_softmax(model, hidden, temperature=1, top_k=0, top_p=0):
    """Turn a hidden projection into log softmax.

    Adapted from utils/proj_adaptive_softmax.py
    """
    # pas stands for ProjectedAdaptiveSoftmax
    pas = model.crit
    logits = pas._compute_logit(hidden, pas.out_layers[0].weight,
                                pas.out_layers[0].bias, pas.out_projs[0])
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    logits /= temperature
    softmax = F.softmax(logits, dim=-1)
    return softmax


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs >= top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(logits, dtype=torch.uint8).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove )
        logits[indices_to_remove] = filter_value
    return logits

if __name__ == '__main__':
    print('loading')
    enc = GPT2Tokenizer.from_pretrained('gpt2')
    model = init()
    print('generating')
    print(main(model, enc, 'Ben Mann is known for'))
