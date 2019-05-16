import numpy as np
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
import sys

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

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0,
                    device='cuda', top_p=0, stop_token=[]):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None

    count = 0

    with torch.no_grad():
        while True:
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_top_p_filtering(logits, top_p=top_p, top_k=top_k)
            probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(probs, num_samples=1)
            
            output = torch.cat((output, prev), dim=1)
            count += 1
            if prev in stop_token or count > length:
                break
    return output


def init():
    #seed = 42
    #np.random.seed(seed)
    #torch.random.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(device)
    model.eval()
    return enc, model

def main(model: GPT2LMHeadModel, enc: GPT2Tokenizer, phrase: str = ''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nsamples = 1
    length = 40
    temperature = 1
    top_k = 0
    top_p = 0.9
    batch_size = 1
    stop_token = [enc.encoder[x] for x in ('<|endoftext|>', '.', '?', '!')]
    assert nsamples % batch_size == 0

    if length == -1:
        length = model.config.n_ctx // 2
    elif length > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

    context_tokens = enc.encode(phrase) if phrase else [enc.encoder['<|endoftext|>']]
    generated = 0
    results = []
    for _ in range(nsamples // batch_size):
        out = sample_sequence(
            model=model, length=length,
            context=context_tokens,
            start_token=None,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, device=device,
            top_p=top_p,
            stop_token=stop_token
        )
        out = out[:, len(context_tokens):].tolist()

        for i in range(batch_size):
            generated += 1
            text = enc.decode(out[i])
            results.append(text)
    return results

if __name__ == '__main__':
    enc_, model_ = init()
    print(main(model_, enc_, None))
