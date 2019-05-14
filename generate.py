import numpy as np
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import trange


def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0,
                    device='cuda', sample=True):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None

    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output


def init():
    seed = 42
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(device)
    model.eval()
    return enc, model

def main(model: GPT2LMHeadModel, enc: GPT2Tokenizer, phrase: str = ''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nsamples = 1
    length = 20
    temperature = .8
    top_k = 0
    batch_size = 1
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
            temperature=temperature, top_k=top_k, device=device
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
