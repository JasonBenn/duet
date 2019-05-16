import contextlib
import math
import os
from collections import Counter, OrderedDict

import portalocker
import torch


class Vocab:
    def __init__(self, special=[], min_freq=0, max_size=None, lower_case=True,
                 delimiter=None, vocab_file=None):
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.max_size = max_size
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.vocab_file = vocab_file

    def tokenize(self, line, add_eos=False, add_double_eos=False):
        line = line.strip()
        # convert to lower case
        if self.lower_case:
            line = line.lower()

        # empty delimiter '' will evaluate False
        if self.delimiter == '':
            symbols = line
        else:
            symbols = line.split(self.delimiter)

        if add_double_eos: # lm1b
            return ['<S>'] + symbols + ['<S>']
        elif add_eos:
            return symbols + ['<eos>']
        else:
            return symbols

    def count_file(self, path, verbose=False, add_eos=False):
        """Update self.counter with tokenized symbol counts."""
        if verbose: 
            print(f'counting file {path} ...')
        assert os.path.exists(path), f"{path} doesn't exist"

        sents = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenize(line, add_eos=add_eos)
                self.counter.update(symbols)
                sents.append(symbols)

        return sents

    def count_sents(self, sents, verbose=False):
        """
            sents : a list of sentences, each a list of tokenized symbols
        """
        if verbose: print('counting {} sents ...'.format(len(sents)))
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    line {}'.format(idx))
            self.counter.update(symbols)

    def _build_from_file(self, vocab_file):
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                symb = line.strip().split()[0]
                self.add_symbol(symb)
        self.unk_idx = self.sym2idx['<UNK>']

    def build_vocab(self):
        if self.vocab_file:
            print('building vocab from {}'.format(self.vocab_file))
            self._build_from_file(self.vocab_file)
            print('final vocab size {}'.format(len(self)))
        else:
            print('building vocab with min_freq={}, max_size={}'.format(
                self.min_freq, self.max_size))
            self.idx2sym = []
            self.sym2idx = OrderedDict()

            for sym in self.special:
                self.add_special(sym)

            for sym, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_freq: break
                self.add_symbol(sym)

            print('final vocab size {} from {} unique tokens'.format(
                len(self), len(self.counter)))

    def encode_file(self, path: str, ordered=False, verbose=False, add_eos=True,
            add_double_eos=False) -> torch.LongTensor:
        if verbose: 
            print(f'encoding file {path} ...')
        assert os.path.exists(path), f"{path} doesn't exist"
        encoded = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenize(line, add_eos=add_eos,
                    add_double_eos=add_double_eos)
                encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    def encode_sents(self, sents, ordered=False, verbose=False):
        if verbose: print('encoding {} sents ...'.format(len(sents)))
        encoded = []
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    line {}'.format(idx))
            encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def get_sym(self, idx):
        assert 0 <= idx < len(self), 'Index {} out of range'.format(idx)
        return self.idx2sym[idx]

    def get_idx(self, sym):
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            # print('encounter unk {}'.format(sym))
            assert '<eos>' not in sym
            assert hasattr(self, 'unk_idx')
            return self.sym2idx.get(sym, self.unk_idx)

    def get_symbols(self, indices):
        return [self.get_sym(idx) for idx in indices]

    def get_indices(self, symbols):
        return [self.get_idx(sym) for sym in symbols]

    def convert_to_tensor(self, symbols):
        return torch.LongTensor(self.get_indices(symbols))

    def convert_to_sent(self, indices, exclude=None):
        if exclude is None:
            return ' '.join([self.get_sym(idx) for idx in indices])
        else:
            return ' '.join([self.get_sym(idx) for idx in indices if idx not in exclude])

    def __len__(self):
        # Force a multiple of 8 for efficient CUDA.
        return math.ceil(len(self.idx2sym) / 8) * 8

class OpenAIVocab(Vocab):
    def __init__(self, max_size, vocab_file=None):
        from pytorch_pretrained_bert import GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.EOT = self.tokenizer.encoder['<|endoftext|>']
        self.max_size = max_size
        self.vocab_file = vocab_file

    def __len__(self):
        return len(self.tokenizer)

    def count_file(self, path, verbose=False, add_eos=False):
        # TODO: train from scratch, respect self.max_size
        pass

    def build_vocab(self):
        pass

    def encode_file(self, path, ordered=False, verbose=False, add_eos=True, add_double_eos=False) -> torch.LongTensor:
        cached = path + '.tokenized'
        if os.path.exists(cached):
            return torch.load(cached)
        print(f'encoding file {path} ...')
        assert os.path.exists(path), f"{path} doesn't exist"

        with open(path, encoding='utf-8') as f:
            # Suppress warnings about length.
            with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
                out = torch.LongTensor(self.tokenizer.encode(f.read()) + [self.EOT])
                with portalocker.Lock(cached, timeout=60) as _:
                    torch.save(out, cached)
                return out


class GoogleBPEVocab(Vocab):
    """Don't use this until this issue is fixed.

    https://github.com/google/sentencepiece/issues/318
    """
    def __init__(self, max_size, vocab_file=None):
        import sentencepiece as spm
        self.spm = spm
        self.max_size = max_size
        self.vocab_file = vocab_file
        self.sp = spm.SentencePieceProcessor()
    def count_file(self, path, verbose=False, add_eos=False):
        self.spm.SentencePieceTrainer.Train(
            f'--input={self.vocab_file} --model_prefix=m --vocab_size={self.max_size} --model_type=bpe')

    def build_vocab(self):
        if self.vocab_file:
            self.sp.Load(self.vocab_file)
        else:
            pass

    def encode_file(self, path, ordered=False, verbose=False, add_eos=True, add_double_eos=False) -> torch.LongTensor:
        with open(path, encoding='utf-8') as f:
            return torch.LongTensor(self.sp.EncodeAsIds(f.read()))
