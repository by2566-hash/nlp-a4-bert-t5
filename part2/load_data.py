"""AI-assisted self-study reference: aligned, lossless text-to-SQL input pipeline."""
from functools import partial
from pathlib import Path
import re

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AddedToken, T5TokenizerFast

MODEL_NAME = 'google-t5/t5-small'
PAD_IDX = 0
DEFAULT_PREFIX = 'translate English to SQL: '


def build_tokenizer(model_name=MODEL_NAME, local_files_only=False):
    tokenizer = T5TokenizerFast.from_pretrained(model_name, local_files_only=local_files_only)
    # SentencePiece's original T5 vocabulary maps '<' to UNK. A normal added token
    # survives skip_special_tokens=True and is saved together with the checkpoint.
    tokenizer.add_tokens([AddedToken('<', normalized=False, special=False)])
    return tokenizer


def decode_sql(tokenizer, token_ids):
    """Restore '<' token boundary spacing outside SQL quoted strings/identifiers.

    T5's decoder inserts whitespace between an added '<' and the original '=' or
    '>' token. Preserve quoted content exactly; only repair SQL operators.
    """
    text = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    pieces = re.split(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")", text)
    for index in range(0, len(pieces), 2):
        pieces[index] = re.sub(r'\s*<\s*([=>]?)\s*',
                               lambda match: ' <' + match.group(1) + ' ', pieces[index])
    return ''.join(pieces).strip()


def load_lines(path):
    with open(path, encoding='utf-8') as handle:
        return [line.strip() for line in handle]


class T5Dataset(Dataset):
    def __init__(self, data_folder, split, tokenizer=None, prefix=DEFAULT_PREFIX,
                 max_source_length=512, max_target_length=512):
        if split not in {'train', 'dev', 'test'}:
            raise ValueError(f'Unknown split: {split}')
        self.tokenizer = tokenizer if tokenizer is not None else build_tokenizer()
        self.prefix = prefix
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.split = split
        self.examples = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        data_folder = Path(data_folder)
        inputs = load_lines(data_folder / f'{split}.nl')
        targets = None if split == 'test' else load_lines(data_folder / f'{split}.sql')
        if not inputs or (targets is not None and len(inputs) != len(targets)):
            raise ValueError(f'{split}: empty or misaligned input/target files')
        examples = []
        for index, text in enumerate(inputs):
            if not text or (targets is not None and not targets[index]):
                raise ValueError(f'{split} line {index + 1}: blank input/target')
            source = tokenizer(self.prefix + text, truncation=False)['input_ids']
            target = None if targets is None else tokenizer(targets[index], truncation=False)['input_ids']
            for name, ids, limit in [('input', source, self.max_source_length),
                                     ('SQL', target, self.max_target_length)]:
                if ids is not None and len(ids) > limit:
                    raise ValueError(f'{split} line {index + 1}: {name} has {len(ids)} tokens, '
                                     f'exceeding {limit}; increase limit, never silently truncate SQL')
            if target is not None and tokenizer.unk_token_id in target:
                raise ValueError(f'{split} line {index + 1}: SQL contains an unrepresentable token')
            examples.append({'input_ids': source, 'labels': target})
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def normal_collate_fn(batch, pad_token_id=PAD_IDX, decoder_start_token_id=PAD_IDX):
    """T5 starts with PAD; shifted labels provide teacher-forced decoder inputs."""
    encoder_ids, encoder_mask, initial = test_collate_fn(batch, pad_token_id, decoder_start_token_id)
    targets = pad_sequence([torch.tensor(item['labels'], dtype=torch.long) for item in batch],
                           batch_first=True, padding_value=-100)
    decoder_inputs = torch.full_like(targets, pad_token_id)
    decoder_inputs[:, 0] = decoder_start_token_id
    decoder_inputs[:, 1:] = targets[:, :-1].masked_fill(targets[:, :-1] == -100, pad_token_id)
    return encoder_ids, encoder_mask, decoder_inputs, targets, initial


def test_collate_fn(batch, pad_token_id=PAD_IDX, decoder_start_token_id=PAD_IDX):
    encoder_ids = pad_sequence([torch.tensor(item['input_ids'], dtype=torch.long) for item in batch],
                               batch_first=True, padding_value=pad_token_id)
    encoder_mask = encoder_ids.ne(pad_token_id).long()
    initial = torch.full((len(batch), 1), decoder_start_token_id, dtype=torch.long)
    return encoder_ids, encoder_mask, initial


def get_dataloader(batch_size, split, tokenizer=None, data_folder=None, seed=42,
                   prefix=DEFAULT_PREFIX, max_source_length=512, max_target_length=512):
    data_folder = data_folder or Path(__file__).resolve().parent / 'data'
    dset = T5Dataset(data_folder, split, tokenizer, prefix, max_source_length, max_target_length)
    collate = normal_collate_fn if split != 'test' else test_collate_fn
    return DataLoader(dset, batch_size=batch_size, shuffle=split == 'train',
                      collate_fn=partial(collate, pad_token_id=dset.tokenizer.pad_token_id,
                                         decoder_start_token_id=dset.tokenizer.pad_token_id),
                      generator=torch.Generator().manual_seed(seed))


def load_t5_data(batch_size, test_batch_size, **kwargs):
    tokenizer = kwargs.pop('tokenizer', None)
    if tokenizer is None:
        tokenizer = build_tokenizer()
    return tuple(get_dataloader(batch_size if split == 'train' else test_batch_size,
                                split, tokenizer=tokenizer, **kwargs)
                 for split in ('train', 'dev', 'test'))


def load_prompting_data(data_folder):
    folder = Path(data_folder)
    return tuple(load_lines(folder / name) for name in
                 ('train.nl', 'train.sql', 'dev.nl', 'dev.sql', 'test.nl'))
