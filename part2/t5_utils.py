import json
from pathlib import Path

import torch
import transformers
from transformers import T5Config, T5ForConditionalGeneration
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_wandb(args):
    import wandb
    wandb.init(project='dsga1011-a4-text-to-sql', name=args.experiment_name, config=vars(args))


def initialize_model(args, tokenizer=None):
    kwargs = {'local_files_only': args.local_files_only}
    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name, **kwargs)
    else:
        model = T5ForConditionalGeneration(T5Config.from_pretrained(args.model_name, **kwargs))
    if tokenizer is not None and len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    return model.to(DEVICE)


def mkdir(dirpath):
    Path(dirpath).mkdir(parents=True, exist_ok=True)


def checkpoint_directory(args):
    model_type = 'ft' if args.finetune else 'scr'
    return Path(args.output_dir) / 'checkpoints' / f'{model_type}_experiments' / args.experiment_name


def save_model(checkpoint_dir, model, best, tokenizer=None, args=None, metrics=None):
    path = Path(checkpoint_dir) / ('best' if best else 'last')
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    if args is not None:
        (path / 'run_config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    if metrics is not None:
        (path / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')


def load_model_from_checkpoint(args, best):
    path = Path(args.checkpoint_path) if args.checkpoint_path else checkpoint_directory(args) / ('best' if best else 'last')
    return T5ForConditionalGeneration.from_pretrained(path, local_files_only=True).to(DEVICE)


def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    return optimizer, initialize_scheduler(args, optimizer, epoch_length)


def initialize_optimizer(args, model):
    decay_parameters = {name for name in get_parameter_names(model, ALL_LAYERNORM_LAYERS) if 'bias' not in name}
    groups = [
        {'params': [p for n, p in model.named_parameters() if n in decay_parameters and p.requires_grad],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if n not in decay_parameters and p.requires_grad],
         'weight_decay': 0.0},
    ]
    if args.optimizer_type != 'AdamW':
        raise ValueError(f'Unsupported optimizer: {args.optimizer_type}')
    return torch.optim.AdamW(groups, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999))


def initialize_scheduler(args, optimizer, epoch_length):
    if args.scheduler_type == 'none':
        return None
    return transformers.get_scheduler(args.scheduler_type, optimizer=optimizer,
                                      num_warmup_steps=epoch_length * args.num_warmup_epochs,
                                      num_training_steps=epoch_length * args.max_n_epochs)


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        if not isinstance(child, tuple(forbidden_layer_types)):
            result += [f'{name}.{n}' for n in get_parameter_names(child, forbidden_layer_types)]
    result += list(model._parameters.keys())
    return result
