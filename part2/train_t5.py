import argparse
import hashlib
import json
import math
import platform
import transformers
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import T5TokenizerFast

from load_data import DEFAULT_PREFIX, MODEL_NAME, build_tokenizer, decode_sql, get_dataloader
from t5_utils import (DEVICE, checkpoint_directory, initialize_model,
                      initialize_optimizer_and_scheduler, load_model_from_checkpoint,
                      save_model, setup_wandb)
from utils import compute_metrics, save_queries_and_records, set_random_seeds

ROOT = Path(__file__).resolve().parent


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Fine-tune T5 on text-to-SQL; test generation is opt-in')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--finetune', dest='finetune', action='store_true')
    mode.add_argument('--from_scratch', dest='finetune', action='store_false')
    parser.set_defaults(finetune=True)
    parser.add_argument('--model_name', default=MODEL_NAME)
    parser.add_argument('--local_files_only', action='store_true')
    parser.add_argument('--optimizer_type', default='AdamW', choices=['AdamW'])
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--scheduler_type', default='linear', choices=['none', 'cosine', 'linear'])
    parser.add_argument('--num_warmup_epochs', type=int, default=1)
    parser.add_argument('--max_n_epochs', type=int, default=30)
    parser.add_argument('--patience_epochs', type=int, default=5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--experiment_name', default='experiment')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prefix', default=DEFAULT_PREFIX)
    parser.add_argument('--max_source_length', type=int, default=512)
    parser.add_argument('--max_target_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--data_dir', default=str(ROOT / 'data'))
    parser.add_argument('--output_dir', default=str(ROOT))
    parser.add_argument('--ground_truth_records', default=str(ROOT / 'records/ground_truth_dev.pkl'))
    parser.add_argument('--checkpoint_path', default=None, help='Saved best/last directory for --test_only')
    parser.add_argument('--predict_test', action='store_true', help='Generate test once after model selection')
    parser.add_argument('--test_only', action='store_true', help='Load selected checkpoint and generate test; no training')
    args = parser.parse_args(argv)
    for field in ('batch_size', 'test_batch_size', 'max_n_epochs', 'patience_epochs',
                  'max_source_length', 'max_target_length', 'max_new_tokens', 'num_beams'):
        if getattr(args, field) <= 0:
            parser.error(f'--{field} must be positive')
    if args.learning_rate <= 0 or args.max_grad_norm <= 0 or args.weight_decay < 0:
        parser.error('learning_rate/max_grad_norm must be positive and weight_decay nonnegative')
    if args.num_warmup_epochs < 0 or args.num_warmup_epochs > args.max_n_epochs:
        parser.error('warmup epochs must lie between 0 and max_n_epochs')
    if Path(args.experiment_name).name != args.experiment_name or args.experiment_name in {'.', '..'}:
        parser.error('experiment_name must be a single filename component')
    if args.checkpoint_path and not args.test_only:
        parser.error('--checkpoint_path is only used with --test_only')
    return args


def output_paths(args, split):
    model_type = 'ft' if args.finetune else 'scr'
    stem = f't5_{model_type}_{args.experiment_name}_{split}'
    if split == 'test':
        stem = 't5_ft_experiment_test' if args.finetune else 't5_ft_experiment_ec_test'
    root = Path(args.output_dir)
    sql_path, record_path = root / 'results' / f'{stem}.sql', root / 'records' / f'{stem}.pkl'
    sql_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    return str(sql_path), str(record_path)


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1, epochs_since_improvement = -math.inf, 0
    checkpoint_dir = checkpoint_directory(args)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_dir / 'history.jsonl'
    if history_path.exists() or (checkpoint_dir / 'best/config.json').exists():
        raise FileExistsError(f'Existing run at {checkpoint_dir}; use a new experiment_name')
    tokenizer = train_loader.dataset.tokenizer
    (checkpoint_dir / 'run_config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    runtime = {'python': platform.python_version(), 'torch': torch.__version__,
               'transformers': transformers.__version__, 'device': str(next(model.parameters()).device),
               'cuda_version': torch.version.cuda,
               'data_sha256': {name: file_sha256(Path(args.data_dir) / name)
                               for name in ('train.nl', 'train.sql', 'dev.nl', 'dev.sql')}}
    (checkpoint_dir / 'runtime.json').write_text(json.dumps(runtime, indent=2), encoding='utf-8')
    model_sql_path, model_record_path = output_paths(args, 'dev')
    for epoch in range(args.max_n_epochs):
        train_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        values = eval_epoch(args, model, dev_loader, str(Path(args.data_dir) / 'dev.sql'),
                            model_sql_path, args.ground_truth_records, model_record_path)
        loss, record_f1, record_em, sql_em, error_rate = values
        if not all(math.isfinite(float(value)) for value in (train_loss, *values)):
            raise RuntimeError('Nonfinite training/evaluation result; refusing checkpoint selection')
        metrics = {'epoch': epoch + 1, 'train_loss': train_loss, 'dev_loss': loss,
                   'record_f1': float(record_f1), 'record_em': float(record_em),
                   'sql_em': float(sql_em), 'sql_error_rate': error_rate}
        print(json.dumps(metrics), flush=True)
        with history_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(metrics) + '\n')
        if args.use_wandb:
            import wandb
            wandb.log(metrics, step=epoch + 1)
        improved = record_f1 > best_f1
        if improved:
            best_f1, epochs_since_improvement = record_f1, 0
        else:
            epochs_since_improvement += 1
        save_model(checkpoint_dir, model, best=False, tokenizer=tokenizer, args=args, metrics=metrics)
        if improved:
            save_model(checkpoint_dir, model, best=True, tokenizer=tokenizer, args=args, metrics=metrics)
        if epochs_since_improvement >= args.patience_epochs:
            break


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss, total_tokens = 0.0, 0
    device = next(model.parameters()).device
    for encoder_input, encoder_mask, decoder_input, targets, _ in tqdm(train_loader, desc='train'):
        optimizer.zero_grad(set_to_none=True)
        targets = targets.to(device)
        output = model(input_ids=encoder_input.to(device), attention_mask=encoder_mask.to(device),
                       decoder_input_ids=decoder_input.to(device), labels=targets)
        loss = output.loss
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        tokens = targets.ne(-100).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
    if not total_tokens:
        raise ValueError('Training loader has no supervised target tokens')
    return total_loss / total_tokens


def generate_queries(args, model, batch, tokenizer):
    device = next(model.parameters()).device
    generated = model.generate(input_ids=batch[0].to(device), attention_mask=batch[1].to(device),
                               max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
                               do_sample=False)
    # An EOS-free sequence exhausted the generation budget. Keep evidence visible.
    exhausted = sum(tokenizer.eos_token_id not in row.tolist() for row in generated)
    if exhausted:
        print(f'WARNING: {exhausted} generated SQL queries reached the token limit without EOS', flush=True)
    return [decode_sql(tokenizer, row) for row in generated], exhausted


@torch.no_grad()
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    model.eval()
    device = next(model.parameters()).device
    total_loss, total_tokens, exhausted = 0.0, 0, 0
    queries = []
    for batch in tqdm(dev_loader, desc='dev'):
        encoder_input, encoder_mask, decoder_input, targets, _ = batch
        targets = targets.to(device)
        output = model(input_ids=encoder_input.to(device), attention_mask=encoder_mask.to(device),
                       decoder_input_ids=decoder_input.to(device), labels=targets)
        tokens = targets.ne(-100).sum().item()
        total_loss += output.loss.item() * tokens
        total_tokens += tokens
        predictions, truncated = generate_queries(args, model, batch, dev_loader.dataset.tokenizer)
        queries.extend(predictions)
        exhausted += truncated
    if not total_tokens or len(queries) != len(dev_loader.dataset):
        raise ValueError('Empty or misaligned development predictions')
    save_queries_and_records(queries, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, errors = compute_metrics(gt_sql_pth, model_sql_path,
                                                         gt_record_path, model_record_path)
    Path(model_sql_path + '.generation.json').write_text(json.dumps({'count': len(queries),
          'without_eos_count': exhausted, 'num_beams': args.num_beams,
          'max_new_tokens': args.max_new_tokens}, indent=2), encoding='utf-8')
    return total_loss / total_tokens, record_f1, record_em, sql_em, sum(bool(e) for e in errors) / len(queries)


@torch.no_grad()
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    if Path(model_sql_path).exists() or Path(model_record_path).exists():
        raise FileExistsError('Test output already exists; choose a fresh --output_dir for an explicitly new run')
    model.eval()
    queries, exhausted = [], 0
    for batch in tqdm(test_loader, desc='test'):
        predictions, truncated = generate_queries(args, model, batch, test_loader.dataset.tokenizer)
        queries.extend(predictions)
        exhausted += truncated
    if len(queries) != len(test_loader.dataset):
        raise ValueError('Test predictions are not aligned with source examples')
    save_queries_and_records(queries, model_sql_path, model_record_path)
    Path(model_sql_path + '.generation.json').write_text(json.dumps({'count': len(queries),
          'without_eos_count': exhausted, 'num_beams': args.num_beams,
          'max_new_tokens': args.max_new_tokens}, indent=2), encoding='utf-8')
    checkpoint = Path(args.checkpoint_path) if args.checkpoint_path else checkpoint_directory(args) / 'best'
    source_path = Path(args.data_dir) / 'test.nl'
    manifest = {'checkpoint': str(checkpoint.resolve()), 'experiment_name': args.experiment_name,
                'model_type': 'finetuned' if args.finetune else 'from_scratch',
                'examples': len(queries), 'source_sha256': file_sha256(source_path),
                'sql_sha256': file_sha256(model_sql_path),
                'records_sha256': file_sha256(model_record_path),
                'num_beams': args.num_beams, 'max_new_tokens': args.max_new_tokens,
                'prefix': args.prefix, 'test_metrics': None,
                'checkpoint_sha256': {path.name: file_sha256(path) for path in sorted(checkpoint.glob('*'))
                                      if path.is_file() and path.suffix in {'.json', '.safetensors', '.bin', '.model'}}}
    Path(model_sql_path + '.manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return queries


def main():
    args = get_args()
    set_random_seeds(args.seed)
    if args.use_wandb:
        setup_wandb(args)
    if args.test_only:
        checkpoint = Path(args.checkpoint_path) if args.checkpoint_path else checkpoint_directory(args) / 'best'
        saved_args = json.loads((checkpoint / 'run_config.json').read_text(encoding='utf-8'))
        for name in ('prefix', 'max_source_length', 'max_target_length', 'num_beams', 'max_new_tokens', 'finetune', 'experiment_name'):
            setattr(args, name, saved_args[name])
        args.checkpoint_path = str(checkpoint)
        tokenizer = T5TokenizerFast.from_pretrained(checkpoint, local_files_only=True)
        model = load_model_from_checkpoint(args, best=True)
    else:
        tokenizer = build_tokenizer(args.model_name, args.local_files_only)
        model = initialize_model(args, tokenizer)
    loader_kwargs = dict(tokenizer=tokenizer, data_folder=args.data_dir, seed=args.seed,
                         prefix=args.prefix, max_source_length=args.max_source_length,
                         max_target_length=args.max_target_length)
    if not args.test_only:
        train_loader = get_dataloader(args.batch_size, 'train', **loader_kwargs)
        dev_loader = get_dataloader(args.test_batch_size, 'dev', **loader_kwargs)
        longest_target = max(len(example['labels']) for dataset in (train_loader.dataset, dev_loader.dataset)
                             for example in dataset.examples)
        if args.max_new_tokens < longest_target:
            raise ValueError(f'max_new_tokens={args.max_new_tokens} cannot cover longest train/dev SQL '
                             f'({longest_target} tokens including EOS)')
        optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
        train(args, model, train_loader, dev_loader, optimizer, scheduler)
        model = load_model_from_checkpoint(args, best=True)
        dev_sql, dev_records = output_paths(args, 'dev')
        metrics = eval_epoch(args, model, dev_loader, str(Path(args.data_dir) / 'dev.sql'),
                             dev_sql, args.ground_truth_records, dev_records)
        summary = dict(zip(('loss', 'record_f1', 'record_em', 'sql_em', 'sql_error_rate'), map(float, metrics)))
        (checkpoint_directory(args) / 'selected_dev_metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print('Selected checkpoint development results:', summary, flush=True)
    if args.test_only or args.predict_test:
        test_loader = get_dataloader(args.test_batch_size, 'test', **loader_kwargs)
        test_inference(args, model, test_loader, *output_paths(args, 'test'))
    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
