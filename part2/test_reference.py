"""Offline regression tests. Tiny random T5 scores are not experiment results."""
import json
import pickle
import math
import sqlite3
import utils
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, T5TokenizerFast, T5Config, T5ForConditionalGeneration
import train_t5
from load_data import T5Dataset, build_tokenizer, decode_sql, get_dataloader, normal_collate_fn
from t5_utils import checkpoint_directory, initialize_model, initialize_optimizer, load_model_from_checkpoint


def tiny_tokenizer():
    vocab = {'<pad>': 0, '</s>': 1, '<unk>': 2, 'find': 3, 'flights': 4,
             'SELECT': 5, '1': 6, '2': 7, 'WHERE': 8, '<': 9}
    backend = Tokenizer(WordLevel(vocab, unk_token='<unk>'))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single='$A </s>', special_tokens=[('</s>', 1)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>', eos_token='</s>', unk_token='<unk>')


class PipelineTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.tokenizer = tiny_tokenizer()
        for split in ('train', 'dev', 'test'):
            (self.root / f'{split}.nl').write_text('find flights\nfind\n', encoding='utf-8')
            if split != 'test':
                (self.root / f'{split}.sql').write_text('SELECT 1\nSELECT 2 WHERE 1 < 2\n', encoding='utf-8')
        self.args = train_t5.get_args(['--output_dir', str(self.root), '--data_dir', str(self.root),
                                      '--max_n_epochs', '2', '--patience_epochs', '1',
                                      '--num_beams', '1', '--max_new_tokens', '8', '--prefix', ''])
        self.model = T5ForConditionalGeneration(T5Config(vocab_size=len(self.tokenizer), d_model=16,
                    d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2, d_kv=8,
                    pad_token_id=0, eos_token_id=1, decoder_start_token_id=0, dropout_rate=0.0))

    def tearDown(self):
        self.tmp.cleanup()

    def loader(self, split, batch_size=2):
        return get_dataloader(batch_size, split, tokenizer=self.tokenizer, data_folder=self.root, prefix='')

    def test_dynamic_padding_and_exact_shift(self):
        batch = normal_collate_fn(self.loader('dev').dataset.examples)
        ids, mask, shifted, labels, initial = batch
        self.assertEqual(ids.shape, (2, 3))
        self.assertEqual(mask.tolist(), [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(labels[0].tolist(), [5, 6, 1, -100, -100, -100, -100])
        self.assertTrue(torch.equal(shifted, self.model._shift_right(labels)))
        self.assertEqual(initial.tolist(), [[0], [0]])

    def test_lengths_alignment_and_unknown_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, 'never silently truncate'):
            T5Dataset(self.root, 'dev', self.tokenizer, prefix='', max_target_length=3)
        (self.root / 'dev.sql').write_text('SELECT 1\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'misaligned'):
            self.loader('dev')
        (self.root / 'dev.sql').write_text('SELECT mystery\nSELECT 1\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'unrepresentable'):
            self.loader('dev')

    def test_test_order_needs_no_labels(self):
        loader = self.loader('test', 1)
        first, second = list(loader)
        self.assertEqual(first[0][0].tolist(), [3, 4, 1])
        self.assertEqual(second[0][0].tolist(), [3, 1])
        self.assertIsNone(loader.dataset[0]['labels'])

    def test_real_t5_training_and_generation_evaluation(self):
        optimizer = initialize_optimizer(self.args, self.model)
        before = self.model.shared.weight.detach().clone()
        loss = train_t5.train_epoch(self.args, self.model, self.loader('train'), optimizer, None)
        self.assertTrue(0 < loss < 100)
        self.assertFalse(torch.equal(before, self.model.shared.weight))
        with patch.object(self.model, 'generate', wraps=self.model.generate) as generation, \
             patch.object(train_t5, 'save_queries_and_records') as saving, \
             patch.object(train_t5, 'compute_metrics', return_value=(0.25, 0.5, 0.75, ['', 'bad'])):
            result = train_t5.eval_epoch(self.args, self.model, self.loader('dev', 1), 'gt.sql',
                                         str(self.root / 'prediction.sql'), 'gt.pkl', 'model.pkl')
            self.assertEqual(result[1:], (0.75, 0.5, 0.25, 0.5))
            self.assertEqual(generation.call_count, 2)
            for call in generation.call_args_list:
                self.assertNotIn('labels', call.kwargs)
                self.assertNotIn('decoder_input_ids', call.kwargs)
            self.assertEqual(len(saving.call_args.args[0]), 2)
        batch = next(iter(self.loader('dev')))
        with torch.no_grad():
            combined = self.model(input_ids=batch[0], attention_mask=batch[1],
                                  decoder_input_ids=batch[2], labels=batch[3]).loss.item()
        self.assertAlmostEqual(result[0], combined, places=5)

    def test_zero_f1_checkpoint_early_stop_and_roundtrip(self):
        optimizer = initialize_optimizer(self.args, self.model)
        with patch.object(train_t5, 'train_epoch', return_value=1.0), \
             patch.object(train_t5, 'eval_epoch', return_value=(1.0, 0.0, 0.0, 0.0, 1.0)) as evaluation:
            train_t5.train(self.args, self.model, self.loader('train'), self.loader('dev'), optimizer, None)
        self.assertEqual(evaluation.call_count, 2)
        checkpoint = checkpoint_directory(self.args) / 'best'
        self.assertEqual(json.loads((checkpoint / 'metrics.json').read_text())['epoch'], 1)
        self.assertTrue((checkpoint / 'tokenizer.json').exists())
        loaded = load_model_from_checkpoint(self.args, best=True)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value.cpu(), loaded.state_dict()[key].cpu()))
        with self.assertRaises(FileExistsError):
            train_t5.train(self.args, self.model, self.loader('train'), self.loader('dev'), optimizer, None)

    def test_real_tokenizer_roundtrip_all_labeled_sql(self):
        asset = Path(__file__).resolve().parents[2] / 'assets/t5-small'
        if not (asset / 'tokenizer.json').exists():
            self.skipTest('Optional local tokenizer-only asset unavailable')
        tokenizer = build_tokenizer(str(asset), local_files_only=True)
        folder = Path(__file__).resolve().parent / 'data'
        for split in ('train', 'dev'):
            data = T5Dataset(folder, split, tokenizer)
            raw = (folder / f'{split}.sql').read_text().splitlines()
            self.assertEqual(raw, [decode_sql(tokenizer, example['labels']) for example in data.examples])
        tokenizer.save_pretrained(self.root / 'tokenizer')
        restored = T5TokenizerFast.from_pretrained(self.root / 'tokenizer', local_files_only=True)
        self.assertEqual(restored.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(decode_sql(restored, restored('SELECT 1 WHERE 1 <= 2')['input_ids']), 'SELECT 1 WHERE 1 <= 2')
        query = "SELECT 'a < b', \"x<y\" WHERE x <= 10 AND y <> 2"
        with patch.object(tokenizer, 'decode', return_value=query):
            self.assertEqual(decode_sql(tokenizer, []), query)

    def test_test_export_manifest_order_and_overwrite_guard(self):
        sql_path, record_path = train_t5.output_paths(self.args, 'test')
        self.assertTrue(sql_path.endswith('t5_ft_experiment_test.sql'))
        with patch.object(train_t5, 'generate_queries', return_value=(['SELECT 1', 'SELECT 2'], 0)):
            queries = train_t5.test_inference(self.args, self.model, self.loader('test'), sql_path, record_path)
        self.assertEqual(Path(sql_path).read_text().splitlines(), queries)
        with open(record_path, 'rb') as handle:
            records, errors = pickle.load(handle)
        self.assertEqual(records, [[(1,)], [(2,)]])
        self.assertEqual(errors, ['', ''])
        manifest = json.loads(Path(sql_path + '.manifest.json').read_text())
        self.assertEqual(manifest['examples'], 2)
        self.assertIsNone(manifest['test_metrics'])
        with self.assertRaises(FileExistsError):
            train_t5.test_inference(self.args, self.model, self.loader('test'), sql_path, record_path)
        self.args.finetune = False
        self.assertTrue(train_t5.output_paths(self.args, 'test')[0].endswith('t5_ft_experiment_ec_test.sql'))

    def test_actual_generation_sql_execution_and_export_chain(self):
        database = self.root / 'temporary.db'
        with sqlite3.connect(database) as connection:
            connection.execute('CREATE TABLE smoke (value INTEGER)')
        original_db = utils.DB_PATH
        utils.DB_PATH = str(database)
        try:
            gold_sql, gold_records = self.root / 'gold.sql', self.root / 'gold.pkl'
            utils.save_queries_and_records(['SELECT 1', 'SELECT 2 WHERE 1 < 2'], str(gold_sql), str(gold_records))
            values = train_t5.eval_epoch(self.args, self.model, self.loader('dev'), str(gold_sql),
                     str(self.root / 'actual_dev.sql'), str(gold_records), str(self.root / 'actual_dev.pkl'))
            self.assertTrue(all(math.isfinite(float(value)) for value in values))
            sql_path, record_path = train_t5.output_paths(self.args, 'test')
            queries = train_t5.test_inference(self.args, self.model, self.loader('test'), sql_path, record_path)
            self.assertEqual(len(queries), 2)
            with open(record_path, 'rb') as handle:
                records, errors = pickle.load(handle)
            self.assertEqual(len(records), 2)
            self.assertEqual(len(errors), 2)
            manifest = json.loads(Path(sql_path + '.manifest.json').read_text())
            self.assertEqual(manifest['sql_sha256'], train_t5.file_sha256(sql_path))
            self.assertIsNone(manifest['test_metrics'])
        finally:
            utils.DB_PATH = original_db

    def test_initializer_preserves_padded_original_vocabulary(self):
        model = T5ForConditionalGeneration(T5Config(vocab_size=32, d_model=16, d_ff=32,
                 num_layers=1, num_heads=2, d_kv=8, decoder_start_token_id=0))
        with patch('t5_utils.T5ForConditionalGeneration.from_pretrained', return_value=model):
            loaded = initialize_model(self.args, self.tokenizer)
        self.assertEqual(loaded.config.vocab_size, 32)

    def test_default_is_finetune_without_test_prediction(self):
        args = train_t5.get_args([])
        self.assertTrue(args.finetune)
        self.assertFalse(args.predict_test)
        self.assertFalse(args.test_only)


if __name__ == '__main__':
    unittest.main()
