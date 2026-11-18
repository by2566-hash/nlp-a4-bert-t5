"""Offline unit checks only; these do not validate IMDB/BERT integration or scores."""
import json
import importlib.util
import random
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from utils import custom_transform
import main as training
from main import do_train
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer


class TransformTests(unittest.TestCase):
    def test_deterministic_and_does_not_mutate(self):
        example = {"text": "A remarkably compelling performance with fantastic storytelling. " * 20,
                   "label": 1, "other": "untouched"}
        original = dict(example)
        first = custom_transform(example)
        random.seed(981)
        for _ in range(20):
            random.random()
        second = custom_transform(example)
        self.assertEqual(first, second)
        self.assertEqual(example, original)
        self.assertEqual(first["label"], 1)
        self.assertEqual(first["other"], "untouched")
        self.assertNotEqual(first["text"], example["text"])

    def test_protected_words(self):
        text = "No NOT never neither nothing without cannot hardly barely scarcely don't isn't isn't!"
        self.assertEqual(custom_transform({"text": text, "label": 0})["text"], text)

    def test_preserves_layout_and_word_endpoints(self):
        text = "Beautiful\tstorytelling, delightful\nperformances! " * 30
        result = custom_transform({"text": text, "label": 1})["text"]
        self.assertEqual(re.sub(r"[A-Za-z]", "", result), re.sub(r"[A-Za-z]", "", text))
        for before, after in zip(re.findall(r"[A-Za-z]+", text), re.findall(r"[A-Za-z]+", result)):
            self.assertEqual((before[0], before[-1]), (after[0], after[-1]))
            self.assertEqual(sorted(before), sorted(after))


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, input_ids, labels):
        logits = self.linear(input_ids)
        return SimpleNamespace(loss=torch.nn.functional.cross_entropy(logits, labels), logits=logits)

    def save_pretrained(self, directory):
        torch.save(self.state_dict(), Path(directory) / "tiny_weights.pt")


class TrainingTests(unittest.TestCase):
    def test_real_autograd_updates_and_checkpoint_reload(self):
        torch.manual_seed(7)
        model = TinyClassifier()
        samples = [
            {"input_ids": torch.tensor([2., 0.]), "labels": torch.tensor(0)},
            {"input_ids": torch.tensor([0., 2.]), "labels": torch.tensor(1)},
        ] * 4
        loader = DataLoader(samples, batch_size=2)
        before = model.linear.weight.detach().clone()
        batch = next(iter(loader))
        initial_loss = model(**batch).loss.item()
        with tempfile.TemporaryDirectory() as directory:
            do_train(SimpleNamespace(learning_rate=0.1, num_epochs=4), model, loader, directory)
            self.assertFalse(torch.equal(before, model.linear.weight))
            self.assertLess(model(**batch).loss.item(), initial_loss)
            log = json.loads((Path(directory) / "training_log.json").read_text())
            self.assertEqual(log["training_steps"], 16)
            self.assertEqual([row["examples"] for row in log["epochs"]], [8] * 4)
            reloaded = TinyClassifier()
            reloaded.load_state_dict(torch.load(Path(directory) / "tiny_weights.pt", weights_only=True))
            self.assertTrue(torch.equal(model(**batch).logits, reloaded(**batch).logits))


class EvaluationTests(unittest.TestCase):
    def test_real_bert_training_reload(self):
        torch.manual_seed(9)
        config = BertConfig(vocab_size=8, hidden_size=8, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=16, num_labels=2,
                            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        model = BertForSequenceClassification(config)
        samples = [{"input_ids": torch.tensor([2, 3]), "labels": torch.tensor(1)}] * 8
        loader = DataLoader(samples, batch_size=2)
        batch = next(iter(loader))
        initial_loss = model(**batch).loss.item()
        initial_weights = model.bert.embeddings.word_embeddings.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            do_train(SimpleNamespace(learning_rate=0.01, num_epochs=2), model, loader, directory)
            self.assertFalse(torch.equal(initial_weights, model.bert.embeddings.word_embeddings.weight))
            model.eval()
            self.assertLess(model(**batch).loss.item(), initial_loss)
            reloaded = BertForSequenceClassification.from_pretrained(directory).eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(model(**batch).logits, reloaded(**batch).logits))

    def test_real_bert_checkpoint_eval_contract(self):
        config = BertConfig(vocab_size=8, hidden_size=8, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=16, num_labels=2)
        model = BertForSequenceClassification(config)
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.copy_(torch.tensor([0., 3.]))
        samples = [{"input_ids": torch.tensor([2, 3]), "labels": torch.tensor(label)}
                   for label in [0, 1, 0, 1]]
        previous_device = getattr(training, "device", None)
        training.device = torch.device("cpu")
        try:
            with tempfile.TemporaryDirectory() as directory:
                model.save_pretrained(directory)
                output = Path(directory) / "predictions.txt"
                score = training.do_eval(DataLoader(samples, batch_size=2), directory,
                                         str(output), transformed=True)
                self.assertEqual(score, {"accuracy": 0.5})
                self.assertEqual(output.read_text(), "1\n0\n1\n1\n1\n0\n1\n1\n")
                metadata = json.loads(output.with_suffix(".json").read_text())
                self.assertEqual((metadata["correct"], metadata["total"]), (2, 4))
                self.assertEqual(metadata["transformation"]["version"], "internal_transposition_v2_p60")
        finally:
            if previous_device is None:
                del training.device
            else:
                training.device = previous_device


@unittest.skipUnless(importlib.util.find_spec("datasets"), "datasets is not installed")
class AugmentationTests(unittest.TestCase):
    def test_real_datasets_training_only_augmentation(self):
        from datasets import Dataset, DatasetDict
        train = Dataset.from_dict({
            "text": [f"Wonderful performance number {i}" for i in range(5001)],
            "label": [i % 2 for i in range(5001)],
        })
        test = Dataset.from_dict({"text": ["TEST MUST NOT BE AUGMENTED"], "label": [0]})
        source = DatasetDict({"train": train, "test": test})
        original_test = test.to_dict()
        previous_tokenizer = getattr(training, "tokenizer", None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                vocab = Path(directory) / "vocab.txt"
                vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nwonderful\nperformance\nnumber\n")
                training.tokenizer = BertTokenizer(str(vocab), model_max_length=12)
                loader = training.create_augmented_dataloader(SimpleNamespace(batch_size=16), source)
                self.assertEqual(len(loader.dataset), 10001)
                indices = loader.augmentation_metadata["source_indices"]
                self.assertEqual(len(indices), 5000)
                self.assertEqual(len(set(indices)), 5000)
                self.assertTrue(all(0 <= i < len(train) for i in indices))
                self.assertEqual(loader.augmentation_metadata["source_split"], "train")
                self.assertEqual(source["test"].to_dict(), original_test)
                self.assertEqual(len(source["train"]), 5001)
                expected_labels = train["label"] + [train[i]["label"] for i in indices]
                self.assertEqual(loader.dataset["labels"].tolist(), expected_labels)
        finally:
            if previous_tokenizer is None:
                del training.tokenizer
            else:
                training.tokenizer = previous_tokenizer


if __name__ == "__main__":
    unittest.main()
