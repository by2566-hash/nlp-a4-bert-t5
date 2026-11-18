from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import random
import argparse
from utils import custom_transform, TRANSFORM_VERSION, TRANSFORM_SEED, TRANSFORM_PROBABILITY
import os
import json
import platform
import importlib.metadata

# Set seed
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Tokenize the input
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Core training function
def do_train(args, model, train_dataloader, save_dir="./out"):
    if args.num_epochs < 1 or args.learning_rate <= 0 or len(train_dataloader) == 0:
        raise ValueError("Training requires positive epochs/rate and a nonempty loader.")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    model.train()
    progress_bar = tqdm(range(num_training_steps))

    training_device = next(model.parameters()).device
    epoch_history = []
    for epoch in range(num_epochs):
        loss_sum, example_count = 0.0, 0
        for batch in train_dataloader:
            batch = {key: value.to(training_device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss in epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            batch_size = batch["labels"].shape[0]
            loss_sum += loss.detach().item() * batch_size
            example_count += batch_size
            progress_bar.update(1)
        epoch_history.append({"epoch": epoch + 1,
                              "mean_train_loss": loss_sum / max(example_count, 1),
                              "examples": example_count})
        print(epoch_history[-1])
    progress_bar.close()

    print("Training completed...")
    print("Saving Model....")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    if "tokenizer" in globals():
        tokenizer.save_pretrained(save_dir)
    evidence = {
        "arguments": vars(args), "seed": 0, "device": str(training_device),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "training_steps": num_training_steps, "epochs": epoch_history,
        "augmentation": getattr(train_dataloader, "augmentation_metadata", None),
    }
    with open(os.path.join(save_dir, "training_log.json"), "w") as stream:
        json.dump(evidence, stream, indent=2)

    return


# Core evaluation function
def do_eval(eval_dataloader, output_dir, out_file, transformed=False):
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.to(device)
    model.eval()

    correct, total = 0, 0
    with open(out_file, "w") as stream:
        for batch in tqdm(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].numel()
            # Preserve the starter contract: alternating prediction and label.
            for pred, label in zip(predictions, batch["labels"]):
                stream.write(f"{pred.item()}\n")
                stream.write(f"{label.item()}\n")
    if total == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    score = {"accuracy": correct / total}
    metadata = {
        **score, "correct": correct, "total": total,
        "model_dir": os.path.abspath(output_dir),
        "transformation": ({"version": TRANSFORM_VERSION, "seed": TRANSFORM_SEED,
                            "probability": TRANSFORM_PROBABILITY} if transformed else None),
    }
    with open(os.path.splitext(out_file)[0] + ".json", "w") as stream:
        json.dump(metadata, stream, indent=2)

    return score


# Created a dataladoer for the augmented training dataset
def create_augmented_dataloader(args, dataset):
    from datasets import concatenate_datasets

    original = dataset["train"]
    if len(original) < 5000:
        raise ValueError("Augmentation requires at least 5,000 original training examples.")
    indices = random.Random(42).sample(range(len(original)), 5000)
    transformed = original.select(indices).map(custom_transform, load_from_cache_file=False)
    combined = concatenate_datasets([original, transformed])
    tokenized = combined.map(tokenize_function, batched=True, load_from_cache_file=False)
    tokenized = tokenized.remove_columns(["text"]).rename_column("label", "labels")
    tokenized.set_format("torch")
    train_dataloader = DataLoader(
        tokenized, shuffle=True, batch_size=args.batch_size,
        generator=torch.Generator().manual_seed(0),
    )
    train_dataloader.augmentation_metadata = {
        "original_examples": len(original), "added_examples": len(transformed),
        "source_split": "train", "sampling_seed": 42, "source_indices": indices,
        "transform_version": TRANSFORM_VERSION, "transform_seed": TRANSFORM_SEED,
        "transform_probability": TRANSFORM_PROBABILITY,
    }

    return train_dataloader


# Create a dataloader for the transformed test set
def create_transformed_dataloader(args, dataset, debug_transformation):
    # Print 5 random transformed examples
    if debug_transformation:
        small_dataset = dataset["test"].shuffle(seed=42).select(range(5))
        small_transformed_dataset = small_dataset.map(custom_transform, load_from_cache_file=False)
        for k in range(5):
            print("Original Example ", str(k))
            print(small_dataset[k])
            print("\n")
            print("Transformed Example ", str(k))
            print(small_transformed_dataset[k])
            print('=' * 30)

        return None

    transformed_dataset = dataset["test"].map(custom_transform, load_from_cache_file=False)
    transformed_tokenized_dataset = transformed_dataset.map(tokenize_function, batched=True, load_from_cache_file=False)
    transformed_tokenized_dataset = transformed_tokenized_dataset.remove_columns(["text"])
    transformed_tokenized_dataset = transformed_tokenized_dataset.rename_column("label", "labels")
    transformed_tokenized_dataset.set_format("torch")

    transformed_val_dataset = transformed_tokenized_dataset
    eval_dataloader = DataLoader(transformed_val_dataset, batch_size=args.batch_size)

    return eval_dataloader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--train", action="store_true", help="train a model on the training data")
    parser.add_argument("--train_augmented", action="store_true", help="train a model on the augmented training data")
    parser.add_argument("--eval", action="store_true", help="evaluate model on the test set")
    parser.add_argument("--eval_transformed", action="store_true", help="evaluate model on the transformed test set")
    parser.add_argument("--model_dir", type=str, default="./out")
    parser.add_argument("--debug_train", action="store_true",
                        help="use a subset for training to debug your training loop")
    parser.add_argument("--debug_transformation", action="store_true",
                        help="print a few transformed examples for debugging")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch_size must be positive.")
    if args.train and args.train_augmented:
        parser.error("Run original and augmented training separately for comparable initialization.")
    if args.train_augmented and args.debug_train:
        parser.error("Augmented training uses the full training split plus 5,000 examples.")

    global device
    global tokenizer

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    from datasets import load_dataset, DatasetDict
    dataset = load_dataset("imdb")
    if args.debug_transformation:
        create_transformed_dataloader(args, dataset, True)
        raise SystemExit(0)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Tokenize the dataset
    # The unlabeled IMDB split is not used by this supervised experiment.
    selected = DatasetDict({"train": dataset["train"], "test": dataset["test"]})
    if args.debug_train:
        selected = DatasetDict({
            "train": selected["train"].shuffle(seed=42).select(range(4000)),
            "test": selected["test"].shuffle(seed=42).select(range(1000)),
        })
    tokenized_dataset = selected.map(tokenize_function, batched=True)

    # Prepare dataset for use by model
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    small_train_dataset = tokenized_dataset["train"]
    small_eval_dataset = tokenized_dataset["test"]

    # Create dataloaders for iterating over the dataset
    if args.debug_train:
        train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(small_eval_dataset, batch_size=args.batch_size)
        print(f"Debug training...")
        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"len(eval_dataloader): {len(eval_dataloader)}")
    else:
        train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(tokenized_dataset["test"], batch_size=args.batch_size)
        print(f"Actual training...")
        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"len(eval_dataloader): {len(eval_dataloader)}")

    # Train model on the original training dataset
    if args.train:
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        model.to(device)
        do_train(args, model, train_dataloader, save_dir="./out_debug" if args.debug_train else "./out")
        # Change eval dir
        args.model_dir = "./out_debug" if args.debug_train else "./out"

    # Train model on the augmented training dataset
    if args.train_augmented:
        train_dataloader = create_augmented_dataloader(args, dataset)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        model.to(device)
        do_train(args, model, train_dataloader, save_dir="./out_augmented")
        # Change eval dir
        args.model_dir = "./out_augmented"

    # Evaluate the trained model on the original test dataset
    if args.eval:
        out_file = os.path.basename(os.path.normpath(args.model_dir))
        if args.debug_train and not out_file.endswith("_debug"):
            out_file += "_debug"
        out_file = out_file + "_original.txt"
        score = do_eval(eval_dataloader, args.model_dir, out_file)
        print("Score: ", score)

    # Evaluate the trained model on the transformed test dataset
    if args.eval_transformed:
        out_file = os.path.basename(os.path.normpath(args.model_dir))
        if args.debug_train and not out_file.endswith("_debug"):
            out_file += "_debug"
        out_file = out_file + "_transformed.txt"
        eval_transformed_dataloader = create_transformed_dataloader(args, selected, args.debug_transformation)
        score = do_eval(eval_transformed_dataloader, args.model_dir, out_file, transformed=True)
        print("Score: ", score)
