# NLP Assignment 4: Q1-Q7 implementation

It contains only the code needed for the BERT and pretrained T5 experiments; no extra-credit experiment was run. Course submission and sharing permissions must be handled separately.

The official starter is https://github.com/gregdurrett/nyu-dsga1011-a4 at commit `e67bb117926a8179baf8eb00ee4d94544408e550`. Obtain the released Part 2 `data/` directory and ground-truth development records from that starter. No hidden test labels are used.

Use Python 3.11 and **separate environments** for Part 1 and Part 2, installing each directory's official `requirements.txt`. Full training requires an allocated CUDA GPU. Run commands from the relevant part directory.

## Part 1

The included `utils.py` implements the final fixed transformation, `internal_transposition_v2_p60`, probability 0.60 and seed 0. It is a disclosed follow-up to the earlier p30 experiment, not an untouched-test selection. The baseline training code is unchanged.

```bash
python main.py --train --eval --debug_train
python main.py --train --eval
python main.py --eval_transformed --model_dir out
python main.py --train_augmented --eval --eval_transformed
```

These command forms were used on HPC. The four final output names are `out_original.txt`, `out_transformed.txt`, `out_augmented_original.txt`, and `out_augmented_transformed.txt`. Each contains alternating prediction/label lines. Preserve existing checkpoints and outputs before rerunning.

## Part 2

```bash
python train_t5.py --finetune --experiment_name baseline_lr3e4
python train_t5.py --finetune --experiment_name comparison_lr1e4 --learning_rate 1e-4
```

The configuration selected by development Record F1 is `baseline_lr3e4/best`. Final inference was performed once using the saved model and tokenizer:

```bash
python train_t5.py --test_only --checkpoint_path checkpoints/ft_experiments/baseline_lr3e4/best
```

Outputs are `results/t5_ft_experiment_test.sql` and `records/t5_ft_experiment_test.pkl`. The pickle contains `(records, error_msgs)`, in original input order. Development scores and output checks do not establish the hidden test score.

`source_manifest.json` records the provenance and SHA-256 of the copied implementation files. Model weights are separate artifacts and are not included in this code directory.
