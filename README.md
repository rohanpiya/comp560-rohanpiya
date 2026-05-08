# comp560-rohanpiya

Independent research project for COMP 560 at Dickinson College.

The main project investigates whether Chain-of-Thought reasoning improves a small language model's ability to learn integer addition — and whether the resulting reasoning can generalize beyond the training distribution.

---

## Repository Structure

```
comp560-rohanpiya/
├── addition-experiments-new/   # main project (see below)
├── comma-semicolon/            # early warmup experiments
└── translate-roman-numbers/    # early warmup experiments
```

---

## Main Project: Integer Addition with Chain-of-Thought

**Folder:** `addition-experiments-new/`

A GPT-style language model (~0.79M parameters) trained from scratch on synthetic integer addition problems, comparing plain format vs. Chain-of-Thought (CoT) format across 2-, 3-, and 4-digit addition. A final experiment trains a single model on all three digit counts simultaneously.

### Research Questions

- Does CoT improve in-distribution accuracy, and by how much?
- Does CoT help with carry propagation specifically?
- Do CoT models generalize to unseen digit counts?
- Can mixed-digit training overcome the generalization failure of per-digit models?

### Data Formats

**Plain:**
```
47+26=73
```

**CoT** (column-by-column arithmetic, ones column first):
```
47+26=7+6=13;4+2+1=7;73
473+261=3+1=4;7+6+0=13;4+2+1=7;734
4731+2614=1+4=5;3+1+0=4;7+6+0=13;4+2+1=7;7345
```

**Mixed CoT** (same format, all digit counts in one dataset):
```
# 2-digit, 3-digit, and 4-digit examples mixed together
```

### Models Trained

| Model | Format | Digits | Iters |
|---|---|---|---|
| Plain 2-digit | Plain | 2 | 3,000 |
| CoT 2-digit | CoT | 2 | 3,000 |
| Plain 3-digit | Plain | 3 | 3,000 |
| CoT 3-digit | CoT | 3 | 3,000 |
| Plain 4-digit | Plain | 4 | 3,000 |
| CoT 4-digit | CoT | 4 | 3,000 |
| Mixed CoT | CoT | 2+3+4 | 3,000 |

### Key Results

| Model | Overall | No-carry | Carry |
|---|---|---|---|
| Plain 2-digit | 96.37% | 99.90% | 95.23% |
| CoT 2-digit | 99.90% | 99.60% | 100.00% |
| Plain 3-digit | 61.60% | 94.44% | 56.64% |
| CoT 3-digit | 99.71% | 100.00% | 99.67% |
| Plain 4-digit | 94.85% | 94.30% | 94.90% |
| CoT 4-digit | 99.66% | 100.00% | 99.63% |
| Mixed CoT (2-digit) | 97.01% | 98.43% | 96.55% |
| Mixed CoT (3-digit) | 98.42% | 98.40% | 98.42% |
| Mixed CoT (4-digit) | 89.79% | 99.34% | 89.01% |

All per-digit models (plain and CoT) produce 0% accuracy on out-of-distribution digit counts.
Mixed CoT produces correct 5-step reasoning on 5-digit inputs (98.3% correct step count) despite never seeing 5-digit problems during training.

### Project Structure

```
addition-experiments-new/
├── generators.py           # all data generators (plain, CoT, mixed)
├── prepare_utils.py        # shared dataset-building logic
├── evaluate.py             # main evaluation script
├── evaluate_new.py         # evaluation with CoT step diagnostics
├── evaluate_mixed.py       # evaluation for the mixed CoT model
│
├── data/
│   ├── twoDigit/
│   ├── twoDigitCoT/
│   ├── threeDigit/
│   ├── threeDigitCoT/
│   ├── fourDigit/
│   ├── fourDigitCoT/
│   └── mixedCoT/
│
├── config/
│   ├── config_train_twoDigit.py
│   ├── config_train_twoDigitCoT.py
│   ├── config_train_threeDigit.py
│   ├── config_train_threeDigitCoT.py
│   ├── config_train_fourDigit.py
│   ├── config_train_fourDigitCoT.py
│   └── config_train_mixedCoT.py
│
└── out/
    ├── twoDigit/           ckpt.pt
    ├── twoDigitCoT/        ckpt.pt
    ├── threeDigit/         ckpt.pt
    ├── threeDigitCoT/      ckpt.pt
    ├── fourDigit/          ckpt.pt
    ├── fourDigitCoT/       ckpt.pt
    └── mixedCoT/           ckpt.pt
```

### Quickstart

```bash
cd addition-experiments-new

# 1. generate a dataset
python data/twoDigitCoT/prepare.py

# 2. train
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
python -u ../../comp560-nanoGPT/train.py config/config_train_twoDigitCoT.py

# 3. evaluate (in-distribution)
python evaluate.py --dataset twoDigitCoT --train_digits 2 --eval_digits 2 --cot

# 4. evaluate mixed CoT
python evaluate_mixed.py --eval_digits 5 --num_samples 10000
```

### Model Config (identical for all runs)

| Param | Value |
|---|---|
| Parameters | ~0.79M |
| n_layer | 4 |
| n_head | 4 |
| n_embd | 128 |
| block_size | 128 |
| batch_size | 32 |
| max_iters | 3,000 |
| learning_rate | 1e-3 → 1e-4 (cosine) |
| device | CPU (MacBook Pro M3) |

---

## Warmup Experiments

Two smaller projects done earlier in the semester to get familiar with nanoGPT before the main project.

### Comma-Semicolon (`comma-semicolon/`)

Three experiments testing whether a character-level model can learn symbol roles from structured synthetic data — commas as separators, semicolons as line terminators, dollar signs as field delimiters. Ran at 200 and 2000 iterations. Main finding: the model learns structural rules (e.g., semicolons always precede newlines) even when content is noisy.

### Roman Numeral Translation (`translate-roman-numbers/`)

Three datasets of increasing complexity: Roman → English, Roman → Spanish, and Roman → all five representations simultaneously (English, Spanish, Arabic numerals, Devanagari numerals, Nepali words). Ran at 200 and 2000 iterations. Main finding: 2000 iterations was sufficient for near-perfect accuracy on all three datasets, including accented Spanish characters and Devanagari script.

---

## Dependencies

```bash
pip install torch numpy wandb
```

Requires [comp560-nanoGPT](https://github.com/karpathy/nanoGPT) as a sibling directory:

```
parent/
├── comp560-rohanpiya/
└── comp560-nanoGPT/
```

---

## Report

The final report (`addition_report_final.docx`) covers the full addition experiments including Mixed CoT. It is in the `addition-experiments-new/` folder.
