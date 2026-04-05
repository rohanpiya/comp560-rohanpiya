# addition-experiments

NanoGPT trained to perform integer addition, with and without Chain-of-Thought.
Research question: does CoT improve performance and generalization?

---

## Project structure

```
addition-experiments/
├── generators.py          # all data generators (CoT + plain, all digit counts)
├── prepare_utils.py       # shared dataset-building logic
├── evaluate.py            # unified evaluation script
│
├── config/
│   ├── base.py            # shared model + training config (identical for all runs)
│   ├── train_twoDigit.py
│   ├── train_twoDigitCoT.py
│   ├── train_threeDigit.py
│   └── train_threeDigitCoT.py
│
├── data/
│   ├── twoDigit/          prepare.py  →  train.bin  val.bin  meta.pkl
│   ├── twoDigitCoT/       prepare.py  →  ...
│   ├── threeDigit/        prepare.py  →  ...
│   └── threeDigitCoT/     prepare.py  →  ...
│
└── out/
    ├── twoDigit/          ckpt.pt
    ├── twoDigitCoT/       ckpt.pt
    ├── threeDigit/        ckpt.pt
    └── threeDigitCoT/     ckpt.pt
```

---

## Model config (fixed for ALL experiments)

| Param        | Value | Reason                                     |
|--------------|-------|--------------------------------------------|
| n_layer      | 4     | sufficient capacity for algorithmic tasks  |
| n_head       | 4     | matches n_embd=128 (32 dims/head)          |
| n_embd       | 128   | ~0.8M params — small but trainable         |
| block_size   | 128   | fits longest 3-digit CoT example (~42 ch)  |
| batch_size   | 32    | stable gradients, reasonable speed on CPU  |
| max_iters    | 3000  | enough for convergence on 100k examples    |
| carry split  | 50/50 | balanced — model sees equal carry/no-carry |

---

## Workflow

### Step 1 — Generate datasets (run once per dataset)

```bash
cd addition-experiments

python data/twoDigit/prepare.py
python data/twoDigitCoT/prepare.py
python data/threeDigit/prepare.py
python data/threeDigitCoT/prepare.py
```

### Step 2 — Train models

Run from the `comp560-nanoGPT/` directory (or wherever your train.py lives):

```bash
python train.py addition-experiments/config/train_twoDigit.py
python train.py addition-experiments/config/train_twoDigitCoT.py
python train.py addition-experiments/config/train_threeDigit.py
python train.py addition-experiments/config/train_threeDigitCoT.py
```

### Step 3 — Evaluate

Run from `addition-experiments/`:

```bash
# 2-digit plain model
python evaluate.py --dataset twoDigit     --train_digits 2 --eval_digits 1
python evaluate.py --dataset twoDigit     --train_digits 2 --eval_digits 2
python evaluate.py --dataset twoDigit     --train_digits 2 --eval_digits 3 --num_samples 2000

# 2-digit CoT model
python evaluate.py --dataset twoDigitCoT  --train_digits 2 --eval_digits 1 --cot
python evaluate.py --dataset twoDigitCoT  --train_digits 2 --eval_digits 2 --cot
python evaluate.py --dataset twoDigitCoT  --train_digits 2 --eval_digits 3 --cot --num_samples 2000

# 3-digit plain model
python evaluate.py --dataset threeDigit    --train_digits 3 --eval_digits 1
python evaluate.py --dataset threeDigit    --train_digits 3 --eval_digits 2
python evaluate.py --dataset threeDigit    --train_digits 3 --eval_digits 3 --num_samples 2000
python evaluate.py --dataset threeDigit    --train_digits 3 --eval_digits 4 --num_samples 2000

# 3-digit CoT model
python evaluate.py --dataset threeDigitCoT --train_digits 3 --eval_digits 1 --cot
python evaluate.py --dataset threeDigitCoT --train_digits 3 --eval_digits 2 --cot
python evaluate.py --dataset threeDigitCoT --train_digits 3 --eval_digits 3 --cot --num_samples 2000
python evaluate.py --dataset threeDigitCoT --train_digits 3 --eval_digits 4 --cot --num_samples 2000
```

---

## Data formats

**Plain:**
```
47+26=73
```

**CoT (2-digit):**
```
47+26=6+7=13;4+2+1=7;73
```

**CoT (3-digit):**
```
473+261=3+1=4;7+6+0=13;4+2+1=7;734
```

---

## Evaluation flags

| Flag            | Default | Description                                    |
|-----------------|---------|------------------------------------------------|
| --dataset       | required| Dataset name (matches data/ and out/ folders)  |
| --train_digits  | required| Digits the model was trained on                |
| --eval_digits   | required| Digits to evaluate on                          |
| --cot           | off     | Use CoT extraction                             |
| --num_samples   | 2000    | Random samples for eval_digits >= 3            |
| --batch_size    | 32      | Generation batch size                          |
| --pad_eval      | off     | Zero-pad inputs to train_digits width          |
| --seed          | 42      | Reproducibility seed                           |
