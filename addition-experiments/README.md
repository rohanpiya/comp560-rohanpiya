# Addition Experiments with nanoGPT

This project explores whether a small character-level transformer (nanoGPT) can learn arithmetic through simple prompt-response training.

The core idea is simple:
Short input → Short output

Example:
2+3= → 5

---

## Goals

### 1. Framework Verification
Build a clean and reusable framework for prompt-response style tasks using nanoGPT.

### 2. Learning Behavior
Test whether the model can:
- Memorize simple addition
- Generalize to unseen inputs
- Handle carry operations

### 3. Scaling
Gradually increase difficulty:
- 1-digit → 2-digit → 3-digit
- Standard output → Chain-of-Thought → Scratchpad (planned)

---

## Experiments

### 1. Single Digit Addition
**ID:** `basic`

- Task: Learn addition from 0–9  
- Format:
- a+b=c\n

Example:

2+3=5


- Goal:
- Check memorization
- Verify training pipeline

---

### 2. Two Digit Addition
**ID:** `intermediate`

- Task: Learn addition from 00–99  
- Dataset size: 10,000 samples  

- Format:
12+34=46

- Goal:
- Test generalization
- Introduce carry handling

---

### 3. Two Digit Addition with Chain-of-Thought
**ID:** `twoDigitCoT`

- Format:
55+82=5+2=7;5+8+0=13;137


- Goal:
- Encourage reasoning
- Improve carry handling
- Study structured outputs

---

### 4. (Planned) Three Digit Evaluation
- Evaluate generalization to unseen complexity
- Note: Full evaluation is expensive (1M pairs)

---

### 5. (Planned) Scratchpad Reasoning
- Extend CoT into more explicit intermediate states

---

## Directory Structure
addition-experiments/
│
├── data/
│ ├── basic/
│ ├── basic_carry_5percent/
│ ├── twoDigitCarry/
│ ├── twoDigitCoT/
│ └── twoDigitSimple/
│
├── config/
│ ├── basic.py
│ ├── basic_carry_5percent.py
│ ├── twoDigitCarry.py
│ └── twoDigitCoT.py
│ ├── twoDigitSimple.py
│
├── out/
│
├── generators.py
├── cotGenerators.py
├── evaluation.py
└──  README.md


---

## How to Run

Run all commands from the project root (addition-experiments).

---

### 1. Data Preparation

Generate dataset binaries:

```bash
python data/basic/prepare.py
```

### 2. Training


```bash
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
python -u ../../comp560-nanoGPT/train.py config/basic.py
```

### 3. Sampling (Quick Test)

Edit config:
```bash
start = "2+2="
max_new_tokens = 5
```

Then run:
```bash
python sample.py
```

### 4. Evaluation
To evaluate model trained without CoT:
```bash
python evaluation.py --dataset [dataset] --train_digits [number of digits trained] --eval_digits [number of digits to test the model on]
```
For example:
```bash
python evaluation.py --dataset [dataset] --train_digits 2 --eval_digits 1
```

To evaluate model trained with CoT:
```bash
python evaluation.py --dataset [dataset] --train_digits [number of digits trained] --eval_digits [number of digits to test the model on] --cot
```
For example:
```bash
python evaluation.py --dataset [dataset] --train_digits 2 --eval_digits 1 --cot
```

Important Notes
### 1. Output Parsing (CoT)

For Chain-of-Thought outputs:

- Only the final answer should be evaluated
- Ignore extra generated text

### 2. Evaluation Scaling

Evaluation time grows quickly:

Digits	Total Pairs
- 1	100
- 2	10,000
- 3	1,000,000

Use sampling for large cases.

### 3. Token Limits

Recommended:

max_new_tokens ≈ 10 × num_digits

### Key Findings (So Far)
- The model easily memorizes 1-digit addition
- It learns 2-digit addition with high accuracy
- Chain-of-Thought improves structure but requires careful parsing
- Generation control (tokens, stopping) is critical

### Future Work
- 3-digit generalization
- Scratchpad reasoning
- Error analysis (carry vs no-carry)
- Extending to subtraction or other tasks
