Roman Number Translation Experiments

This project explores whether a small character-level language model can learn structured number translations starting from Roman numerals. Multiple datasets of increasing complexity were used to test how well the model scales from simple bilingual translation to multilingual and multi-format outputs.

All experiments are based on Andrej Karpathy’s nanoGPT framework and trained on CPU.

Experiments Overview

Three datasets were used:

Basic (English) – Roman → English

Spanish – Roman → Spanish

Advanced (Multilingual & Multiformat) – Roman → English, Spanish, Arabic numerals, Devanagari numerals, and Nepali words

Each experiment was trained for 200 iterations (pipeline verification) and 2000 iterations (full training).

Data Format
1. Basic (English)
I one
II two
III three
IV four
...
XX twenty

2. Spanish
I uno
II dos
III tres
IV cuatro
...
XX veinte

3. Advanced (Multilingual)

Each line contains multiple representations of the same number:

I one uno 1 १ एक
II two dos 2 २ दुई
III three tres 3 ३ तीन
IV four cuatro 4 ४ चार
...
XX twenty veinte 20 २० बीस


This dataset combines:

Roman numerals

English words

Spanish words

Arabic numerals

Devanagari numerals

Nepali number words

Directory Structure
translate-roman-numbers/
├── README.md
├── config/
│   ├── basic.py
│   ├── spanish.py
│   └── advanced.py
├── data/
│   ├── basic/
│   │   ├── prepare.py
│   │   ├── train.bin
│   │   ├── val.bin
│   │   └── meta.pkl
│   ├── spanish/
│   └── advanced/
└── out/

Setup

Create and activate a Python virtual environment:

python -m venv venv
source venv/bin/activate


Install dependencies:

pip install torch numpy tqdm wandb


Prepare a dataset (example: basic):

cd translate-roman-numbers
python data/basic/prepare.py


Repeat for spanish and advanced as needed.

Training

From the translate-roman-numbers directory:

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
python -u ../../comp560-nanoGPT/train.py config/basic.py


Replace basic.py with spanish.py or advanced.py for other experiments.

Sampling
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
python -u ../../comp560-nanoGPT/sample.py config/basic.py \
--num_samples=1 --max_new_tokens=300 --seed=1337

Experiment Logs
🔹 Basic (English)
Run 1: max_iters = 200

Purpose: Verify training and sampling pipeline

Results:

Model learned line structure and repetition

Many translations incomplete or truncated

Limited semantic accuracy

Run 2: max_iters = 2000

Purpose: Full training

Results:

Consistently correct Roman → English translations

Strong pattern repetition

Minor truncation near sample boundaries

🔹 Spanish
Run 1: max_iters = 200

Purpose: Verify multilingual learning feasibility

Results:

Correct structure learned

Some spelling errors and partial words

Spanish vocabulary partially captured

Run 2: max_iters = 2000

Results:

Accurate Roman → Spanish translations

Accents and longer words learned surprisingly well (e.g., dieciséis)

Comparable performance to English despite higher complexity

🔹 Advanced (Multilingual)
Run 1: max_iters = 200

Purpose: Stress-test model capacity

Results:

Learned output structure (token ordering)

Many incomplete or mixed-language outputs

Expected underfitting due to task complexity

Run 2: max_iters = 2000

Results:

Significant improvement across all representations

Correct alignment between Roman numerals and multiple formats

Some minor inconsistencies, but strong overall learning

Demonstrates the model’s ability to handle multilingual and numeric mappings simultaneously

Weights & Biases (WandB)

Training loss curves for all experiments were logged using WandB.

Observations:

Loss decreased smoothly for basic and Spanish

Advanced experiment showed slower convergence and higher final loss

Increasing iterations consistently improved results across all datasets

Screenshots and outputs are included in the accompanying PDF.

Conclusion
What Worked Well

Clean, repetitive datasets enabled rapid learning

Increasing training iterations from 200 → 2000 dramatically improved accuracy

Character-level modeling successfully handled multiple languages and scripts

CPU training was sufficient for all experiments

Key Takeaways

Model complexity must match task complexity

Advanced multilingual translation is achievable even with a small GPT

Structured symbolic data is well-suited for character-level transformers