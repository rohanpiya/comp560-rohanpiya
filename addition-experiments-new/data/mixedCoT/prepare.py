"""
data/mixedCoT/prepare.py
Generates the mixed-digit CoT dataset.

A single model is trained on 2, 3, and 4-digit addition problems together.
The CoT step count varies naturally with digit count:
  47+26=7+6=13;4+2+1=7;73\n              (2 steps)
  473+261=3+1=4;7+6+0=13;4+2+1=7;734\n  (3 steps)
  4731+2614=1+4=5;...;7345\n             (4 steps)

This forces the model to learn "count the columns, write one step per column"
rather than memorizing a fixed step count — which is what individual per-digit
CoT models do (the "rigid template" failure mode).

Dataset composition: 100,000 examples total, evenly split across digit counts
and carry/no-carry within each digit count.
  ~16,667 examples per digit count
  ~8,333 carry + ~8,333 no-carry per digit count

Run from addition-experiments-new/:
  python data/mixedCoT/prepare.py
"""

import sys, os, random, pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from generators import (
    generate_2digit_cot_no_carry, generate_2digit_cot_carry,
    generate_3digit_cot_no_carry, generate_3digit_cot_carry,
    generate_4digit_cot_no_carry, generate_4digit_cot_carry,
)

# ── config ────────────────────────────────────────────────────────────────────
NUM_EXAMPLES  = 100_000
PER_DIGIT     = NUM_EXAMPLES // 3          # ~33,333 per digit count
CARRY_RATIO   = 0.5
TRAIN_SPLIT   = 0.9
SEED          = 42
OUTPUT_DIR    = os.path.dirname(__file__)

random.seed(SEED)
np.random.seed(SEED)

# ── generate ──────────────────────────────────────────────────────────────────
print(f"Generating {NUM_EXAMPLES:,} mixed-digit CoT examples "
      f"(~{PER_DIGIT:,} per digit count, {CARRY_RATIO:.0%} carry)...")

examples = []

for digit, no_carry_fn, carry_fn in [
    (2, generate_2digit_cot_no_carry, generate_2digit_cot_carry),
    (3, generate_3digit_cot_no_carry, generate_3digit_cot_carry),
    (4, generate_4digit_cot_no_carry, generate_4digit_cot_carry),
]:
    n_carry    = int(PER_DIGIT * CARRY_RATIO)
    n_no_carry = PER_DIGIT - n_carry
    for _ in range(n_no_carry):
        examples.append(no_carry_fn())
    for _ in range(n_carry):
        examples.append(carry_fn())
    print(f"  {digit}-digit: {n_no_carry:,} no-carry + {n_carry:,} carry")

random.shuffle(examples)

# ── vocabulary ────────────────────────────────────────────────────────────────
data_str   = "".join(examples)
chars      = sorted(set(data_str))
vocab_size = len(chars)
stoi       = {ch: i for i, ch in enumerate(chars)}
itos       = {i: ch for i, ch in enumerate(chars)}

print(f"\nVocab ({vocab_size}): {'  '.join(repr(c) for c in chars)}")
print(f"Dataset length: {len(data_str):,} characters")

# Sanity check — must match individual CoT vocab exactly
expected_chars = set('\n+0123456789;=')
assert set(chars) == expected_chars, \
    f"Vocab mismatch! Got {set(chars)}, expected {expected_chars}"
print("Vocab check: matches individual CoT datasets ✓")

# ── train / val split ─────────────────────────────────────────────────────────
cutoff    = int(len(examples) * TRAIN_SPLIT)
train_str = "".join(examples[:cutoff])
val_str   = "".join(examples[cutoff:])

train_ids = np.array([stoi[c] for c in train_str], dtype=np.uint16)
val_ids   = np.array([stoi[c] for c in val_str],   dtype=np.uint16)

print(f"Train tokens: {len(train_ids):,}  |  Val tokens: {len(val_ids):,}")

# ── write ─────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
train_ids.tofile(os.path.join(OUTPUT_DIR, "train.bin"))
val_ids.tofile(  os.path.join(OUTPUT_DIR, "val.bin"))

meta = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos}
with open(os.path.join(OUTPUT_DIR, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

# ── sample ────────────────────────────────────────────────────────────────────
print("\nFirst 12 examples (showing mixed lengths):")
for ex in examples[:12]:
    print(" ", repr(ex.strip()))

print(f"\nDone. Files written to: {OUTPUT_DIR}")
