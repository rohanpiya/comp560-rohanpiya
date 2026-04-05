"""
prepare_utils.py — shared dataset preparation logic
Called by each data/*/prepare.py script.
"""

import os
import random
import pickle
import numpy as np


def build_dataset(
    no_carry_fn,
    carry_fn,
    num_examples: int,
    carry_ratio: float,
    output_dir: str,
    train_split: float = 0.9,
    seed: int = 42,
):
    """
    Generate a balanced dataset and write train.bin, val.bin, meta.pkl.

    Args:
        no_carry_fn:  callable() -> str   (one training example)
        carry_fn:     callable() -> str   (one training example)
        num_examples: total number of examples to generate
        carry_ratio:  fraction that should be carry examples (e.g. 0.5)
        output_dir:   directory to write train.bin / val.bin / meta.pkl
        train_split:  fraction used for training (rest = validation)
        seed:         random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)

    num_carry    = int(num_examples * carry_ratio)
    num_no_carry = num_examples - num_carry

    print(f"Generating {num_no_carry:,} no-carry + {num_carry:,} carry examples...")

    examples = []
    for _ in range(num_no_carry):
        examples.append(no_carry_fn())
    for _ in range(num_carry):
        examples.append(carry_fn())

    random.shuffle(examples)

    # ── vocabulary ──────────────────────────────────────────────
    data_str = "".join(examples)
    chars     = sorted(set(data_str))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    print(f"Vocab ({vocab_size}): {'  '.join(repr(c) for c in chars)}")
    print(f"Dataset length: {len(data_str):,} characters")

    # ── train / val split ───────────────────────────────────────
    cutoff     = int(len(examples) * train_split)
    train_str  = "".join(examples[:cutoff])
    val_str    = "".join(examples[cutoff:])

    train_ids = np.array([stoi[c] for c in train_str], dtype=np.uint16)
    val_ids   = np.array([stoi[c] for c in val_str],   dtype=np.uint16)

    print(f"Train tokens: {len(train_ids):,}  |  Val tokens: {len(val_ids):,}")

    # ── write files ─────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    train_ids.tofile(os.path.join(output_dir, "train.bin"))
    val_ids.tofile(  os.path.join(output_dir, "val.bin"))

    meta = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos}
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    # ── sanity-check sample ──────────────────────────────────────
    print("\nFirst 10 examples:")
    for ex in examples[:10]:
        print(" ", repr(ex.strip()))

    print(f"\nDone. Files written to: {output_dir}")
