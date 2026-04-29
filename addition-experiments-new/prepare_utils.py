"""
prepare_utils.py — shared dataset preparation logic
Called by each data/*/prepare.py script.

Two builders:
  build_dataset()         — plain / CoT datasets (no masking)
  build_dataset_masked()  — scratch datasets (writes mask files for loss masking)
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


def build_dataset_masked(
    no_carry_fn,
    carry_fn,
    num_examples: int,
    carry_ratio: float,
    output_dir: str,
    train_split: float = 0.9,
    seed: int = 42,
):
    """
    Generate a masked scratch dataset and write:
      train.bin, val.bin         — token IDs (uint16), same as build_dataset
      train_mask.bin, val_mask.bin — uint8 arrays, 1=supervise, 0=ignore
      meta.pkl                   — vocab + has_mask=True flag

    Each generator function must return (text: str, mask: str) where mask is
    a string of '0' and '1' characters of the same length as text.

    The mask files are read by train.py's get_batch() to zero out y at
    positions where mask=0, exploiting nanoGPT's ignore_index=-1 in
    F.cross_entropy — no change to model.py required.
    """
    random.seed(seed)
    np.random.seed(seed)

    num_carry    = int(num_examples * carry_ratio)
    num_no_carry = num_examples - num_carry

    print(f"Generating {num_no_carry:,} no-carry + {num_carry:,} carry examples (with loss mask)...")

    examples = []   # list of (text, mask) tuples
    for _ in range(num_no_carry):
        examples.append(no_carry_fn())
    for _ in range(num_carry):
        examples.append(carry_fn())

    random.shuffle(examples)

    texts = [t for t, _ in examples]
    masks = [m for _, m in examples]

    # ── sanity check ────────────────────────────────────────────
    for i, (t, m) in enumerate(zip(texts, masks)):
        assert len(t) == len(m), (
            f"Example {i}: text length {len(t)} != mask length {len(m)}\n"
            f"  text: {repr(t)}\n  mask: {repr(m)}"
        )

    # ── vocabulary (built from text only) ───────────────────────
    data_str   = "".join(texts)
    chars      = sorted(set(data_str))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    print(f"Vocab ({vocab_size}): {'  '.join(repr(c) for c in chars)}")
    print(f"Dataset length: {len(data_str):,} characters")

    # ── mask stats ───────────────────────────────────────────────
    mask_str      = "".join(masks)
    total_tokens  = len(mask_str)
    masked_tokens = mask_str.count("0")
    print(f"Loss mask: {masked_tokens:,} / {total_tokens:,} tokens ignored "
          f"({masked_tokens/total_tokens*100:.1f}% scratch, "
          f"{(total_tokens-masked_tokens)/total_tokens*100:.1f}% supervised)")

    # ── train / val split ────────────────────────────────────────
    cutoff    = int(len(texts) * train_split)
    train_str = "".join(texts[:cutoff])
    val_str   = "".join(texts[cutoff:])
    train_msk = "".join(masks[:cutoff])
    val_msk   = "".join(masks[cutoff:])

    train_ids  = np.array([stoi[c] for c in train_str], dtype=np.uint16)
    val_ids    = np.array([stoi[c] for c in val_str],   dtype=np.uint16)
    train_mask = np.array([int(c)  for c in train_msk], dtype=np.uint8)
    val_mask   = np.array([int(c)  for c in val_msk],   dtype=np.uint8)

    print(f"Train tokens: {len(train_ids):,}  |  Val tokens: {len(val_ids):,}")

    # ── write files ──────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    train_ids.tofile( os.path.join(output_dir, "train.bin"))
    val_ids.tofile(   os.path.join(output_dir, "val.bin"))
    train_mask.tofile(os.path.join(output_dir, "train_mask.bin"))
    val_mask.tofile(  os.path.join(output_dir, "val_mask.bin"))

    meta = {
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": itos,
        "has_mask": True,        # signals to train.py that mask files exist
    }
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    # ── sanity-check sample ──────────────────────────────────────
    print("\nFirst 10 examples (text | mask):")
    for text, mask in zip(texts[:10], masks[:10]):
        print(f"  text: {repr(text.strip())}")
        print(f"  mask: {repr(mask.strip())}")
        print()

    print(f"Done. Files written to: {output_dir}")
