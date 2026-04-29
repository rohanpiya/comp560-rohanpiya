"""
data/twoDigitScratch/prepare.py
Generates the 2-digit scratch-space dataset with loss masking.

Format:  47+26=________________73\n
                ^^^^^^^^^^^^^^^^
                K=16 scratch tokens — NOT supervised during training.
                Only the prompt (47+26=) and answer (73\n) are supervised.

Writes:
  train.bin / val.bin           — token IDs (uint16)
  train_mask.bin / val_mask.bin — loss mask (uint8): 1=supervise, 0=ignore
  meta.pkl                      — vocab + has_mask=True
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from generators    import generate_2digit_scratch_no_carry, generate_2digit_scratch_carry
from prepare_utils import build_dataset_masked

build_dataset_masked(
    no_carry_fn  = generate_2digit_scratch_no_carry,
    carry_fn     = generate_2digit_scratch_carry,
    num_examples = 100_000,
    carry_ratio  = 0.5,
    output_dir   = os.path.dirname(__file__),
)
