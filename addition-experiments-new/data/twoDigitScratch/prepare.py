"""
data/twoDigitScratch/prepare.py
Generates the 2-digit scratch-space dataset.

Format: 47+26=;;;;;;;;;;;;;;;;73\n
               ^^^^^^^^^^^^^^^^
               K=16 semicolons (the scratch buffer).

Why semicolons?
  The CoT vocab is: \n + 0-9 = ;  (14 chars total).
  By using ';' as the scratch token, the scratch dataset has the
  exact same 14-char vocab as the CoT dataset.  This means the
  CoT checkpoint loads cleanly with no embedding size mismatch.

Run from addition-experiments-new/:
  python data/twoDigitScratch/prepare.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from generators    import generate_2digit_scratch_no_carry, generate_2digit_scratch_carry
from prepare_utils import build_dataset

build_dataset(
    no_carry_fn  = generate_2digit_scratch_no_carry,
    carry_fn     = generate_2digit_scratch_carry,
    num_examples = 100_000,
    carry_ratio  = 0.5,
    output_dir   = os.path.dirname(__file__),
)
