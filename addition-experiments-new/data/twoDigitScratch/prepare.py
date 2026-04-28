"""
data/twoDigitScratch/prepare.py
Generates the 2-digit scratch-space dataset.

Format:  47+26=________________73\n
                ^^^^^^^^^^^^^^^^
                K=16 underscore tokens the model fills freely.

The model is trained on all positions (including underscores) via standard
cross-entropy. No prescribed structure is given for the scratch region.
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
