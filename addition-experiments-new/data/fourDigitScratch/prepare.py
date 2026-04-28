"""
data/fourDigitScratch/prepare.py
Generates the 4-digit scratch-space dataset.

Format:  4731+2614=__________________________________7345\n
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    K=34 underscore tokens the model fills freely.

The model is trained on all positions (including underscores) via standard
cross-entropy. No prescribed structure is given for the scratch region.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from generators    import generate_4digit_scratch_no_carry, generate_4digit_scratch_carry
from prepare_utils import build_dataset

build_dataset(
    no_carry_fn  = generate_4digit_scratch_no_carry,
    carry_fn     = generate_4digit_scratch_carry,
    num_examples = 100_000,
    carry_ratio  = 0.5,
    output_dir   = os.path.dirname(__file__),
)
