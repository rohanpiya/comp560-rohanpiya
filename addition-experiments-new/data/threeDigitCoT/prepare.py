"""
data/threeDigitCoT/prepare.py
Generates the 3-digit Chain-of-Thought dataset.

Format:  473+261=3+1=4;7+6+0=13;4+2+1=7;734\n
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from generators    import generate_3digit_cot_no_carry, generate_3digit_cot_carry
from prepare_utils import build_dataset

build_dataset(
    no_carry_fn  = generate_3digit_cot_no_carry,
    carry_fn     = generate_3digit_cot_carry,
    num_examples = 100_000,
    carry_ratio  = 0.5,
    output_dir   = os.path.dirname(__file__),
)
