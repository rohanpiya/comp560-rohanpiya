"""
data/fourDigitCoT/prepare.py
Generates the 4-digit Chain-of-Thought dataset.

Format:  4731+2614=1+4=5;3+1+0=4;7+6+0=13;4+2+1=7;7345\n
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from generators    import generate_4digit_cot_no_carry, generate_4digit_cot_carry
from prepare_utils import build_dataset

build_dataset(
    no_carry_fn  = generate_4digit_cot_no_carry,
    carry_fn     = generate_4digit_cot_carry,
    num_examples = 100_000,
    carry_ratio  = 0.5,
    output_dir   = os.path.dirname(__file__),
)
