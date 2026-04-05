"""
data/twoDigitCoT/prepare.py
Generates the 2-digit Chain-of-Thought dataset.

Format:  47+26=6+7=13;4+2+1=7;73\n
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from generators    import generate_2digit_cot_no_carry, generate_2digit_cot_carry
from prepare_utils import build_dataset

build_dataset(
    no_carry_fn  = generate_2digit_cot_no_carry,
    carry_fn     = generate_2digit_cot_carry,
    num_examples = 100_000,
    carry_ratio  = 0.5,
    output_dir   = os.path.dirname(__file__),
)
