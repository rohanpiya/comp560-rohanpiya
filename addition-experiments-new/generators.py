"""
generators.py — synthetic addition data generators
Supports 1, 2, 3, and 4-digit addition in plain, CoT, and scratch formats.
Each function returns a single training example string ending with \n.

Plain format:   47+26=73\n
CoT format:     47+26=6+7=13;4+2+1=7;73\n
Scratch format: 47+26=________________73\n
                       ^^^^^^^^^^^^^^^^
                       K underscore tokens the model fills freely
                       K=16 for 2-digit, 25 for 3-digit, 34 for 4-digit
                       (calibrated to match max CoT scratch-region length + small buffer)
"""

import random


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _has_carry(a: int, b: int) -> bool:
    """Return True if adding a+b produces any carry at any digit position."""
    carry = 0
    while a > 0 or b > 0:
        digit_sum = (a % 10) + (b % 10) + carry
        carry = digit_sum // 10
        a //= 10
        b //= 10
    return False  # carry from the loop body was already checked inside


def has_carry(a: int, b: int) -> bool:
    """
    Correct carry detection: returns True if any column in a+b produces a carry.
    Works for any number of digits.
    """
    carry = 0
    while a > 0 or b > 0:
        digit_sum = (a % 10) + (b % 10) + carry
        if digit_sum >= 10:
            return True
        carry = digit_sum // 10
        a //= 10
        b //= 10
    return False


# ─────────────────────────────────────────────
#  Plain (no-CoT) generators
# ─────────────────────────────────────────────

def generate_plain(a: int, b: int) -> str:
    return f"{a}+{b}={a+b}\n"


def generate_1digit_plain_no_carry() -> str:
    while True:
        a, b = random.randint(0, 9), random.randint(0, 9)
        if not has_carry(a, b):
            return generate_plain(a, b)


def generate_1digit_plain_carry() -> str:
    while True:
        a, b = random.randint(0, 9), random.randint(0, 9)
        if has_carry(a, b):
            return generate_plain(a, b)


def generate_2digit_plain_no_carry() -> str:
    while True:
        a, b = random.randint(10, 99), random.randint(10, 99)
        if not has_carry(a, b):
            return generate_plain(a, b)


def generate_2digit_plain_carry() -> str:
    while True:
        a, b = random.randint(10, 99), random.randint(10, 99)
        if has_carry(a, b):
            return generate_plain(a, b)


def generate_3digit_plain_no_carry() -> str:
    while True:
        a, b = random.randint(100, 999), random.randint(100, 999)
        if not has_carry(a, b):
            return generate_plain(a, b)


def generate_3digit_plain_carry() -> str:
    while True:
        a, b = random.randint(100, 999), random.randint(100, 999)
        if has_carry(a, b):
            return generate_plain(a, b)


# ─────────────────────────────────────────────
#  CoT generators
# ─────────────────────────────────────────────

def _cot_2digit(a: int, b: int) -> str:
    """Build a 2-digit CoT string for given a, b (no range check)."""
    a1, a10 = a % 10, a // 10
    b1, b10 = b % 10, b // 10

    ones_sum = a1 + b1
    carry1   = ones_sum // 10

    tens_sum = a10 + b10 + carry1

    return (
        f"{a}+{b}="
        f"{a1}+{b1}={ones_sum};"
        f"{a10}+{b10}+{carry1}={tens_sum};"
        f"{a+b}\n"
    )


def _cot_3digit(a: int, b: int) -> str:
    """Build a 3-digit CoT string for given a, b (no range check)."""
    a1,  a10,  a100 = a % 10, (a // 10) % 10, a // 100
    b1,  b10,  b100 = b % 10, (b // 10) % 10, b // 100

    ones_sum     = a1  + b1
    carry1       = ones_sum // 10

    tens_sum     = a10 + b10 + carry1
    carry2       = tens_sum // 10

    hundreds_sum = a100 + b100 + carry2

    return (
        f"{a}+{b}="
        f"{a1}+{b1}={ones_sum};"
        f"{a10}+{b10}+{carry1}={tens_sum};"
        f"{a100}+{b100}+{carry2}={hundreds_sum};"
        f"{a+b}\n"
    )


def generate_2digit_cot_no_carry() -> str:
    while True:
        a, b = random.randint(10, 99), random.randint(10, 99)
        if not has_carry(a, b):
            return _cot_2digit(a, b)


def generate_2digit_cot_carry() -> str:
    while True:
        a, b = random.randint(10, 99), random.randint(10, 99)
        if has_carry(a, b):
            return _cot_2digit(a, b)


def generate_3digit_cot_no_carry() -> str:
    while True:
        a, b = random.randint(100, 999), random.randint(100, 999)
        if not has_carry(a, b):
            return _cot_3digit(a, b)


def generate_3digit_cot_carry() -> str:
    while True:
        a, b = random.randint(100, 999), random.randint(100, 999)
        if has_carry(a, b):
            return _cot_3digit(a, b)


# ─────────────────────────────────────────────
#  4-digit generators
# ─────────────────────────────────────────────

def generate_4digit_plain_no_carry() -> str:
    while True:
        a, b = random.randint(1000, 9999), random.randint(1000, 9999)
        if not has_carry(a, b):
            return f"{a}+{b}={a+b}\n"


def generate_4digit_plain_carry() -> str:
    while True:
        a, b = random.randint(1000, 9999), random.randint(1000, 9999)
        if has_carry(a, b):
            return f"{a}+{b}={a+b}\n"


def _cot_4digit(a: int, b: int) -> str:
    """Build a 4-digit CoT string for given a, b."""
    a1,    a10,   a100,  a1000  = a % 10, (a // 10) % 10, (a // 100) % 10, a // 1000
    b1,    b10,   b100,  b1000  = b % 10, (b // 10) % 10, (b // 100) % 10, b // 1000

    ones_sum      = a1    + b1
    carry1        = ones_sum // 10

    tens_sum      = a10   + b10   + carry1
    carry2        = tens_sum // 10

    hundreds_sum  = a100  + b100  + carry2
    carry3        = hundreds_sum // 10

    thousands_sum = a1000 + b1000 + carry3

    return (
        f"{a}+{b}="
        f"{a1}+{b1}={ones_sum};"
        f"{a10}+{b10}+{carry1}={tens_sum};"
        f"{a100}+{b100}+{carry2}={hundreds_sum};"
        f"{a1000}+{b1000}+{carry3}={thousands_sum};"
        f"{a+b}\n"
    )


def generate_4digit_cot_no_carry() -> str:
    while True:
        a, b = random.randint(1000, 9999), random.randint(1000, 9999)
        if not has_carry(a, b):
            return _cot_4digit(a, b)


def generate_4digit_cot_carry() -> str:
    while True:
        a, b = random.randint(1000, 9999), random.randint(1000, 9999)
        if has_carry(a, b):
            return _cot_4digit(a, b)

# ─────────────────────────────────────────────
#  Scratch space generators
#
#  Format:  47+26=________________73\n
#           K underscores between '=' and the answer.
#           K is calibrated to match the max CoT scratch-region length + buffer:
#             2-digit: K=16  (CoT max=15)
#             3-digit: K=25  (CoT max=24)
#             4-digit: K=34  (CoT max=33)
#
#  LOSS MASKING — the scratch positions are NOT supervised.
#  Each generator returns a (text, mask) tuple where mask is a string of the
#  same length as text: '1' = compute loss on this token, '0' = ignore it.
#  The scratch '_' positions are '0'; everything else (prompt + answer + \n) is '1'.
#
#  prepare_utils.build_dataset_masked() uses these masks to write a parallel
#  train_mask.bin / val_mask.bin alongside train.bin / val.bin.
#  train.py reads the mask and sets y=-1 at ignored positions before the
#  forward pass — nanoGPT already uses ignore_index=-1 in F.cross_entropy,
#  so no change to model.py is required.
# ─────────────────────────────────────────────

SCRATCH_K = {2: 16, 3: 25, 4: 34}   # digit_count -> number of scratch tokens


def _scratch_with_mask(a: int, b: int, k: int) -> tuple[str, str]:
    """
    Build a scratch-format example and its loss mask.

    Returns:
        text  — e.g. '47+26=________________73\n'
        mask  — same length, '1' where loss is computed, '0' for scratch positions

    Example (k=4 for clarity):
        text = '47+26=____73\n'
        mask = '111111000011 1'   <- prompt=1s, scratch=0s, answer+newline=1s
    """
    prompt = f"{a}+{b}="
    scratch = "_" * k
    answer = f"{a + b}\n"
    text = prompt + scratch + answer
    mask = "1" * len(prompt) + "0" * k + "1" * len(answer)
    return text, mask


# ── 2-digit scratch ──────────────────────────────────────────────────────────

def generate_2digit_scratch_no_carry() -> tuple[str, str]:
    while True:
        a, b = random.randint(10, 99), random.randint(10, 99)
        if not has_carry(a, b):
            return _scratch_with_mask(a, b, SCRATCH_K[2])


def generate_2digit_scratch_carry() -> tuple[str, str]:
    while True:
        a, b = random.randint(10, 99), random.randint(10, 99)
        if has_carry(a, b):
            return _scratch_with_mask(a, b, SCRATCH_K[2])


# ── 3-digit scratch ──────────────────────────────────────────────────────────

def generate_3digit_scratch_no_carry() -> tuple[str, str]:
    while True:
        a, b = random.randint(100, 999), random.randint(100, 999)
        if not has_carry(a, b):
            return _scratch_with_mask(a, b, SCRATCH_K[3])


def generate_3digit_scratch_carry() -> tuple[str, str]:
    while True:
        a, b = random.randint(100, 999), random.randint(100, 999)
        if has_carry(a, b):
            return _scratch_with_mask(a, b, SCRATCH_K[3])


# ── 4-digit scratch ──────────────────────────────────────────────────────────

def generate_4digit_scratch_no_carry() -> tuple[str, str]:
    while True:
        a, b = random.randint(1000, 9999), random.randint(1000, 9999)
        if not has_carry(a, b):
            return _scratch_with_mask(a, b, SCRATCH_K[4])


def generate_4digit_scratch_carry() -> tuple[str, str]:
    while True:
        a, b = random.randint(1000, 9999), random.randint(1000, 9999)
        if has_carry(a, b):
            return _scratch_with_mask(a, b, SCRATCH_K[4])
