"""
generators.py — synthetic addition data generators
Supports 1, 2, and 3-digit addition in both plain and CoT formats.
Each function returns a single training example string ending with \n.

Plain format:   47+26=73\n
CoT format:     47+26=6+7=13;4+2+1=7;73\n
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
