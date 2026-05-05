"""
evaluate_mixed.py — evaluation script for the mixed-digit CoT model

The mixed CoT model is trained on 2, 3, and 4-digit problems together.
This script evaluates it across all digit counts including 5-digit OOD.

The key question: does variable-length CoT training break the rigid template
and allow the model to generalize to unseen digit counts?

We compare against the per-digit CoT baselines from the main experiments:
  Per-digit CoT 2-digit: 99.90% overall
  Per-digit CoT 3-digit: 99.71% overall
  Per-digit CoT 4-digit: 99.66% overall
  Per-digit CoT OOD:     0% (hard boundary)

Usage
-----
# In-distribution evaluations (all three training digit counts)
python evaluate_mixed.py --eval_digits 2
python evaluate_mixed.py --eval_digits 3 --num_samples 10000
python evaluate_mixed.py --eval_digits 4 --num_samples 10000

# The key OOD test — 5-digit inputs never seen during training
python evaluate_mixed.py --eval_digits 5 --num_samples 10000

# 1-digit OOD (below training distribution)
python evaluate_mixed.py --eval_digits 1
"""

import os
import sys
import pickle
import argparse
import random
import re
import time
from collections import Counter

import torch

# ── locate nanoGPT ────────────────────────────────────────────────────────────
nanogpt_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../comp560-nanoGPT")
)
if not os.path.isdir(nanogpt_path):
    raise FileNotFoundError(
        f"Cannot find comp560-nanoGPT at: {nanogpt_path}\n"
        f"Expected: comp560-nanoGPT/ as sibling of comp560-rohanpiya/"
    )
sys.path.insert(0, nanogpt_path)
from model import GPTConfig, GPT

DATASET = "mixedCoT"

# Max CoT step tokens per digit count (for generation budget)
# Format: prompt(~10) + steps(~8*n_digits) + answer(~6) + newline
# We use a generous budget to handle variable-length output
MAX_NEW_BY_DIGITS = {
    1: 20,
    2: 30,
    3: 40,
    4: 50,
    5: 65,    # OOD — generous budget since we don't know exact length
    6: 80,    # extra headroom just in case
}


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_digits",  type=int, required=True,
                   help="Digit count to evaluate on (1-5).")
    p.add_argument("--num_samples",  type=int, default=2000,
                   help="Random samples for 3+ digit eval. Default 2000.")
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Model and tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def load_model(device):
    ckpt_path = os.path.join("out", DATASET, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}\n"
            f"Run training first with config/config_train_mixedCoT.py"
        )
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    state = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in ckpt["model"].items()
    }
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def load_tokenizer():
    meta_path = os.path.join("data", DATASET, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No meta.pkl at {meta_path}\n"
            f"Run: python data/mixedCoT/prepare.py"
        )
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join(itos[i] for i in l)
    return encode, decode


# ─────────────────────────────────────────────────────────────────────────────
#  Answer extraction from CoT output
# ─────────────────────────────────────────────────────────────────────────────

def extract_answer(raw: str, eval_digits: int) -> str:
    """
    Extract the final answer from a CoT generation.

    CoT format ends with: ...;{answer}\n
    The answer is the last semicolon-delimited field before the newline.

    Falls back to: find the last digit sequence of plausible length.
    """
    first_line = raw.split("\n")[0]

    # Primary: last field after the last semicolon
    if ";" in first_line:
        candidate = first_line.rsplit(";", 1)[-1].strip()
        if candidate.isdigit():
            return candidate

    # Fallback: find digit sequences of expected answer length
    target_lens = {eval_digits, eval_digits + 1}  # +1 for carry overflow
    nums = re.findall(r"\d+", first_line)
    for num in reversed(nums):   # prefer later (answer) over earlier (steps)
        if len(num) in target_lens:
            return num

    # Last resort: any digit sequence
    if nums:
        return nums[-1]
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Carry detection
# ─────────────────────────────────────────────────────────────────────────────

def has_carry(a: int, b: int) -> bool:
    carry = 0
    while a > 0 or b > 0:
        s = (a % 10) + (b % 10) + carry
        if s >= 10:
            return True
        carry = s // 10
        a //= 10
        b //= 10
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Step counting analysis
# ─────────────────────────────────────────────────────────────────────────────

def count_steps(raw: str) -> int:
    """Count the number of ';' delimited steps in the generated output."""
    first_line = raw.split("\n")[0]
    # Remove the prompt part (everything up to and including '=')
    if "=" in first_line:
        after_eq = first_line.split("=", 1)[1]
        return after_eq.count(";")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
#  Eval pair generation
# ─────────────────────────────────────────────────────────────────────────────

def get_pairs(eval_digits: int, num_samples: int, seed: int):
    low  = 10 ** (eval_digits - 1) if eval_digits > 1 else 0
    high = 10 ** eval_digits
    if eval_digits <= 2:
        pairs = [(a, b) for a in range(low, high) for b in range(low, high)]
        print(f"Exhaustive eval: {len(pairs):,} pairs")
    else:
        random.seed(seed)
        pairs = [(random.randint(low, high-1), random.randint(low, high-1))
                 for _ in range(num_samples)]
        print(f"Sampled eval: {len(pairs):,} pairs")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  Batched generation
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(prompts, model, encode, decode, device, max_new):
    encoded = [encode(p) for p in prompts]
    maxlen  = max(len(e) for e in encoded)
    padded  = [[0] * (maxlen - len(e)) + e for e in encoded]
    x       = torch.tensor(padded, dtype=torch.long, device=device)
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_new, temperature=1.0, top_k=1)
    results = []
    for i, prompt in enumerate(prompts):
        full = decode(y[i].tolist())
        idx  = full.find(prompt)
        results.append(full[idx + len(prompt):] if idx != -1 else full[len(prompt):])
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = "cpu"
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    max_new = MAX_NEW_BY_DIGITS.get(args.eval_digits, 80)
    in_dist = args.eval_digits in {2, 3, 4}

    print(f"\n{'='*64}")
    print(f"  Mixed-digit CoT model evaluation")
    print(f"  Model:         {DATASET}  (trained on 2+3+4-digit mixed)")
    print(f"  Eval digits:   {args.eval_digits}")
    print(f"  Max new tok:   {max_new}")
    dist_label = "IN-DISTRIBUTION" if in_dist else "OUT-OF-DISTRIBUTION"
    print(f"  Distribution:  {dist_label}")
    print(f"{'='*64}\n")

    model          = load_model(device)
    encode, decode = load_tokenizer()
    pairs          = get_pairs(args.eval_digits, args.num_samples, args.seed)

    # ── counters ──────────────────────────────────────────────────
    correct_carry    = correct_no_carry = 0
    total_carry      = total_no_carry   = 0
    step_counts      = Counter()
    len_counter      = Counter()
    n_debug          = 0
    t0               = time.time()

    for start in range(0, len(pairs), args.batch_size):
        batch   = pairs[start : start + args.batch_size]
        prompts = [f"{a}+{b}=" for a, b in batch]
        outputs = run_batch(prompts, model, encode, decode, device, max_new)

        for (a, b), raw in zip(batch, outputs):
            gt   = str(a + b)
            pred = extract_answer(raw, args.eval_digits)
            ok   = (pred == gt)
            steps = count_steps(raw)
            step_counts[steps] += 1
            len_counter[len(pred)] += 1

            if n_debug < 5:
                print(f"[DEBUG {n_debug+1}]")
                print(f"  Prompt : '{a}+{b}='")
                print(f"  Raw    : {raw[:70]!r}{'...' if len(raw)>70 else ''}")
                print(f"  Steps  : {steps}  (expected {args.eval_digits} for {args.eval_digits}-digit)")
                print(f"  Pred   : {pred!r}  |  GT: {gt!r}  |  Correct: {ok}")
                print()
                n_debug += 1

            if has_carry(a, b):
                total_carry += 1
                if ok: correct_carry += 1
            else:
                total_no_carry += 1
                if ok: correct_no_carry += 1

        done = min(start + args.batch_size, len(pairs))
        if done % 500 == 0 or done == len(pairs):
            print(f"  {done:>6}/{len(pairs)} evaluated  [{time.time()-t0:.1f}s]")

    elapsed = time.time() - t0
    total   = total_carry + total_no_carry
    correct = correct_carry + correct_no_carry

    def pct(n, d):
        return f"{n/d*100:.2f}%" if d else "N/A"

    # ── accuracy ───────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  ACCURACY — mixedCoT  (eval on {args.eval_digits}-digit inputs)")
    print(f"{'='*64}")
    print(f"  Overall  : {pct(correct, total):>8}  ({correct}/{total})")
    print(f"  No-carry : {pct(correct_no_carry, total_no_carry):>8}  ({correct_no_carry}/{total_no_carry})")
    print(f"  Carry    : {pct(correct_carry, total_carry):>8}  ({correct_carry}/{total_carry})")
    print(f"  Time     : {elapsed:.1f}s  ({elapsed/total*1000:.1f} ms/sample)")
    print(f"  Output lengths: {dict(sorted(len_counter.items()))}")

    # ── step count analysis ────────────────────────────────────────
    # This is the key diagnostic for the rigid template question.
    # If the model learned column-counting, it should produce exactly
    # eval_digits steps for eval_digits-digit inputs — even for OOD 5-digit.
    # If it's still rigid, OOD inputs will get the wrong number of steps.
    print(f"\n  --- Step Count Analysis ---")
    print(f"  (Expected: {args.eval_digits} steps for {args.eval_digits}-digit inputs)")
    print(f"  (This is the key diagnostic: does the model count columns")
    print(f"   or memorize a fixed step count?)")
    print()
    total_gens = sum(step_counts.values())
    for n_steps, count in sorted(step_counts.items()):
        marker = " ← correct" if n_steps == args.eval_digits else ""
        print(f"    {n_steps} steps: {count:>6} ({count/total_gens*100:.1f}%){marker}")

    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
