"""
evaluate_scratch.py — evaluation script for scratch-space addition models

Scratch format:  47+26=________________73\n
                        ^^^^^^^^^^^^^^^^
                        K underscore tokens the model fills freely,
                        followed immediately by the answer.

Key differences from evaluate.py / evaluate_new.py:
  - Answer extraction skips K scratch tokens, then reads the digit sequence
  - Additional scratch analysis metrics:
      * scratch_digit_frac    — fraction of scratch positions filled with digits
      * scratch_diversity     — unique scratch strings / total samples (0=collapsed, 1=fully varied)
      * top_scratch_patterns  — most common scratch strings seen
  - These let us answer: did the model learn to use the scratch space, or ignore it?

Usage:
  # In-distribution eval (2-digit scratch model on 2-digit inputs)
  python evaluate_scratch.py --dataset twoDigitScratch --train_digits 2 --eval_digits 2

  # Out-of-distribution eval
  python evaluate_scratch.py --dataset twoDigitScratch --train_digits 2 --eval_digits 1
  python evaluate_scratch.py --dataset twoDigitScratch --train_digits 2 --eval_digits 3

  # Larger digit counts (use --num_samples)
  python evaluate_scratch.py --dataset threeDigitScratch --train_digits 3 --eval_digits 3 --num_samples 10000
  python evaluate_scratch.py --dataset fourDigitScratch  --train_digits 4 --eval_digits 4 --num_samples 10000
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

# ── nanoGPT path ──────────────────────────────────────────────────────────────
nanogpt_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../comp560-nanoGPT")
)
if not os.path.isdir(nanogpt_path):
    raise FileNotFoundError(
        f"Cannot find comp560-nanoGPT at: {nanogpt_path}\n"
        f"Expected structure:\n"
        f"  <parent>/\n"
        f"    comp560-nanoGPT/\n"
        f"    comp560-rohanpiya/addition-experiments-new/evaluate_scratch.py"
    )
sys.path.insert(0, nanogpt_path)
from model import GPTConfig, GPT


# ── scratch K values (must match generators.py) ───────────────────────────────
SCRATCH_K = {2: 16, 3: 25, 4: 34}


# ─────────────────────────────────────────────────────────────────────────────
#  Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a scratch-space addition model.")
    p.add_argument("--dataset",      type=str, required=True,
                   help="Dataset name, e.g. twoDigitScratch")
    p.add_argument("--train_digits", type=int, required=True,
                   help="Number of digits the model was trained on (2, 3, or 4)")
    p.add_argument("--eval_digits",  type=int, required=True,
                   help="Number of digits to evaluate on (can differ from train)")
    p.add_argument("--num_samples",  type=int, default=2000,
                   help="Samples for large search spaces (>= 3 digits). Default 2000.")
    p.add_argument("--batch_size",   type=int, default=32,
                   help="Generation batch size. Default 32.")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--top_patterns", type=int, default=10,
                   help="How many top scratch patterns to print. Default 10.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Model + tokenizer loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(dataset: str, device: str):
    ckpt_path = os.path.join("out", dataset, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    state = checkpoint["model"]
    state = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
             for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def load_tokenizer(dataset: str):
    meta_path = os.path.join("data", dataset, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No meta.pkl found at {meta_path}")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
    return encode, decode


# ─────────────────────────────────────────────────────────────────────────────
#  Token budget
# ─────────────────────────────────────────────────────────────────────────────

def max_new_tokens(train_digits: int, eval_digits: int) -> int:
    """
    Scratch model output = K scratch tokens + answer digits + newline.

    When eval_digits != train_digits the model may still try to produce K
    tokens based on what it learned, so we always budget at least K_train
    scratch positions.  We also add slack for the answer itself.

    Formula:  max(K_train, K_eval_if_known) + (eval_digits + 2)
    """
    k_train = SCRATCH_K.get(train_digits, train_digits * 9)
    # also budget for the answer: eval_digits+1 digits possible (carry) + newline
    return k_train + eval_digits + 2


# ─────────────────────────────────────────────────────────────────────────────
#  Answer extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_scratch(generated: str, train_digits: int, eval_digits: int) -> tuple[str, str]:
    """
    Extract (answer, scratch_content) from a scratch model's raw output.

    The model was trained to output K scratch tokens then the answer.
    K is determined by train_digits (the training distribution).

    Strategy:
      1. Take first line only (everything before first \n)
      2. The first K characters are the scratch region
      3. Everything after position K is the answer region — take the first
         contiguous digit sequence found there

    When eval_digits != train_digits the model may produce garbled output,
    but we still extract whatever we can so the failure mode is visible.

    Returns:
        answer  (str): extracted digit string, or "" if nothing found
        scratch (str): the K-character scratch region (may contain non-_ chars)
    """
    k = SCRATCH_K.get(train_digits, train_digits * 9)
    first_line = generated.split("\n")[0]

    scratch_region = first_line[:k]
    answer_region  = first_line[k:]

    # Primary: first digit sequence in the answer region
    nums = re.findall(r"\d+", answer_region)
    if nums:
        # prefer a number of the right length, otherwise take the first
        valid_lens = {eval_digits, eval_digits + 1}
        for num in nums:
            if len(num) in valid_lens:
                return num, scratch_region
        return nums[0], scratch_region

    # Fallback: scan entire first line for any number of the right length
    # (handles cases where the model didn't respect the K-token boundary)
    valid_lens = {eval_digits, eval_digits + 1}
    for num in re.findall(r"\d+", first_line):
        if len(num) in valid_lens:
            return num, scratch_region

    return "", scratch_region


# ─────────────────────────────────────────────────────────────────────────────
#  Scratch analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_scratch(scratch_strings: list[str], k: int) -> dict:
    """
    Compute scratch utilization and content metrics.

    scratch_strings: list of K-character scratch region strings from all samples
    k: expected scratch length (train_digits K value)

    Returns dict with:
      digit_frac       — fraction of all scratch characters that are digits
      underscore_frac  — fraction still filled with '_' (model didn't overwrite)
      diversity        — unique strings / total (0.0 = fully collapsed, 1.0 = all unique)
      top_patterns     — Counter of most common scratch strings
      mean_len         — mean length of scratch strings produced
    """
    if not scratch_strings:
        return {}

    total_chars = sum(len(s) for s in scratch_strings)
    digit_chars = sum(sum(c.isdigit() for c in s) for s in scratch_strings)
    under_chars = sum(s.count("_") for s in scratch_strings)

    counter    = Counter(scratch_strings)
    n          = len(scratch_strings)
    diversity  = len(counter) / n if n > 0 else 0.0
    mean_len   = total_chars / n if n > 0 else 0.0

    return {
        "digit_frac"      : digit_chars / total_chars if total_chars > 0 else 0.0,
        "underscore_frac" : under_chars / total_chars if total_chars > 0 else 0.0,
        "diversity"       : diversity,
        "top_patterns"    : counter,
        "mean_len"        : mean_len,
        "n_samples"       : n,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Carry detection
# ─────────────────────────────────────────────────────────────────────────────

def has_carry(a: int, b: int) -> bool:
    carry = 0
    while a > 0 or b > 0:
        digit_sum = (a % 10) + (b % 10) + carry
        if digit_sum >= 10:
            return True
        carry = digit_sum // 10
        a //= 10
        b //= 10
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt building
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(a: int, b: int) -> str:
    """Scratch prompt ends at '=' — the model generates the rest."""
    return f"{a}+{b}="


# ─────────────────────────────────────────────────────────────────────────────
#  Eval pairs
# ─────────────────────────────────────────────────────────────────────────────

def get_eval_pairs(eval_digits: int, num_samples: int, seed: int):
    """
    Exhaustive for eval_digits <= 2, random sample for eval_digits >= 3.
    """
    low  = 10 ** (eval_digits - 1) if eval_digits > 1 else 0
    high = 10 ** eval_digits

    if eval_digits <= 2:
        pairs = [(a, b) for a in range(low, high) for b in range(low, high)]
        print(f"Exhaustive eval: {len(pairs):,} pairs")
    else:
        random.seed(seed)
        pairs = [
            (random.randint(low, high - 1), random.randint(low, high - 1))
            for _ in range(num_samples)
        ]
        print(f"Sampled eval: {len(pairs):,} pairs")

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  Batched generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_batch(prompts, model, encode, decode, device, mnt):
    """Left-pad prompts, run one batched forward pass, return generated text."""
    encoded = [encode(p) for p in prompts]
    max_len = max(len(e) for e in encoded)
    pad_id  = 0
    padded  = [([pad_id] * (max_len - len(e))) + e for e in encoded]

    x = torch.tensor(padded, dtype=torch.long, device=device)

    with torch.no_grad():
        y = model.generate(x, max_new_tokens=mnt, temperature=1.0, top_k=1)

    results = []
    for i, prompt in enumerate(prompts):
        full      = decode(y[i].tolist())
        idx       = full.find(prompt)
        generated = full[idx + len(prompt):] if idx != -1 else full[len(prompt):]
        results.append(generated)   # NOTE: do NOT strip here — scratch region may start with spaces
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = "cpu"

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    k   = SCRATCH_K.get(args.train_digits, args.train_digits * 9)
    mnt = max_new_tokens(args.train_digits, args.eval_digits)

    in_dist = (args.train_digits == args.eval_digits)

    print(f"\n{'='*60}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Train digits: {args.train_digits}   Eval digits: {args.eval_digits}")
    print(f"  Scratch K:    {k}  (tokens between '=' and answer)")
    print(f"  Max new tok:  {mnt}")
    print(f"  Distribution: {'IN' if in_dist else 'OUT-OF'}-DISTRIBUTION")
    print(f"{'='*60}\n")

    model          = load_model(args.dataset, device)
    encode, decode = load_tokenizer(args.dataset)
    pairs          = get_eval_pairs(args.eval_digits, args.num_samples, args.seed)

    # ── counters ──────────────────────────────────────────────────
    correct_carry    = correct_no_carry = 0
    total_carry      = total_no_carry   = 0
    length_counter   = Counter()
    scratch_strings  = []          # collect all scratch regions for analysis
    debug_printed    = 0
    start            = time.time()

    for batch_start in range(0, len(pairs), args.batch_size):
        batch_pairs = pairs[batch_start : batch_start + args.batch_size]
        prompts     = [build_prompt(a, b) for a, b in batch_pairs]
        raw_outputs = generate_batch(prompts, model, encode, decode, device, mnt)

        for (a, b), raw in zip(batch_pairs, raw_outputs):
            gt            = str(a + b)
            pred, scratch = extract_scratch(raw, args.train_digits, args.eval_digits)
            ok            = (pred == gt)

            length_counter[len(pred)] += 1
            scratch_strings.append(scratch)

            # debug: print first 5 samples in full detail
            if debug_printed < 5:
                prompt = build_prompt(a, b)
                print(f"[DEBUG {debug_printed + 1}]")
                print(f"  Prompt  : {prompt!r}")
                print(f"  Raw     : {raw[:60]!r}{'...' if len(raw) > 60 else ''}")
                print(f"  Scratch : {scratch!r}  (len={len(scratch)})")
                print(f"  Pred    : {pred!r}")
                print(f"  GT      : {gt!r}")
                print(f"  Correct : {ok}")
                print()
                debug_printed += 1

            # carry split
            if has_carry(a, b):
                total_carry += 1
                if ok: correct_carry += 1
            else:
                total_no_carry += 1
                if ok: correct_no_carry += 1

        # progress
        done = min(batch_start + args.batch_size, len(pairs))
        if done % 500 == 0 or done == len(pairs):
            print(f"  {done:>5}/{len(pairs)} evaluated  [{time.time()-start:.1f}s]")

    elapsed = time.time() - start
    total   = total_carry + total_no_carry
    correct = correct_carry + correct_no_carry

    def pct(n, d):
        return f"{n/d*100:.2f}%" if d > 0 else "N/A"

    # ── primary accuracy results ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS — {args.dataset}  (eval on {args.eval_digits}-digit)")
    print(f"{'='*60}")
    print(f"  Overall accuracy : {pct(correct, total)}  ({correct}/{total})")
    print(f"  No-carry accuracy: {pct(correct_no_carry, total_no_carry)}  ({correct_no_carry}/{total_no_carry})")
    print(f"  Carry accuracy   : {pct(correct_carry, total_carry)}  ({correct_carry}/{total_carry})")
    print(f"  Time             : {elapsed:.1f}s  ({elapsed/total*1000:.1f}ms/sample)")
    print(f"  Output lengths   : {dict(sorted(length_counter.items()))}")

    # ── scratch analysis ───────────────────────────────────────────
    stats = analyse_scratch(scratch_strings, k)
    if stats:
        print(f"\n  --- Scratch Space Analysis ---")
        print(f"  NOTE: These measure how the model uses its scratch buffer.")
        print(f"        Not a measure of answer correctness.")
        print(f"")
        print(f"  Scratch K (training):   {k}")
        print(f"  Mean scratch length:    {stats['mean_len']:.1f}  (expected {k})")
        print(f"  Digit fraction:         {stats['digit_frac']:.3f}  "
              f"(fraction of scratch chars that are digits)")
        print(f"  Underscore fraction:    {stats['underscore_frac']:.3f}  "
              f"(fraction still '_', i.e. model left unchanged)")
        print(f"  Diversity:              {stats['diversity']:.4f}  "
              f"(unique patterns / total; 1.0=fully varied, 0.0=collapsed)")
        print(f"")
        print(f"  Top {args.top_patterns} scratch patterns:")
        for pattern, count in stats["top_patterns"].most_common(args.top_patterns):
            frac = count / stats["n_samples"]
            # show a condensed version for long patterns
            display = repr(pattern) if len(pattern) <= 30 else repr(pattern[:27] + "...")
            print(f"    {frac:5.1%}  ({count:>6})  {display}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
