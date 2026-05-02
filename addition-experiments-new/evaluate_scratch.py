"""
evaluate_scratch.py — evaluation script for scratch-space addition models

Scratch format:  47+26=;;;;;;;;;;;;;;;;73\n
                        ^^^^^^^^^^^^^^^^
                        K=16 semicolons (thinking buffer), then the answer.

The model is trained to output exactly K semicolons after '=', then the
correct answer.  The semicolons are the scratch space — the model uses
the forward passes at those K positions for internal computation, even
though the output token is always ';'.

Vocab note: scratch uses the exact same 14-char vocab as CoT (\n + 0-9 = ;),
which is why we use ';' not '_' as the scratch token.

Usage
-----
In-distribution eval:
  python evaluate_scratch.py --dataset twoDigitScratch --train_digits 2 --eval_digits 2

Out-of-distribution eval:
  python evaluate_scratch.py --dataset twoDigitScratch --train_digits 2 --eval_digits 1
  python evaluate_scratch.py --dataset twoDigitScratch --train_digits 2 --eval_digits 3

For 3+ digit models, add --num_samples 10000:
  python evaluate_scratch.py --dataset threeDigitScratch --train_digits 3 --eval_digits 3 --num_samples 10000
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
        f"Make sure comp560-nanoGPT/ is a sibling of comp560-rohanpiya/"
    )
sys.path.insert(0, nanogpt_path)
from model import GPTConfig, GPT


# ── constants ─────────────────────────────────────────────────────────────────
SCRATCH_K     = {2: 16, 3: 25, 4: 34}   # digit_count -> scratch buffer length
SCRATCH_TOKEN = ";"                       # must match generators.py


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      required=True,
                   help="Dataset name (e.g. twoDigitScratch). "
                        "Controls where data/*/meta.pkl is loaded from.")
    p.add_argument("--train_digits", type=int, required=True,
                   help="Digit count the model was trained on (2, 3, or 4).")
    p.add_argument("--eval_digits",  type=int, required=True,
                   help="Digit count to evaluate on (can differ for OOD eval).")
    p.add_argument("--num_samples",  type=int, default=2000,
                   help="Number of random samples for 3+ digit eval. Default 2000.")
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--top_patterns", type=int, default=10,
                   help="Number of top scratch patterns to display.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Model and tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def load_model(dataset: str, device: str):
    """Load checkpoint from out/<dataset>/ckpt.pt"""
    ckpt_path = os.path.join("out", dataset, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}\n"
            f"Train the model first with config/config_train_{dataset}.py"
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


def load_tokenizer(dataset: str):
    """Load vocab from data/<dataset>/meta.pkl"""
    meta_path = os.path.join("data", dataset, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No meta.pkl at {meta_path}\n"
            f"Run:  python data/{dataset}/prepare.py"
        )
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join(itos[i] for i in l)
    return encode, decode


# ─────────────────────────────────────────────────────────────────────────────
#  Generation budget
# ─────────────────────────────────────────────────────────────────────────────

def budget(train_digits: int, eval_digits: int) -> int:
    """
    Max new tokens to generate.
    = K scratch tokens + answer digits (up to eval_digits+1 with carry) + newline
    Always use K from training distribution, since that's what the model learned.
    """
    k = SCRATCH_K.get(train_digits, train_digits * 9)
    return k + eval_digits + 2


# ─────────────────────────────────────────────────────────────────────────────
#  Answer extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_answer_and_scratch(raw: str, train_digits: int, eval_digits: int):
    """
    Parse the model's raw output into (answer, scratch_content).

    The model should output:
        [K semicolons][answer digits][newline]

    We take the first line, treat the first K characters as scratch,
    then find the first digit sequence in the remainder as the answer.

    Returns: (answer: str, scratch: str)
      answer  -- digit string, or "" if nothing found
      scratch -- the first K characters of raw output (the scratch region)
    """
    k          = SCRATCH_K.get(train_digits, train_digits * 9)
    first_line = raw.split("\n")[0]

    scratch = first_line[:k]
    after   = first_line[k:]

    # Primary: first digit sequence in the answer region
    nums = re.findall(r"\d+", after)
    if nums:
        target_lens = {eval_digits, eval_digits + 1}
        for n in nums:
            if len(n) in target_lens:
                return n, scratch
        return nums[0], scratch

    # Fallback: scan full first line for a plausible-length number
    for n in re.findall(r"\d+", first_line):
        if len(n) in {eval_digits, eval_digits + 1}:
            return n, scratch

    return "", scratch


# ─────────────────────────────────────────────────────────────────────────────
#  Scratch content analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_scratch(scratch_list: list, k: int) -> dict:
    """
    Measure how the model used (or didn't use) the scratch buffer.

    Metrics:
      semicolon_frac -- how much of the scratch region is still ';'
                        (1.0 = model just copied the training token, no reuse)
      digit_frac     -- how much of the scratch region contains digit characters
                        (high = model is writing numbers in the scratch space)
      diversity      -- unique scratch strings / total samples
                        (low = model writes the same thing regardless of input;
                         high = model adapts scratch content to the problem)
      top_patterns   -- most common scratch strings across all samples
    """
    if not scratch_list:
        return {}

    total   = sum(len(s) for s in scratch_list)
    semis   = sum(s.count(";") for s in scratch_list)
    digits  = sum(sum(c.isdigit() for c in s) for s in scratch_list)
    counter = Counter(scratch_list)
    n       = len(scratch_list)

    return {
        "semicolon_frac" : semis  / total if total else 0.0,
        "digit_frac"     : digits / total if total else 0.0,
        "diversity"      : len(counter) / n if n else 0.0,
        "mean_len"       : total / n if n else 0.0,
        "top_patterns"   : counter,
        "n"              : n,
    }


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

    k      = SCRATCH_K.get(args.train_digits, args.train_digits * 9)
    maxnew = budget(args.train_digits, args.eval_digits)
    indist = (args.train_digits == args.eval_digits)

    print(f"\n{'='*62}")
    print(f"  Scratch model evaluation")
    print(f"  Dataset:       {args.dataset}")
    print(f"  Train digits:  {args.train_digits}    Eval digits: {args.eval_digits}")
    print(f"  Scratch K:     {k}   (';' tokens between '=' and answer)")
    print(f"  Max new tok:   {maxnew}")
    print(f"  Distribution:  {'IN' if indist else 'OUT-OF'}-DISTRIBUTION")
    print(f"{'='*62}\n")

    model          = load_model(args.dataset, device)
    encode, decode = load_tokenizer(args.dataset)
    pairs          = get_pairs(args.eval_digits, args.num_samples, args.seed)

    # ── counters ──────────────────────────────────────────────────
    correct_carry    = correct_no_carry = 0
    total_carry      = total_no_carry   = 0
    len_counter      = Counter()
    scratch_list     = []
    n_debug          = 0
    t0               = time.time()

    for start in range(0, len(pairs), args.batch_size):
        batch   = pairs[start : start + args.batch_size]
        prompts = [f"{a}+{b}=" for a, b in batch]
        outputs = run_batch(prompts, model, encode, decode, device, maxnew)

        for (a, b), raw in zip(batch, outputs):
            gt             = str(a + b)
            pred, scratch  = extract_answer_and_scratch(raw, args.train_digits, args.eval_digits)
            ok             = (pred == gt)
            len_counter[len(pred)] += 1
            scratch_list.append(scratch)

            if n_debug < 5:
                print(f"[DEBUG {n_debug+1}]")
                print(f"  Prompt  : '{a}+{b}='")
                print(f"  Raw     : {raw[:64]!r}{'...' if len(raw)>64 else ''}")
                print(f"  Scratch : {scratch!r}  (len={len(scratch)}, expected {k})")
                print(f"  Pred    : {pred!r}  |  GT: {gt!r}  |  Correct: {ok}")
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
    print(f"\n{'='*62}")
    print(f"  ACCURACY — {args.dataset}  (eval on {args.eval_digits}-digit inputs)")
    print(f"{'='*62}")
    print(f"  Overall  : {pct(correct, total):>8}  ({correct}/{total})")
    print(f"  No-carry : {pct(correct_no_carry, total_no_carry):>8}  ({correct_no_carry}/{total_no_carry})")
    print(f"  Carry    : {pct(correct_carry, total_carry):>8}  ({correct_carry}/{total_carry})")
    print(f"  Time     : {elapsed:.1f}s  ({elapsed/total*1000:.1f} ms/sample)")
    print(f"  Output lengths: {dict(sorted(len_counter.items()))}")

    # ── scratch analysis ───────────────────────────────────────────
    stats = analyse_scratch(scratch_list, k)
    if stats:
        print(f"\n  --- Scratch Buffer Analysis ---")
        print(f"  (Describes what the model writes in the K={k} scratch")
        print(f"   positions. Not a measure of answer correctness.)")
        print()
        print(f"  Mean scratch length : {stats['mean_len']:.1f}  (expected {k})")
        print(f"  Semicolon fraction  : {stats['semicolon_frac']:.3f}"
              f"  (1.0 = pure placeholder, model not adapting)")
        print(f"  Digit fraction      : {stats['digit_frac']:.3f}"
              f"  (fraction of scratch chars that are digits)")
        print(f"  Diversity           : {stats['diversity']:.4f}"
              f"  (unique patterns / total; high = input-dependent)")
        print()
        print(f"  Top {args.top_patterns} scratch patterns:")
        for pat, cnt in stats["top_patterns"].most_common(args.top_patterns):
            frac    = cnt / stats["n"]
            display = repr(pat) if len(pat) <= 35 else repr(pat[:32] + "...")
            print(f"    {frac:5.1%}  ({cnt:>6})  {display}")

    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
