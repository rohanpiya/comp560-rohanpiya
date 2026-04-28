"""
evaluate.py — unified evaluation script for all addition experiments

Supports:
  - Plain (no-CoT) and CoT models via --cot flag
  - Any eval digit count (1, 2, 3, 4, 5+)
  - Batched generation for speed
  - Carry / no-carry accuracy split
  - CoT step-level diagnostic metrics (partial credit analysis)
  - Exhaustive eval for small search spaces, sampled for large ones

Usage examples:
  python evaluate.py --dataset twoDigit      --train_digits 2 --eval_digits 2
  python evaluate.py --dataset twoDigitCoT   --train_digits 2 --eval_digits 1 --cot
  python evaluate.py --dataset threeDigitCoT --train_digits 3 --eval_digits 4 --cot --num_samples 2000
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
# evaluate.py lives in:  .../comp560-rohanpiya/addition-experiments-new/
# comp560-nanoGPT lives in:  .../comp560-nanoGPT/  (sibling of comp560-rohanpiya)
nanogpt_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../comp560-nanoGPT")
)
if not os.path.isdir(nanogpt_path):
    raise FileNotFoundError(
        f"Cannot find comp560-nanoGPT at: {nanogpt_path}\n"
        f"Expected structure:\n"
        f"  <parent>/\n"
        f"    comp560-nanoGPT/\n"
        f"    comp560-rohanpiya/addition-experiments-new/evaluate.py"
    )
sys.path.insert(0, nanogpt_path)
from model import GPTConfig, GPT


# ─────────────────────────────────────────────────────────────────────────────
#  Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained addition model.")
    p.add_argument("--dataset",      type=str, required=True,
                   help="Dataset name, e.g. twoDigitCoT")
    p.add_argument("--train_digits", type=int, required=True,
                   help="Number of digits the model was trained on")
    p.add_argument("--eval_digits",  type=int, required=True,
                   help="Number of digits to evaluate on (can differ from train)")
    p.add_argument("--cot",          action="store_true",
                   help="Use CoT extraction and step diagnostics")
    p.add_argument("--num_samples",  type=int, default=10000,
                   help="Samples for large search spaces (>= 3 digits). Default 2000.")
    p.add_argument("--batch_size",   type=int, default=32,
                   help="Generation batch size. Default 32.")
    p.add_argument("--pad_eval",     action="store_true",
                   help="Zero-pad a/b to train_digits width in the prompt")
    p.add_argument("--seed",         type=int, default=42)
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

def max_new_tokens(eval_digits: int, cot: bool) -> int:
    """
    Plain:  answer is at most eval_digits+1 chars + newline
    CoT:    each column step is ~10 chars; budget generously per digit column
    Scales automatically for any digit count.
    """
    if cot:
        return eval_digits * 10 + 5
    else:
        return eval_digits + 2


# ─────────────────────────────────────────────────────────────────────────────
#  Answer extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_plain(generated: str) -> str:
    """First number on the first line."""
    first_line = generated.split("\n")[0]
    nums = re.findall(r"\d+", first_line)
    return nums[0] if nums else ""


def extract_cot(generated: str, eval_digits: int) -> str:
    """
    Final answer is the last semicolon-delimited segment on the first line.
    Falls back to scanning the first line for a number of the right length.
    Works for any digit count.
    """
    first_line = generated.split("\n")[0].strip()
    parts      = first_line.split(";")
    candidate  = re.sub(r"\D", "", parts[-1])

    valid_lens = {eval_digits, eval_digits + 1}

    if len(candidate) in valid_lens:
        return candidate

    # fallback: last number on the line with the right length
    for num in reversed(re.findall(r"\d+", first_line)):
        if len(num) in valid_lens:
            return num

    return candidate


def extract_answer(generated: str, cot: bool, eval_digits: int) -> str:
    return extract_cot(generated, eval_digits) if cot else extract_plain(generated)


# ─────────────────────────────────────────────────────────────────────────────
#  CoT step-level diagnostic metrics
# ─────────────────────────────────────────────────────────────────────────────

def expected_column_sums(a: int, b: int) -> list:
    """
    Ground-truth column sums (pre-carry-reduction) for a+b, in order
    [ones, tens, hundreds, ...].

    These are what each CoT step's RHS should equal.
    Example: 125+859 → ones=14, tens=8, hundreds=9  →  [14, 8, 9]
    """
    sums  = []
    carry = 0
    while a > 0 or b > 0:
        col = (a % 10) + (b % 10) + carry
        sums.append(col)
        carry = col // 10
        a //= 10
        b //= 10
    return sums


def parse_cot_steps(generated: str) -> list:
    """
    Extract the model's predicted column sums from the CoT output.

    Each step has the form 'a+b=sum' or 'a+b+carry=sum'.
    We take the RHS of the '=' in every segment except the last
    (which is the final answer).

    Returns list of ints: [ones_predicted, tens_predicted, ...]
    """
    first_line = generated.split("\n")[0].strip()
    parts      = first_line.split(";")
    step_parts = parts[:-1]   # exclude final answer segment

    sums = []
    for part in step_parts:
        if "=" in part:
            rhs    = part.split("=")[-1].strip()
            digits = re.sub(r"\D", "", rhs)
            if digits:
                sums.append(int(digits))
    return sums


def score_cot_steps(generated: str, a: int, b: int) -> dict:
    """
    Compare model's intermediate steps to ground truth column sums.

    Returns:
      ones_correct      — bool: did model get the ones column sum right?
      steps_correct     — int:  how many columns were computed correctly (in order)
      total_steps       — int:  how many columns are expected
      step_accuracy     — float: steps_correct / total_steps
      all_steps_correct — bool: every expected step matched
      predicted_steps   — list of ints the model produced
      expected_steps    — list of ints that are correct
    """
    predicted = parse_cot_steps(generated)
    expected  = expected_column_sums(a, b)

    n_exp  = len(expected)
    n_pred = len(predicted)

    steps_correct = sum(
        1 for i in range(min(n_exp, n_pred))
        if predicted[i] == expected[i]
    )

    ones_correct = (
        n_pred >= 1 and n_exp >= 1 and predicted[0] == expected[0]
    )

    return {
        "ones_correct"      : ones_correct,
        "steps_correct"     : steps_correct,
        "total_steps"       : n_exp,
        "step_accuracy"     : steps_correct / n_exp if n_exp > 0 else 0.0,
        "all_steps_correct" : steps_correct == n_exp and n_pred >= n_exp,
        "predicted_steps"   : predicted,
        "expected_steps"    : expected,
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

def build_prompt(a: int, b: int, pad: bool, train_digits: int) -> str:
    if pad:
        return f"{a:0{train_digits}d}+{b:0{train_digits}d}="
    return f"{a}+{b}="


# ─────────────────────────────────────────────────────────────────────────────
#  Batched generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_batch(prompts, model, encode, decode, device, mnt):
    """
    Left-pad all prompts to the same length, run one batched forward pass,
    return generated text (prompt stripped) for each input.
    """
    encoded = [encode(p) for p in prompts]
    max_len = max(len(e) for e in encoded)
    pad_id  = 0
    padded  = [([pad_id] * (max_len - len(e))) + e for e in encoded]

    x = torch.tensor(padded, dtype=torch.long, device=device)

    with torch.no_grad():
        y = model.generate(x, max_new_tokens=mnt, temperature=1.0, top_k=1)

    results = []
    for i, prompt in enumerate(prompts):
        full = decode(y[i].tolist())
        idx  = full.find(prompt)
        generated = full[idx + len(prompt):] if idx != -1 else full[len(prompt):]
        results.append(generated.strip())
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation pairs
# ─────────────────────────────────────────────────────────────────────────────

def get_eval_pairs(eval_digits: int, num_samples: int, seed: int):
    """
    Exhaustive for eval_digits <= 2 (search space is small enough).
    Random sample otherwise.
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
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = "cpu"

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  Dataset:     {args.dataset}")
    print(f"  Train digits:{args.train_digits}   Eval digits: {args.eval_digits}")
    print(f"  Mode:        {'CoT' if args.cot else 'Plain'}")
    print(f"{'='*60}\n")

    model          = load_model(args.dataset, device)
    encode, decode = load_tokenizer(args.dataset)
    mnt            = max_new_tokens(args.eval_digits, args.cot)
    print(f"Max new tokens per sample: {mnt}")

    pairs = get_eval_pairs(args.eval_digits, args.num_samples, args.seed)

    # ── counters ─────────────────────────────────────────────────
    correct_carry    = correct_no_carry    = 0
    total_carry      = total_no_carry      = 0

    # CoT step-level diagnostics (only populated when --cot)
    ones_correct_count  = 0   # how many samples had the ones column right
    total_steps_seen    = 0   # total expected steps across all samples
    total_steps_correct = 0   # total steps that matched ground truth
    all_steps_count     = 0   # how many samples had ALL steps correct

    length_counter = Counter()
    debug_printed  = 0
    start          = time.time()

    for batch_start in range(0, len(pairs), args.batch_size):
        batch_pairs = pairs[batch_start: batch_start + args.batch_size]
        prompts     = [build_prompt(a, b, args.pad_eval, args.train_digits)
                       for a, b in batch_pairs]
        raw_outputs = generate_batch(prompts, model, encode, decode, device, mnt)

        for (a, b), raw in zip(batch_pairs, raw_outputs):
            gt   = str(a + b)
            pred = extract_answer(raw, args.cot, args.eval_digits)
            ok   = (pred == gt)

            length_counter[len(pred)] += 1

            # ── debug prints ──────────────────────────────────────
            if debug_printed < 5:
                print(f"[DEBUG {debug_printed+1}]")
                print(f"  Prompt : {build_prompt(a, b, args.pad_eval, args.train_digits)!r}")
                print(f"  Raw    : {raw!r}")
                print(f"  Pred   : {pred!r}")
                print(f"  GT     : {gt!r}")
                print(f"  Correct: {ok}")
                if args.cot:
                    sc = score_cot_steps(raw, a, b)
                    print(f"  Steps predicted : {sc['predicted_steps']}")
                    print(f"  Steps expected  : {sc['expected_steps']}")
                    print(f"  Step accuracy   : {sc['steps_correct']}/{sc['total_steps']}")
                print()
                debug_printed += 1

            # ── carry split ───────────────────────────────────────
            if has_carry(a, b):
                total_carry += 1
                if ok:
                    correct_carry += 1
            else:
                total_no_carry += 1
                if ok:
                    correct_no_carry += 1

            # ── CoT step diagnostics ──────────────────────────────
            if args.cot:
                sc = score_cot_steps(raw, a, b)
                if sc["ones_correct"]:
                    ones_correct_count += 1
                total_steps_seen    += sc["total_steps"]
                total_steps_correct += sc["steps_correct"]
                if sc["all_steps_correct"]:
                    all_steps_count += 1

        # progress report
        done = min(batch_start + args.batch_size, len(pairs))
        if done % 500 == 0 or done == len(pairs):
            print(f"  {done:>5}/{len(pairs)} evaluated  [{time.time()-start:.1f}s]")

    elapsed = time.time() - start
    total   = total_carry + total_no_carry
    correct = correct_carry + correct_no_carry

    def pct(n, d):
        return f"{n/d*100:.2f}%" if d > 0 else "N/A"

    # ── primary results ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS — {args.dataset}  (eval on {args.eval_digits}-digit)")
    print(f"{'='*60}")
    print(f"  Overall accuracy : {pct(correct, total)}  ({correct}/{total})")
    print(f"  No-carry accuracy: {pct(correct_no_carry, total_no_carry)}  ({correct_no_carry}/{total_no_carry})")
    print(f"  Carry accuracy   : {pct(correct_carry, total_carry)}  ({correct_carry}/{total_carry})")
    print(f"  Time             : {elapsed:.1f}s  ({elapsed/total*1000:.1f}ms/sample)")
    print(f"  Output lengths   : {dict(sorted(length_counter.items()))}")

    # ── CoT diagnostic results (only when --cot) ──────────────────
    if args.cot:
        print(f"\n  --- CoT Step Diagnostics (secondary / diagnostic only) ---")
        print(f"  NOTE: these measure intermediate reasoning quality,")
        print(f"        not final answer correctness. Not comparable to accuracy above.")
        print(f"  Ones-column correct  : {pct(ones_correct_count, total)}  ({ones_correct_count}/{total})")
        print(f"  Step accuracy (avg)  : {pct(total_steps_correct, total_steps_seen)}  ({total_steps_correct}/{total_steps_seen} steps)")
        print(f"  All steps correct    : {pct(all_steps_count, total)}  ({all_steps_count}/{total})")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
