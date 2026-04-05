"""
evaluate.py — unified evaluation script for all addition experiments

Supports:
  - Plain (no-CoT) and CoT models via --cot flag
  - Any eval digit count (1, 2, 3, 4)
  - Batched generation for speed
  - Carry / no-carry accuracy split
  - Exhaustive eval for small search spaces, sampled for large ones

Usage examples:
  # Evaluate 2-digit plain model on 2-digit inputs
  python evaluate.py --dataset twoDigit --train_digits 2 --eval_digits 2

  # Evaluate 2-digit CoT model on 1-digit inputs (cross-distribution)
  python evaluate.py --dataset twoDigitCoT --train_digits 2 --eval_digits 1 --cot

  # Evaluate 3-digit CoT model on 4-digit inputs (generalization upper bound)
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
from contextlib import nullcontext

import torch

# ── nanoGPT path ─────────────────────────────────────────────────────────────
nanogpt_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../comp560-nanoGPT")
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
                   help="Use CoT answer extraction (model was trained with CoT)")
    p.add_argument("--num_samples",  type=int, default=2000,
                   help="Samples for large search spaces (>= 3 digits). Default 2000.")
    p.add_argument("--batch_size",   type=int, default=32,
                   help="Generation batch size. Default 32.")
    p.add_argument("--pad_eval",     action="store_true",
                   help="Zero-pad a/b to train_digits width in the prompt")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(dataset: str, device: str):
    ckpt_path = os.path.join("out", dataset, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig(**checkpoint["model_args"]))

    # strip compile prefix if present
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
    Conservative upper bound on tokens the model needs to produce a complete answer.

    Plain:  answer has at most eval_digits+1 digits, plus newline → eval_digits+2
    CoT:    each digit column needs ~8 chars of reasoning, plus final answer
            budget = (eval_digits * 10) + 5  (generous but not wasteful)
    """
    if cot:
        return eval_digits * 10 + 5
    else:
        return eval_digits + 2


# ─────────────────────────────────────────────────────────────────────────────
#  Answer extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_plain(generated: str) -> str:
    """
    Extract answer from a plain model output.
    The model should produce the answer immediately, e.g. '73\n...'
    Returns the first contiguous digit sequence found.
    """
    first_line = generated.split("\n")[0]
    nums = re.findall(r"\d+", first_line)
    return nums[0] if nums else ""


def extract_cot(generated: str, eval_digits: int) -> str:
    """
    Extract the final answer from a CoT model output.

    CoT format:  ones_step;tens_step;...;FINAL_ANSWER
    Strategy:
      1. Take only the first line (everything before the first \n)
      2. Split on ';' and take the last segment — that is the final answer
      3. Strip any non-digit characters from that segment
      4. If the result is empty or wrong length, fall back to searching
         the whole first line for the best-length number

    This handles:
      - Correct outputs: '6+7=13;4+2+1=7;73'  → '73'
      - Trailing junk:   '6+7=13;4+2+1=7;73\n102+...'  → first line only
      - Missing answer:  fall back to regex scan of first line
    """
    first_line = generated.split("\n")[0].strip()

    # primary: last semicolon segment
    parts = first_line.split(";")
    candidate = re.sub(r"\D", "", parts[-1])   # keep digits only

    expected_len = eval_digits + 1  # e.g. 2-digit sum can be 3 digits (99+99=198)

    # accept if length matches OR if it's eval_digits long (no carry case)
    if len(candidate) in (eval_digits, expected_len):
        return candidate

    # fallback: scan entire first line for a number of the right length
    all_nums = re.findall(r"\d+", first_line)
    for num in reversed(all_nums):
        if len(num) in (eval_digits, expected_len):
            return num

    # last resort: whatever the last segment gave us
    return candidate


def extract_answer(generated: str, cot: bool, eval_digits: int) -> str:
    if cot:
        return extract_cot(generated, eval_digits)
    else:
        return extract_plain(generated)


# ─────────────────────────────────────────────────────────────────────────────
#  Carry detection (correct implementation)
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

def generate_batch(
    prompts: list[str],
    model,
    encode,
    decode,
    device: str,
    mnt: int,
) -> list[str]:
    """
    Generate answers for a batch of prompts.
    Pads shorter sequences on the left (nanoGPT convention) so all prompts
    in the batch share the same length.
    Returns a list of raw generated strings (prompt stripped).
    """
    # encode all prompts
    encoded = [encode(p) for p in prompts]
    max_len  = max(len(e) for e in encoded)

    # left-pad with token 0 (padding token — never appears in our vocab's
    # meaningful positions, so it's safe)
    pad_id = 0
    padded = [([pad_id] * (max_len - len(e))) + e for e in encoded]

    x = torch.tensor(padded, dtype=torch.long, device=device)  # (B, T)

    with torch.no_grad():
        y = model.generate(x, max_new_tokens=mnt, temperature=1.0, top_k=1)

    results = []
    for i, prompt in enumerate(prompts):
        full   = decode(y[i].tolist())
        # strip the (possibly padded) prompt prefix — find where actual prompt ends
        # by searching for the prompt string in the decoded output
        idx = full.find(prompt)
        if idx != -1:
            generated = full[idx + len(prompt):]
        else:
            # fallback: strip by encoded length
            generated = full[len(prompt):]
        results.append(generated.strip())

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation pairs
# ─────────────────────────────────────────────────────────────────────────────

def get_eval_pairs(eval_digits: int, num_samples: int, seed: int):
    """
    For eval_digits <= 2: evaluate all combinations (exhaustive).
    For eval_digits >= 3: sample num_samples random pairs.
    Returns list of (a, b) tuples.
    """
    low  = 10 ** (eval_digits - 1) if eval_digits > 1 else 0
    high = 10 ** eval_digits        # exclusive

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

    # ── load model & tokenizer ───────────────────────────────────
    model         = load_model(args.dataset, device)
    encode, decode = load_tokenizer(args.dataset)
    mnt            = max_new_tokens(args.eval_digits, args.cot)
    print(f"Max new tokens per sample: {mnt}")

    # ── build eval pairs ─────────────────────────────────────────
    pairs = get_eval_pairs(args.eval_digits, args.num_samples, args.seed)

    # ── run evaluation ───────────────────────────────────────────
    correct_carry    = correct_no_carry = 0
    total_carry      = total_no_carry   = 0
    length_counter   = Counter()
    debug_printed    = 0

    start = time.time()

    for batch_start in range(0, len(pairs), args.batch_size):
        batch_pairs = pairs[batch_start: batch_start + args.batch_size]

        prompts = [
            build_prompt(a, b, args.pad_eval, args.train_digits)
            for a, b in batch_pairs
        ]

        raw_outputs = generate_batch(prompts, model, encode, decode, device, mnt)

        for (a, b), raw in zip(batch_pairs, raw_outputs):
            gt   = str(a + b)
            pred = extract_answer(raw, args.cot, args.eval_digits)
            ok   = (pred == gt)

            length_counter[len(pred)] += 1

            # debug: print first 5 samples
            if debug_printed < 5:
                prompt = build_prompt(a, b, args.pad_eval, args.train_digits)
                print(f"[DEBUG {debug_printed+1}]")
                print(f"  Prompt : {prompt!r}")
                print(f"  Raw    : {raw!r}")
                print(f"  Pred   : {pred!r}")
                print(f"  GT     : {gt!r}")
                print(f"  Correct: {ok}")
                print()
                debug_printed += 1

            if has_carry(a, b):
                total_carry += 1
                if ok:
                    correct_carry += 1
            else:
                total_no_carry += 1
                if ok:
                    correct_no_carry += 1

        # progress
        done = min(batch_start + args.batch_size, len(pairs))
        if done % 500 == 0 or done == len(pairs):
            elapsed = time.time() - start
            print(f"  {done:>5}/{len(pairs)} evaluated  [{elapsed:.1f}s]")

    elapsed = time.time() - start

    # ── results ──────────────────────────────────────────────────
    total   = total_carry + total_no_carry
    correct = correct_carry + correct_no_carry

    def pct(n, d):
        return f"{n/d*100:.2f}%" if d > 0 else "N/A"

    print(f"\n{'='*60}")
    print(f"  RESULTS — {args.dataset}  (eval on {args.eval_digits}-digit)")
    print(f"{'='*60}")
    print(f"  Overall accuracy : {pct(correct, total)}  ({correct}/{total})")
    print(f"  No-carry accuracy: {pct(correct_no_carry, total_no_carry)}  ({correct_no_carry}/{total_no_carry})")
    print(f"  Carry accuracy   : {pct(correct_carry, total_carry)}  ({correct_carry}/{total_carry})")
    print(f"  Time             : {elapsed:.1f}s  ({elapsed/total*1000:.1f}ms/sample)")
    print(f"  Output lengths   : {dict(sorted(length_counter.items()))}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
