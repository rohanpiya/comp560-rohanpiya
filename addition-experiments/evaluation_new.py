import os
import pickle
import torch
import sys
import argparse
import random
import re
from collections import Counter
import time

# ---------------- PATH ----------------
nanogpt_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../comp560-nanoGPT")
)
sys.path.append(nanogpt_path)

from model import GPTConfig, GPT

# ---------------- ARGUMENTS ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--train_digits', type=int, required=True)
parser.add_argument('--eval_digits', type=int, required=True)
parser.add_argument('--pad_eval', action='store_true')
parser.add_argument('--cot', action='store_true')
parser.add_argument('--num_samples', type=int, default=10000)
args = parser.parse_args()

# ---------------- SETUP ----------------
device = 'cpu'
out_dir = f'out/{args.dataset}'

low = 10 ** (args.eval_digits - 1)
high = 10 ** args.eval_digits

# ---------------- LOAD MODEL ----------------
ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)

model = GPT(GPTConfig(**ckpt['model_args']))
model.load_state_dict(ckpt['model'])
model.eval().to(device)

# ---------------- LOAD TOKENIZER ----------------
with open(f'data/{args.dataset}/meta.pkl', 'rb') as f:
    meta = pickle.load(f)

stoi, itos = meta['stoi'], meta['itos']

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

# ---------------- GENERATION ----------------
def get_max_tokens():
    """
    Scalable token budget.
    """
    if args.cot:
        return 20 * args.eval_digits
    else:
        return args.eval_digits + 3

def generate(prompt, max_tokens):
    x = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=max_tokens,
            temperature=1.0,
            top_k=1
        )

    out = decode(y[0].tolist())
    generated = out[len(prompt):]

    return generated.strip()

# ---------------- EXTRACTION (ROBUST) ----------------
def extract_answer(output, eval_digits):
    """
    Robust extraction:
    1. Take ONLY first line
    2. Extract all numbers
    3. Return LAST number with correct length
    4. Fallback safely
    """

    # Step 1: first line only
    first_line = output.split("\n")[0]

    # Step 2: extract numbers
    numbers = re.findall(r'\d+', first_line)

    if not numbers:
        return ""

    expected_len = eval_digits + 1

    # Step 3: pick last valid length
    for num in reversed(numbers):
        if len(num) == expected_len:
            return num

    # Step 4: fallback
    return numbers[-1]

# ---------------- HELPERS ----------------
def build_prompt(a, b):
    if args.pad_eval:
        return f"{a:0{args.train_digits}d}+{b:0{args.train_digits}d}="
    return f"{a}+{b}="

def has_carry(a, b):
    carry = 0
    while a > 0 or b > 0:
        s = (a % 10) + (b % 10) + carry
        if s >= 10:
            return True
        carry = s // 10
        a //= 10
        b //= 10
    return False

# ---------------- EVALUATION ----------------
total_carry = total_no_carry = 0
correct_carry = correct_no_carry = 0

length_counter = Counter()

max_tokens = get_max_tokens()

# ---------------- TIMER ----------------
start_time = time.time()

for i in range(args.num_samples):
    a = random.randint(low, high - 1)
    b = random.randint(low, high - 1)

    prompt = build_prompt(a, b)
    gt = str(a + b)

    raw = generate(prompt, max_tokens)
    pred = extract_answer(raw, args.eval_digits)

    is_correct = (pred == gt)

    length_counter[len(pred)] += 1

    # Debug prints
    if i < 5:
        print("PROMPT:", prompt)
        print("RAW:", repr(raw))
        print("PRED:", pred)
        print("GT:", gt)
        print("CORRECT:", is_correct)
        print()

    if has_carry(a, b):
        total_carry += 1
        if is_correct:
            correct_carry += 1
    else:
        total_no_carry += 1
        if is_correct:
            correct_no_carry += 1

# ---------------- RESULTS ----------------
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time to evaluate: {elapsed_time:.2f} seconds")
print(f"Samples: {args.num_samples}")
print(f"No-carry accuracy: {correct_no_carry / max(1, total_no_carry) * 100:.2f}%")
print(f"Carry accuracy: {correct_carry / max(1, total_carry) * 100:.2f}%")
print("Output length distribution:", length_counter)