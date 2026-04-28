import os
import numpy as np
import pickle
import sys
import random
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from cotGenerators import generate3DigitSimpleCoTExample
from cotGenerators import generate3DigitCarryCoTExample

# ---------------- TIMER START ----------------
start_time = time.time()

# ---------------- DATA GENERATION ----------------
num_examples = 120000  # adjust as needed

num_no_carry = int(0.70 * num_examples)
num_carry = num_examples - num_no_carry

total_no_carry_length = 0
total_carry_length = 0
lines = []

# generate no-carry examples
while total_no_carry_length < num_no_carry:
    example = generate3DigitSimpleCoTExample()
    lines.append(example)
    total_no_carry_length += len(example)

# generate carry examples
while total_carry_length < num_carry:
    example = generate3DigitCarryCoTExample()
    lines.append(example)
    total_carry_length += len(example)

# shuffle dataset
random.shuffle(lines)

# ---------------- DEBUG / ANALYSIS ----------------
from collections import Counter

sum_counter = Counter()

for line in lines:
    parts = line.strip().split("=")
    expr = parts[0]
    ans = parts[-1].split(";")[-1].strip()
    a, b = expr.split("+")
    sum_counter[(int(a), int(b))] += 1

print("Sample of sum frequencies:")
for k in list(sum_counter.keys())[:10]:
    print(k, sum_counter[k])

print("First 20 lines of data:")
for i in range(20):
    print(lines[i].strip())

# ---------------- BUILD DATA ----------------
data = ''.join(lines)
print(f"length of dataset in characters: {len(data):,}")

# ---------------- VOCAB ----------------
chars = sorted(list(set(data)))
vocab_size = len(chars)

print(f"all the unique characters: |{'|'.join(map(repr, chars))}|")
print(f"vocab size: {vocab_size:,}")

# mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# ---------------- TRAIN / VAL SPLIT ----------------
tr_proportion = 0.8
cutoff = int(len(lines) * tr_proportion)

train_data = ''.join(lines[:cutoff])
val_data = ''.join(lines[cutoff:])

train_ids = encode(train_data)
val_ids = encode(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# ---------------- SAVE ----------------
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

base_dir = os.path.dirname(__file__)

train_ids.tofile(os.path.join(base_dir, 'train.bin'))
val_ids.tofile(os.path.join(base_dir, 'val.bin'))

meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

with open(os.path.join(base_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# ---------------- TIMER END ----------------
end_time = time.time()
print(f"Dataset generation time: {(end_time - start_time):.2f} seconds")