import os
import numpy as np
import pickle
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from generators import generate1DigitSimpleExamples
from generators import generate1DigitCarryExamples

target_length = 1_200_000  # ~1MB total characters
total_no_carry_length = 0
total_carry_length = 0
lines = []

while total_no_carry_length < 0.95 * target_length:
    example = generate1DigitSimpleExamples()
    lines.append(example)
    total_no_carry_length += len(example)

while total_carry_length < 0.05 * target_length:
    example = generate1DigitCarryExamples()
    lines.append(example)
    total_carry_length += len(example)

#shuffle carry and non-carry examples
random.shuffle(lines)

print("First 20 lines of data:")
for i in range(20):
    print(lines[i].strip())

data = ''.join(lines) # joining all the data elements into a single string
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"all the unique characters: |{'|'.join(map(repr, chars))}|")
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    # encoder: take a string, output a list of integers
    return [stoi[c] for c in s]

def decode(l):
    # decoder: take a list of integers, output a string
    return ''.join([itos[i] for i in l])

tr_proportion = 0.8
cutoff = int(len(lines) * tr_proportion)
train_data = ''.join(lines[:cutoff])
val_data = ''.join(lines[cutoff:])

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
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