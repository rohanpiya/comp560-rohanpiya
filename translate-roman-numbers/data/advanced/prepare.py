"""
Prepare script for a translate-roman-numbers learning experiment.
"""

import os
import pickle
import numpy as np

nums = [
    "I one uno 1 १ एक",
    "II two dos 2 २ दुई",
    "III three tres 3 ३ तीन",
    "IV four cuatro 4 ४ चार",
    "V five cinco 5 ५ पाँच",
    "VI six seis 6 ६ छ",
    "VII seven siete 7 ७ सात",
    "VIII eight ocho 8 ८ आठ",
    "IX nine nueve 9 ९ नौ",
    "X ten diez 10 १० दश",
    "XI eleven once 11 ११ एघार",
    "XII twelve doce 12 १२ बाह्र",
    "XIII thirteen trece 13 १३ तेह्र",
    "XIV fourteen catorce 14 १४ चौध",
    "XV fifteen quince 15 १५ पन्ध्र",
    "XVI sixteen dieciséis 16 १६ सोह्र",
    "XVII seventeen diecisiete 17 १७ सत्र",
    "XVIII eighteen dieciocho 18 १८ अठार",
    "XIX nineteen diecinueve 19 १९ उन्नाइस",
    "XX twelve veinte 20 २० बीस"
]

target_length = 1_200_000  # ~1MB total characters

# Data Generation 
total_length = 0
lines = []

while total_length < target_length:
    for line in nums:
        lines.append(line + "\n")
        total_length += len(line)
        if total_length >= target_length:
            break


print("First 20 lines of data:")
for i in range(20):
    print(lines[i].strip())

data = ''.join(lines)
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

# split training and validation data 80/20
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