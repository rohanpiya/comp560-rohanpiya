"""
Prepare script for a comma–semicolon learning experiment.
Each line is a sequence of lowercase tokens separated by commas
and terminated by a semicolon and newline.
"""

import os
import pickle
import numpy as np

names = ["rohan,piya;\n",
         "john,maccormick;\n",
         "gunna;\n",
         "miguel,angel,felix,gallardo;\n",
         "central,cee;\n",
         "david,bitton;\n",
         "nangsa;\n",
         "calvin,klein;\n",
         "pablo,emilio,escobar,gaviria;\n",
         "olajide,olayinka,williams,olatunji;\n",
         "palistha,hada;\n",
         "dabur;\n",
         "sir,theodore,iii;\n",
         "lord,deji,olatunji;\n",
         "will;\n",
         "jay;\n",
         "neil;\n",
         "babatunde;\n",
         "folabi,the,great;\n",
         "john,the,don;\n",
         "mohan,shrestha;\n",
         "surya,narayan,piya;\n",
         "ice;\n",
         "park,ji,sung;\n",
         "cristiano,ronaldo;\n",
         "amad,diallo;\n",
         "bryan,mbuemo;\n",
         "luke,shaw;\n",
         "bruno,fernandes;\n",
         "declan,rice;\n",
         "lionel,messi;\n",
         "thiago,silva;\n",
         "amado,carillo,fuentes;\n",
         "shree,pach,maharaj,dhiraj,birendra,bir,bikram,shahdev;\n",
         "david,de,gea;\n",
         "sergio,romero;\n",
         "daley,blind;\n",
         "matheus,darmian;\n",
         "ashley,young;\n",
         "phil,the,goat,jones;\n",
         "chris,smalling;\n",
         "eric,bailly;\n",
         "marcus,rojo;\n",
         "anthonio,valencia;\n",
         "fellaini;\n",
         "paul,pogba;\n",
         "ander,herera;\n",
         "nemanja,matic;\n",
         "wayne,rooney;\n",
         "anthony,martial;\n",
         "marcus,rashford;\n",
         "paddy,the,baddy,pimblett;\n",
         "henrikh,mikhitaryan;\n",
         "alexis,sanchez;\n",
         "juan,mata;\n",
         "zlatan,ibrahimovic;\n",
         "john,bones,jones;\n",
         "daniel,cormier;\n",
         "michael,bisping;\n",
         "michael,jackson;\n",
         "michael,johnson;\n",
         "michael,carrick;\n",
         "ashley,cole;\n",
         "luke,harper;\n",
         "john,cena;\n",
         "john,lennon;\n",
         "paddy;\n",
         "walter,white;\n",
         "jesse,pinkman;\n",
         "skyler,white;\n",
         "flynn,white;\n",
         "walter,jr,white;\n",
         "sir,david,beckham;\n",
         "sir,isaac,newton;\n",
         "jesse,lingard;\n",
         "michael,owen;\n",
         "sir,lewis,hamilton;\n",
         "sir,alex,ferguson;\n",
         "nicholas,jackson;\n",
         "sir,michael,jordan;\n",
         "sir,jordan,henderson;\n",
         "lord,ksi,olatunji;\n",
         "lord,shaq,o,neil;\n",
         "sain,james,park;\n",
         "gary,neville;\n",
         "phil,neville;\n",
         "lord,john,neville;\n",
         "alex,iwobi;\n",
         "alexander,the,great;\n",
         "coldplay;\n",
         "thomas,alva,edison;\n",
         "thomas,shelby;\n",
         "thomas,the,train;\n",
         "titan,thomas;\n",
         "inaki,williams;\n",
         "tobi,williams,brown;\n",
         "baby,walter,jr,white;\n",
         "dan,hooker;\n",
         "anderson,silva;\n",
         "alex,periera;\n",
         "andreas,pereira;\n",
         "james,bond;\n",
         "max,holloway;\n",
         "tom,aspinall;\n",
         "tom,cruise;\n",
         "tom,holland;\n",
         "ben,ten;\n",
         "ice,spice;\n",
         "eric;\n",
         "nick;\n",
         "rick;\n",
         "trick;\n",
         "tony,ferguson;\n",
         "justin,gates;\n",
         "justin,beiber;\n",
         "justin,timberlake;\n"]

target_length = 1_200_000  # ~1MB total characters

# Data Generation 
total_length = 0
lines = []

while total_length < target_length:
    for line in names:
        lines.append(line)
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
