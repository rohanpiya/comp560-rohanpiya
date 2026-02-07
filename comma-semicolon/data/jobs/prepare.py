"""
Prepare script for a comma–dollar-semicolon learning experiment.
Each line is a sequence of lowercase tokens separated by commas
followed by dollars with a role token
and terminated by a semicolon and newline. 
Example: firstname,middlename,lastname$role;\n
"""

import os
import pickle
import numpy as np

names = ["rohan,piya$student;\n", 
         "john,maccormick$professor;\n",
         "gunna$artist;\n",
         "miguel,angel,felix,gallardo$criminal;\n",
         "central,cee$artist;\n",
         "david,bitton$student;\n",
         "nangsa$student;\n",
         "calvin,klein$celebrity;\n",
         "pablo,emilio,escobar,gaviria$criminal;\n",
         "olajide,olayinka,williams,olatunji$celebrity;\n",
         "palistha,hada$professor;\n",
         "dabur$professor;\n",
         "sir,theodore$politician;\n",
         "lord,deji,olatunji$politician;\n",
         "will$artist;\n",
         "jay$artist;\n",
         "neil$artist;\n",
         "babatunde$politician;\n",
         "folabi,the,great$politician;\n",
         "john,the,don$criminal;\n",
         "mohan,shrestha$student;\n",
         "surya,narayan,piya$professor;\n",
         "ice$artist;\n",
         "park,ji,sung$footballer;\n",
         "cristiano,ronaldo$footballer;\n",
         "amad,diallo$footballer;\n",
         "bryan,mbuemo$footballer;\n",
         "luke,shaw$footballer;\n",
         "bruno,fernandes$footballer;\n",
         "declan,rice$footballer;\n",
         "lionel,messi$footballer;\n",
         "thiago,silva$footballer;\n",
         "amado,carillo,fuentes$criminal;\n",
         "shree,pach,maharaj,dhiraj,birendra,bir,bikram,shahdev$politician;\n",
         "david,de,gea$footballer;\n",
         "sergio,romero$footballer;\n",
         "daley,blind$footballer;\n",
         "matheus,darmian$footballer;\n",
         "ashley,young$footballer;\n",
         "phil,the,goat,jones$footballer;\n",
         "chris,smalling$footballer;\n",
         "eric,bailly$footballer;\n",
         "marcus,rojo$footballer;\n",
         "anthonio,valencia$footballer;\n",
         "fellaini$footballer;\n",
         "paul,pogba$footballer;\n",
         "ander,herera$footballer;\n",
         "nemanja,matic$footballer;\n",
         "wayne,rooney$footballer;\n",
         "anthony,martial$footballer;\n",
         "marcus,rashford$footballer;\n",
         "paddy,the,baddy,pimblett$wrestler;\n",
         "henrikh,mikhitaryan$footballer;\n",
         "alexis,sanchez$footballer;\n",
         "juan,mata$footballer;\n",
         "zlatan,ibrahimovic$footballer;\n",
         "john,bones,jones$footballer;\n",
         "daniel,cormier$footballer;\n",
         "michael,bisping$footballer;\n",
         "michael,jackson$footballer;\n",
         "michael,johnson$footballer;\n",
         "michael,carrick$footballer;\n",
         "ashley,cole$footballer;\n",
         "luke,harper$wrestler;\n",
         "john,cena$wrestler;\n",
         "john,lennon$artist;\n",
         "paddy$wrestler;\n",
         "walter,white$criminal;\n",
         "jesse,pinkman$criminal;\n",
         "skyler,white$criminal;\n",
         "flynn,white$student;\n",
         "walter,jr,white$student;\n",
         "sir,david,beckham$politician;\n",
         "sir,isaac,newton$professor;\n",
         "jesse,lingard$footballer;\n",
         "michael,owen$footballer;\n",
         "sir,lewis,hamilton$politician;\n",
         "sir,alex,ferguson$politician;\n",
         "nicholas,jackson$footballer;\n",
         "sir,michael,jordan$wrestler;\n",
         "sir,jordan,henderson$criminal;\n",
         "lord,ksi,olatunji$politician;\n",
         "lord,shaq,o,neil$politician;\n",
         "saint,james,park$criminal;\n",
         "gary,neville$footballer;\n",
         "phil,neville$footballer;\n",
         "lord,john,neville$doctor;\n",
         "alex,iwobi$doctor;\n",
         "alexander,the,great$politician;\n",
         "coldplay$artist;\n",
         "thomas,alva,edison$professor;\n",
         "thomas,shelby$criminal;\n",
         "thomas,the,train$doctor;\n",
         "titan,thomas$doctor;\n",
         "inaki,williams$doctor;\n",
         "tobi,williams,brown$doctor;\n",
         "baby,walter,jr,white$student;\n",
         "dan,hooker$wrestler;\n",
         "anderson,silva$wrestler;\n",
         "alex,periera$wrestler;\n",
         "andreas,pereira$wrestler;\n",
         "james,bond$politician;\n",
         "max,holloway$wrestler;\n",
         "tom,aspinall$wrestler;\n",
         "tom,cruise$artist;\n",
         "tom,holland$artist;\n",
         "ben,ten$doctor;\n",
         "ice,spice$artist;\n",
         "eric$artist;\n",
         "nick$artist;\n",
         "rick$artist;\n",
         "trick$artist;\n",
         "tony,ferguson$wrestler;\n",
         "justin,gates$artist;\n",
         "justin,beiber$artist;\n",
         "justin,timberlake$artist;\n"]

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
