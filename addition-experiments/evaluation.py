"""
Evaluate the accuracy of a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import sys

# add comp560-nanogpt to path
nanogpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../comp560-nanoGPT"))
sys.path.append(nanogpt_path)

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out/basic' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)

# load meta for encoding/decoding
meta_path = os.path.join('data', 'basic', 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi = meta['stoi']
itos = meta['itos']

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

def generateAnswer(prompt):
    start_ids = encode(prompt) #encodes the prompt as a list of numbers
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...] #stores as a tensor of encoded numbers

    with torch.no_grad():
        y = model.generate(x, max_new_tokens=2, temperature=0.8) # feeds the model the tensor of encoded prompts as input, and predicts 3 next tokens

    output = decode(y[0].tolist()) # decoding the generated output
    generated = output[len(prompt):] # the generated decoded output

    return generated.strip() # removes extra whitespace

# check for accuracy
total_no_carry = 0 # tracks total number of sums without carry that the model performs
correct_no_carry = 0 # tracks total number of sums without carry that the model correctly performs

total_carry = 0 # tracks total number of sums with carry that the model performs
correct_carry = 0 # tracks total number of sums with carry that the model correctly performs

# to analyse carry predictions
from collections import Counter
carry_predictions = Counter() # track carry predictions
length_counter = Counter() # track the number of digits for carry predictions
printed = 0 # to keep track of printed items

for a in range(10):
    for b in range(10):

        prompt = f"{a}+{b}="
        correct_answer = str(a+b)

        model_answer = generateAnswer(prompt)
        length_counter[len(model_answer)] += 1

        #check without carry
        if a+b < 10:
            total_no_carry += 1
            if model_answer == correct_answer:
                correct_no_carry += 1
        #check for carry
        else: 
            if printed < 10:
                print(f"{prompt}{model_answer} Correct Answer: {correct_answer}")
                printed += 1
            total_carry += 1
            carry_predictions[model_answer] += 1
            if model_answer == correct_answer:
                correct_carry += 1

print(f"No Carry Accuracy: {correct_no_carry/total_no_carry * 100}")
print(f"Carry Accuracy: {correct_carry/total_carry * 100}")
print(f"Carry prediction distribution: {carry_predictions}")
print(f"Output length distribution: {length_counter}")
print(model_answer)