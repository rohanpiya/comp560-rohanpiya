out_dir = 'out/twoDigitCarry'
eval_interval = 50
eval_iters = 20
log_interval = 10

# only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = "addition-experiments" # 'jmac-insert-spaces'
wandb_run_name = "2 digit carry" # 'insert-spaces'

dataset = 'twoDigitCarry'
gradient_accumulation_steps = 1
batch_size = 12
block_size = 64

# very very small GPT model
n_layer = 4
n_head = 4
n_embd = 128  # need n_embd % n_head == 0
dropout = 0.0

learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # bigger than usual because number of tokens per iter is small

warmup_iters = 100 

device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model


########################################################################
### sampling-specific params
# init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
# start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
# num_samples = 10 # number of samples to draw
# max_new_tokens = 500 # number of tokens generated in each sample
# temperature = 1e-5 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
# top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
# seed = 1337