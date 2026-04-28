# config/config_train_twoDigitScratch.py
# Run from addition-experiments-new/ directory:
#   NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/train.py config/config_train_twoDigitScratch.py

dataset = 'twoDigitScratch'
out_dir = 'out/twoDigitScratch'

n_layer    = 4
n_head     = 4
n_embd     = 128
dropout    = 0.0
block_size = 128

batch_size                  = 32
gradient_accumulation_steps = 1
max_iters                   = 3000
lr_decay_iters              = 3000
learning_rate               = 1e-3
min_lr                      = 1e-4
beta2                       = 0.99
warmup_iters                = 100

eval_interval          = 100
eval_iters             = 50
log_interval           = 10
always_save_checkpoint = False

device  = 'cpu'
compile = False

wandb_log      = False
wandb_project  = 'addition-experiments-new'
wandb_run_name = 'twoDigit-scratch'
