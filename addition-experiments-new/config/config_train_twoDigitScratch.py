# config/config_train_twoDigitScratch.py
#
# Trains the 2-digit scratch-space model, initialized from the trained
# 2-digit CoT checkpoint.
#
# The CoT model already knows column arithmetic — it currently writes
# that computation out explicitly as "7+6=13;4+2+1=7;".  This experiment
# asks: can it learn to do that same computation silently, using K=16
# semicolon positions as thinking time, without writing the steps out?
#
# Key differences from config_train_twoDigitCoT.py:
#   dataset       = twoDigitScratch    <- semicolon-padded format, no steps
#   out_dir       = out/twoDigitScratch
#   init_from     = 'resume'           <- start from CoT checkpoint (NOT random)
#   max_iters     = 6000               <- 2x longer: more time to adapt
#   lr_decay_iters= 6000
#   learning_rate = 1e-4               <- 10x lower: gentle fine-tuning
#   min_lr        = 1e-5
#
# REQUIRED before running:
#   mkdir -p out/twoDigitScratch
#   cp out/twoDigitCoT/ckpt.pt out/twoDigitScratch/ckpt.pt
#
# Then run from addition-experiments-new/:
#   NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
#   python -u ../../comp560-nanoGPT/train.py \
#   config/config_train_twoDigitScratch.py
#
# SUCCESS SIGNAL: the very first loss printed should be ~0.42 (the CoT
# model's trained loss), NOT ~9.0 (random init).  If you see ~9.0,
# the checkpoint copy did not work — stop and re-copy before continuing.

dataset = 'twoDigitScratch'
out_dir = 'out/twoDigitScratch'

init_from = 'resume'   # load from out/twoDigitScratch/ckpt.pt (copied from CoT)
reset_optimizer = True # discard CoT optimizer state — start Adam fresh on scratch data

n_layer    = 4
n_head     = 4
n_embd     = 128
dropout    = 0.0
block_size = 128

batch_size                  = 32
gradient_accumulation_steps = 1
max_iters                   = 6000
lr_decay_iters              = 6000
learning_rate               = 1e-4
min_lr                      = 1e-5
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
