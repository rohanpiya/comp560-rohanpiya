# config/config_train_mixedCoT.py
#
# Trains a single CoT model on mixed 2, 3, and 4-digit addition.
#
# MOTIVATION
# ----------
# Individual per-digit CoT models learn a "rigid template" — they memorize
# that N-digit problems always have exactly N steps, rather than learning to
# count columns and derive the step count from the input.  This causes
# complete OOD failure (0%) when evaluated on a different digit count.
#
# By training on mixed digit counts, the model is forced to learn variable-
# length CoT: a 2-digit problem needs 2 steps, a 3-digit needs 3 steps, etc.
# The hypothesis is that this teaches genuine column-counting, which may
# allow the model to generalize to 5-digit inputs it has never seen.
#
# KEY DIFFERENCE FROM PER-DIGIT CONFIGS
# --------------------------------------
# dataset   = mixedCoT       <- 2+3+4-digit problems in one dataset
# max_iters = 5000           <- slightly more iterations for the harder task
# lr_decay_iters = 5000
# (everything else identical to per-digit CoT configs)
#
# Run from addition-experiments-new/:
#   NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
#   python -u ../../comp560-nanoGPT/train.py \
#   config/config_train_mixedCoT.py

dataset = 'mixedCoT'
out_dir = 'out/mixedCoT'

init_from = 'scratch'   # train from random init — no pretrained weights needed

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
wandb_run_name = 'mixed-cot'
