# config/train_threeDigitCoT.py
# Run from addition-experiments-new/ directory:
#   NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/train.py config/train_threeDigitCoT.py

# ── dataset & output ──────────────────────────────────────────────────────────
dataset = 'threeDigitCoT'
out_dir = 'out/threeDigitCoT'

# ── model architecture ────────────────────────────────────────────────────────
n_layer    = 4
n_head     = 4
n_embd     = 128
dropout    = 0.0
block_size = 128

# ── training ──────────────────────────────────────────────────────────────────
batch_size                  = 32
gradient_accumulation_steps = 1
max_iters                   = 3000
lr_decay_iters              = 3000
learning_rate               = 1e-3
min_lr                      = 1e-4
beta2                       = 0.99
warmup_iters                = 100

# ── evaluation & logging ──────────────────────────────────────────────────────
eval_interval          = 100
eval_iters             = 50
log_interval           = 10
always_save_checkpoint = False

# ── system ────────────────────────────────────────────────────────────────────
device  = 'cpu'
compile = False

# ── wandb ─────────────────────────────────────────────────────────────────────
wandb_log      = False
wandb_project  = 'addition-experiments'
wandb_run_name = 'threeDigit-cot'
