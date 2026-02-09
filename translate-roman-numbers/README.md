# Roman Number Translation Experiments

This project explores whether a small character-level language model can learn structured number translations starting from Roman numerals. Multiple datasets of increasing complexity were used to test how well the model scales from simple bilingual translation to multilingual and multi-format outputs.

## Experiments Overview

Three datasets were used:

- Basic (English) – Roman → English

- Spanish – Roman → Spanish

- Advanced (Multilingual & Multiformat) – Roman → English, Spanish, English numerals, Devanagari numerals, and Nepali words

Each experiment was trained for 200 iterations and 2000 iterations.

## Data Format

### 1. Basic (English)
   
I one

II two

III three

IV four

...

XX twenty


### 2. Spanish
   
I uno

II dos

III tres

IV cuatro

...

XX veinte


### 3. Advanced (Multilingual)

Each line contains multiple representations of the same number:

I one uno 1 १ एक

II two dos 2 २ दुई

III three tres 3 ३ तीन

IV four cuatro 4 ४ चार

...

XX twenty veinte 20 २० बीस

## Directory Structure

```text
translate-roman-numbers/
├── README.md
├── config/
│   ├── basic.py
│   ├── spanish.py
│   └── advanced.py
├── data/
│   ├── basic/
│   │   ├── prepare.py
│   │   ├── train.bin
│   │   ├── val.bin
│   │   └── meta.pkl
│   ├── spanish/
│   │   ├── prepare.py
│   │   ├── train.bin
│   │   ├── val.bin
│   │   └── meta.pkl
│   └── advanced/
│       ├── prepare.py
│       ├── train.bin
│       ├── val.bin
│       └── meta.pkl
├── out/
```



## Setup

- Create and activate a Python virtual environment:

``` python -m venv venv
source venv/bin/activate
```

- Prepare a dataset (example: basic):
```
cd translate-roman-numbers
python data/basic/prepare.py
```

- Repeat for spanish and advanced as needed.

- Training:

  ```NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
  python -u ../../comp560-nanoGPT/train.py config/basic.py
  ```
  
  - Replace basic.py with spanish.py or advanced.py for other experiments.

- Sampling
```
NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
python -u ../../comp560-nanoGPT/sample.py config/basic.py \
--num_samples=1 --max_new_tokens=300 --seed=1337
```

## Experiment Logs

### Basic (English)

#### Run 1: max_iters = 200

#### Results:

- The model learned line structure and repetition

- Many translations were incomplete or truncated

- Not very accurate

#### Run 2: max_iters = 2000

#### Results:

- Consistently and almost always perfectly correct Roman → English translations

- Strong pattern repetition

### Spanish
#### Run 1: max_iters = 200

#### Purpose: Verify multilingual learning feasibility

#### Results:

- Correct structure learned

- Spelling errors and partial words


#### Run 2: max_iters = 2000

#### Results:

- Accurate Roman → Spanish translations

- Accents and longer words learned surprisingly well (e.g., dieciséis)

- Comparable performance to English despite higher complexity

### Advanced (Multilingual)
#### Run 1: max_iters = 200

#### Results:

- Inconsistent patterns and structure

- Many incomplete or mixed-language outputs


#### Run 2: max_iters = 2000

#### Results:

- Significant improvement across all representations

- Correct alignment between Roman numerals and multiple formats

- Some minor inconsistencies, but strong overall learning
  
- Almost perfect accuracy

- Demonstrates the model’s ability to handle multilingual and numeric mappings simultaneously

## Weights & Biases (WandB)

Training loss curves for all experiments were logged using WandB.

### Observations:

- Loss decreased smoothly as the number of iterations increased.

- Increasing iterations consistently improved results across all datasets
  
- Loss was near zero at 2000 iterations.


## Conclusion

- The model almost perfectly learnt the basic dataset even when trained at 200 iterations, suggesting that easier patterns could be learnt with fewer iterations.

- The model performed really badly when trained at 200 iterations for the advanced dataset.

- Increasing training iterations from 200 → 2000 significantly improved accuracy for the advanced dataset.

- Character-level modeling successfully handled multiple languages and scripts

- CPU training was sufficient for all experiments

#### Key Takeaways

- Advanced multilingual translation is achievable even with a small GPT

- Structured symbolic data is well-suited for character-level transformers
