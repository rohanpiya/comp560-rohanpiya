Character-Level Symbol Learning Experiments
Overview

This project explores how a character-level language model learns structure and symbol roles from text.
I ran a series of small experiments to test whether the model could learn how commas, semicolons, and dollar signs behave in structured data, using only pattern recognition and limited compute on a laptop.

All experiments use a character-level GPT-style model trained on custom synthetic datasets.

Experiments
Experiment 1: Basic Pattern

Goal:
Test whether the model can learn a very simple and fixed pattern.

Data:
Each line follows this format:

abc,def;


Letters limited to a–e

Fixed length

Comma separates tokens

Semicolon ends a line

Result:
The model learned the pattern very quickly. Results at 200 and 2000 iterations were very similar, showing the task was easy for the model.

Experiment 2: Names

Goal:
Test whether the model can still learn structure with more realistic data.

Data:

Real names

Variable length

Arbitrary number of commas per line

Semicolon still marks the end of a line

Example:

rohan,piya;
peter,spider,parker;


Result:
The model did not always generate correct names, but it consistently placed semicolons before newlines. This shows it learned structural rules even when content was noisy.

Experiment 3: Jobs (Comma–Dollar–Semicolon)

Goal:
Test whether the model can learn multiple symbol roles at once.

Data:

Commas separate name parts

Dollar sign separates name and job

Semicolon ends the line

Example:

rohan,piya$student;
john,maccormick$professor;


Result:
With fewer iterations, outputs were messy, but the model consistently predicted newlines after semicolons.
With more iterations (5000), the model began placing dollar signs and commas in more correct positions, showing improved understanding of symbol roles.