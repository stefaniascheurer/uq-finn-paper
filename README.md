# UQ for FINN (Efficient Confidence Interval Computation for Physics-Aware ML)
Supplementary code to replicate key experiments from our paper.

## Installation
Create a new environment with Python 3.13. Install the required dependencies by running:
```bash
pip install .
```

## Reproducing Results
To reproduce the results presented, run notebooks:
```bash
run-uq.ipnyb
plots_results.ipynb
plots_runtime.ipynb
```
The configuration files are provided in the `configs.py` files for each component, i.e. each subfolder (finn, pi3nn, ddb).
