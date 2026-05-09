# UQ for FINN (Efficient Confidence Interval Computation for Physics-Aware ML)
Supplementary code to replicate key experiments from our [paper](https://www.frontiersin.org/journals/water/articles/10.3389/frwa.2026.1813791/full).

## Installation
Create a new environment with Python 3.13. Install the required dependencies by running:
```bash
pip install .
```

## Reproducing Results
To reproduce the results presented, run notebooks:
```bash
run_uq.ipnyb
plots_results.ipynb
plots_runtime.ipynb
```
and
```bash
run_uq_synth.ipnyb
plots_results_synth.ipynb
```

The configuration files are provided in the `config.py` files for each component, i.e. each subfolder (finn, pi3nn, ddb).
