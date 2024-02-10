# physics-sims

Physics simulations based on Lagrangian and Hamiltonian mechanics with constant
energy integration

## Installation

Install Miniconda:
[instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

Clone the repository.

```bash
git clone https://github.com/kurtamohler/physics-sims.git && cd physics-sims
```

Run the following to create and activate an environment with all dependencies.

```bash
conda env create -f environment.yaml -n physics-sims && conda activate physics-sims
```


```bash
pip install -e .
```

## Run notebooks

```bash
cd notebooks
jupyter notebooks
```
