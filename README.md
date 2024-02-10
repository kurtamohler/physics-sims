# physics-sims

These are my notes and simulations of physics topics that I'm learning about. I
do not have a physics degree and I am not an expert on the subject, so these
notes should be taken with a high degree of skepticism.

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

## Run interactive special relativity simulation

```bash
python interactive/einsteinian.py
```

Use the left and right arrow keys to accelerate in those directions.

## Run notebooks

```bash
cd notebooks
jupyter notebooks
```
