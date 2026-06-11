# LG-DCD

This repository contains the LG-DCD implementation for privacy-preserving
dynamic community detection. It keeps only the code and data needed to run
LG-DCD.

## Layout

- `src/`
  - `LGDCD.py`: main LG-DCD implementation.
  - `LGw.py`: LG-DCD variant for sliding-window sensitivity experiments.
  - `utils.py`: graph loading and shared numerical utilities.
- `scripts/`
  - `precompute_louvain.py`: precompute snapshot-0 Louvain references.
  - `precompute_louvain_all.py`: precompute Louvain references for all snapshots.
  - `test_LGDCD.py`: lightweight smoke test.
- `data/`: dynamic graph snapshots.
- `precomputed_louvain/`: cached non-private Louvain reference partitions.

## Requirements

Install dependencies in a Python environment:

```bash
pip install -r requirements.txt
```

## Run

Run commands from the repository root.

```bash
python src/LGDCD.py
```

By default, this runs:

- dataset: `Forum`
- privacy budget: `eps=2.0`
- sliding window: `w=5`
- repeated runs: `exp_num=5`

### Select Datasets

Use `--datasets` to choose one or more datasets.

Available names:

- `EmailDept1`
- `Forum`
- `Tech_AS`
- `MathOverflow_a2q`
- `all`

Examples:

```bash
python src/LGDCD.py --datasets Forum
python src/LGDCD.py --datasets EmailDept1,Forum,Tech_AS
python src/LGDCD.py --datasets all
```

### Select Privacy Budgets

Use `--eps` to set one or more total window privacy budgets.

Examples:

```bash
python src/LGDCD.py --eps 1.0
python src/LGDCD.py --eps 1.0,2.0,4.0
```

### Select Sliding Window Sizes

Use `--windows` to set one or more sliding window sizes `w`.

Examples:

```bash
python src/LGDCD.py --windows 5
python src/LGDCD.py --windows 1,3,5,7,9
```

### Select Repeated Runs

Use `--exp-num` to set the number of repeated runs per setting.

Examples:

```bash
python src/LGDCD.py --exp-num 1
python src/LGDCD.py --exp-num 10
```

### Common Experiment Commands

Run the default paper-style budget sweep on Forum:

```bash
python src/LGDCD.py --datasets Forum --eps 1.0,2.0,4.0 --windows 5 --exp-num 10
```

Run the window-size sensitivity experiment:

```bash
python src/LGDCD.py --datasets Forum --eps 2.0 --windows 1,3,5,7,9 --exp-num 10
```

Run all datasets with the default setting:

```bash
python src/LGDCD.py --datasets all --eps 2.0 --windows 5 --exp-num 10
```

Quick debug run:

```bash
set LGDCD_MAX_SNAPSHOTS=2
set LGDCD_FAST_EVAL=1
python src/LGDCD.py --datasets Forum --eps 1.0 --windows 5 --exp-num 1
```

For a short smoke test:

```bash
python scripts/test_LGDCD.py
```

For faster evaluation on first run, use the existing Louvain cache in
`precomputed_louvain/`. To regenerate it:

```bash
python scripts/precompute_louvain_all.py
```

Useful environment variables:

- `LGDCD_MAX_SNAPSHOTS`: limit the number of snapshots for quick runs.
- `LGDCD_FAST_EVAL=1`: skip metric evaluation for speed.
- `LGDCD_USE_DENSE=1`: use dense matrix loading instead of sparse loading.
- `LGDCD_T_EPOCH`: set the periodic initialization interval.
- `LGDCD_EPS_INIT_FRAC`: set the initialization budget fraction.

On PowerShell, use `$env:NAME="value"` instead of `set NAME=value`, for example:

```powershell
$env:LGDCD_MAX_SNAPSHOTS="2"
$env:LGDCD_FAST_EVAL="1"
python src/LGDCD.py --datasets Forum --eps 1.0 --windows 5 --exp-num 1
```
