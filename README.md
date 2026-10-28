# RaDiEC

MATLAB implementation of RaDiEC for ensemble clustering.

## Requirements

- MATLAB with Statistics and Machine Learning Toolbox.
- Parallel Computing Toolbox for the provided `parfor` experiment drivers.

## Files

- `alg/`: the solver and its required helper functions.
- `metrics/`: clustering evaluation metrics.
- `utils/initialRandom.m`: base-clustering sampling.
- `compare.m`: run the method on a single dataset.
- `compareAll.m`: run the method on all datasets.

## Data

The `datasets/` folder includes the 50 paper datasets. See
[datasets/README.md](datasets/README.md) for the D01–D50 appendix mapping.
Each MAT file must contain a numeric
`members` matrix with one row per sample and one column per base clustering,
and a ground-truth vector `gt` with the same number of rows. Labels within
each base clustering must be consecutive positive integers starting at 1.
The provided experiment drivers expect at least 100 base clusterings.
Ground truth supplies the target cluster count and evaluation labels.

## Run

Open MATLAB in this repository directory. For a single dataset, edit
`dataName` in `compare.m`, then run:

```matlab
compare
```

For all MAT files under `datasets/`, run:

```matlab
compareAll
```

The solver can also be called directly:

```matlab
addpath('alg');
rng(2026, 'twister');
[labels, U, w, history] = RaDiEC(clusterings, k);
```

Here `clusterings` contains the selected base clusterings and `k` is the
target number of clusters. `U`, `w`, and `history` contain the embedding,
cluster weights, and optimization history.

The 50 dataset MAT files are included. Generated experiment results are not included.

Additional large-scale datasets are available under `largeDatasets/`.
`YahooAnswersTopics.mat` uses Git LFS; run `git lfs pull` after cloning to
download its full contents.
