# A3 -- Rank Fusion + LTR Arena

Fuse four base rankers, then train pointwise / RankNet / LambdaMART rankers on MSLR-WEB10K Fold1 without overfitting. Full spec: the assignment PDF. Held-out data is **not** here and must not be sought (honor code).

## Quick start
```
pip install -r requirements.txt
bash scripts/smoke_test.sh                 # toy data, trains, runs the harness, runs tests
python scripts/prepare_mslr.py --fold-dir <MSLR-WEB10K/Fold1>     # real data -> data/train, data/dev
python scripts/run_recipe.py                      # Track C1 dry run on toy data (synthetic transfer sets; see docs/TRANSFER_TRACK.md)
python scripts/cv_lambdamart.py --train data/train --trees 50 100 200 400   # local train-vs-validation gap
python scripts/train_model.py --train data/train --out submission/model.txt --trees 200
python -m harness.run_harness --features data/dev/vali.txt --model submission/model.txt
```

## Implemented components
| file | what |
|---|---|
| `submission/fusion.py` | Hand-written RRF, CombSUM, CombMNZ, normalization, and weighted CombMNZ implementations |
| `submission/pointwise_pairwise.py` | Hand-written pointwise and RankNet losses, within-query pair sampling, training steps, and the MLP training loop |
| `submission/ranking.py` | Track A fusion and LambdaMART reranking entry, including the approved multi-model ensemble |
| `scripts/train_model.py` | your reproducible LambdaMART training script |
| `submission/recipe.py` | dimension-agnostic training recipe for the undisclosed C1 transfer benchmarks |

Self-check the hand-written components with:

```bash
python -m pytest tests/test_student_todos.py
python -m pytest -q
python scripts/run_recipe.py
```

Docs: `docs/SUBMISSION_INTERFACE.md`, `docs/GRADING.md`, `docs/RANKNET_PRIMER.md`, `docs/DATA.md`, `docs/TRANSFER_TRACK.md`, `docs/SUBMISSION_NOTES.md` (fill in and keep as your README section).

## Fusion method

`fuse_candidates()` uses weighted CombMNZ. For each candidate pool, every
base-ranker channel is min-max normalized independently. The normalized
channel scores are summed using fixed channel weights, and the sum is
multiplied by the number of channels whose normalized score is positive.
The active channel order is:

```text
bm25_whole_document, bm25_title, lmir_dir_whole_document, lmir_jm_whole_document
```

The active weights are:

```text
0.50, 0.25, 0.25, 0.00
```

The same module also contains hand-written RRF and CombSUM implementations.
No rank-fusion library is used.

## Feature engineering

The original MSLR-WEB10K input has 136 features. The submitted LambdaMART
models use these feature sets:

| artifact | features |
|---|---:|
| `model.txt` | 277: 136 original + 136 query-local percentile ranks + 5 ranker-agreement features |
| `model_rank_all.txt` | 272: 136 original + 136 query-local percentile ranks |
| `model_raw.txt` | 136 original features |

The derived features are computed separately for each query and use only the
candidate feature values; they never use relevance labels. The agreement
features summarize whether a document is near the top of the four named
base-ranker channels and include mean, maximum, and minimum channel rank.
`ranking.py` reconstructs these features at inference time from the supplied
136-dimensional vectors.

## LambdaMART hyperparameters

LambdaMART is implemented using LightGBM's permitted `lambdarank` objective.
The submitted artifacts were trained on the released MSLR TRAIN split. Model
selection used grouped query-level cross-validation within TRAIN rather than
splitting individual rows.

| artifact | trees | leaves | feature mode |
|---|---:|---:|---|
| `model.txt` | 600 | 31 | `rank_all_agreement` |
| `model_rank_all.txt` | 800 | 31 | `rank_all` |
| `model_raw.txt` | 600 | 15 | `none` |

The training script also supports learning rate, minimum data per leaf,
L2 regularization, feature fraction, bagging fraction, random seed, and CPU
thread count. These were varied during grouped TRAIN cross-validation to
control overfitting. The exact training-data, tuning, and early-stopping
declarations are in [`docs/SUBMISSION_NOTES.md`](docs/SUBMISSION_NOTES.md).

The normal MSLR entry combines the three LambdaMART predictions after
per-query z-score normalization with default weights `0.45 / 0.25 / 0.30`.
If the companion artifacts are unavailable in the isolated C1 environment,
`ranking.py` falls back to the generated primary model only.

## Layout
`harness/` metrics, I/O, CV, leaderboard formulas, runner (do not edit; staff run an identical copy) - `submission/` your entry - `scripts/` tooling - `data/toy/` and `data/toy_transfer/` synthetic data - `tests/`.

The shipped model artifacts are trained offline; grading loads them and does
not retrain them. The hidden held-out labels are not included and must not be
accessed.
