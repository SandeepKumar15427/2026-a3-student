# Fusion experiments on `data/train`

These experiments evaluated 6,000 training queries. The proxy shown below is
`0.8 * nDCG@10 + 0.2 * MAP@10`, matching Track A's 80/20 metric weighting.
Official leaderboard scoring percentile-ranks nDCG and MAP separately.

Channel order for all weight tuples:

```text
(bm25_whole_document, bm25_title, lmir_dir_whole_document, lmir_jm_whole_document)
```

## RRF and unweighted baselines

| Method | nDCG@10 | MAP@10 | Proxy |
|---|---:|---:|---:|
| RRF c=1 | 0.284300 | 0.114328 | 0.250305 |
| RRF c=5 | 0.287499 | 0.116639 | 0.253327 |
| RRF c=10 | 0.288957 | 0.117657 | 0.254697 |
| RRF c=20 | 0.290905 | 0.118695 | 0.256463 |
| RRF c=30 | 0.291366 | 0.119012 | 0.256895 |
| RRF c=60 | 0.292569 | 0.119462 | 0.257948 |
| RRF c=100 | 0.293192 | 0.119690 | 0.258492 |
| RRF c=200 | 0.293675 | 0.119857 | 0.258911 |
| CombSUM | 0.296150 | 0.121275 | 0.261175 |
| CombMNZ | 0.295797 | 0.120914 | 0.260821 |

## Best weighted configurations

The experiment evaluated 35 nonnegative weight tuples in increments of 0.25
that sum to 1, for both weighted CombSUM and weighted CombMNZ.

| Rank | Method | nDCG@10 | MAP@10 | Proxy |
|---:|---|---:|---:|---:|
| 1 | Weighted CombSUM `(0.25, 0.25, 0.50, 0.00)` | 0.299636 | 0.123018 | 0.264312 |
| 2 | Weighted CombSUM `(0.50, 0.25, 0.25, 0.00)` | 0.299320 | 0.123339 | 0.264124 |
| 3 | Weighted CombMNZ `(0.25, 0.25, 0.50, 0.00)` | 0.299411 | 0.122767 | 0.264082 |
| 4 | Weighted CombMNZ `(0.50, 0.25, 0.25, 0.00)` | 0.299215 | 0.123153 | 0.264003 |
| 5 | Weighted CombSUM `(0.25, 0.50, 0.25, 0.00)` | 0.297124 | 0.120540 | 0.261807 |
| 6 | Weighted CombMNZ `(0.25, 0.50, 0.25, 0.00)` | 0.297056 | 0.120551 | 0.261755 |
| 7 | Weighted CombMNZ `(0.00, 0.25, 0.75, 0.00)` | 0.296654 | 0.120471 | 0.261417 |
| 8 | Weighted CombSUM `(0.25, 0.25, 0.25, 0.25)` | 0.296150 | 0.121275 | 0.261175 |
| 9 | CombSUM | 0.296150 | 0.121275 | 0.261175 |
| 10 | Weighted CombMNZ `(0.00, 0.50, 0.50, 0.00)` | 0.296515 | 0.119696 | 0.261151 |

## Best training-set configuration

```text
Weighted CombSUM
bm25_whole_document       = 0.25
bm25_title                = 0.25
lmir_dir_whole_document  = 0.50
lmir_jm_whole_document   = 0.00
```

This was selected on TRAIN only. It should be evaluated on `data/dev/vali.txt`
before changing `fuse_candidates()`, because a training-set winner can
overfit. No source files were changed by this experiment.

## Evaluation on DEV

The same 80 configurations were evaluated on the 2,000-query dev split. The
TRAIN-selected configuration performed as follows:

```text
Weighted CombSUM (0.25, 0.25, 0.50, 0.00)
nDCG@10 = 0.307785
MAP@10  = 0.128917
proxy   = 0.272011
```

The best DEV configuration was:

```text
Weighted CombMNZ (0.50, 0.25, 0.25, 0.00)
nDCG@10 = 0.310084
MAP@10  = 0.130327
proxy   = 0.274132
```

Top DEV configurations:

| Rank | Method | nDCG@10 | MAP@10 | Proxy |
|---:|---|---:|---:|---:|
| 1 | Weighted CombMNZ `(0.50, 0.25, 0.25, 0.00)` | 0.310084 | 0.130327 | 0.274132 |
| 2 | Weighted CombSUM `(0.50, 0.25, 0.25, 0.00)` | 0.309813 | 0.130188 | 0.273888 |
| 3 | Weighted CombMNZ `(0.25, 0.25, 0.50, 0.00)` | 0.308590 | 0.129371 | 0.272746 |
| 4 | Weighted CombSUM `(0.25, 0.25, 0.50, 0.00)` | 0.307785 | 0.128917 | 0.272011 |
| 5 | Weighted CombMNZ `(0.25, 0.50, 0.25, 0.00)` | 0.307136 | 0.127302 | 0.271170 |

The TRAIN winner remains strong on DEV, but the DEV winner is a different
weighting. Since selecting directly on DEV counts as dev tuning, use the
TRAIN-selected configuration for a clean generalization choice, or declare
DEV tuning if you choose the DEV winner for the submission.
