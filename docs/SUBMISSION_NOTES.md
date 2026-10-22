# Submission notes

## Training-data declaration

| model | trained on | tuned on | early-stopped on |
|---|---|---|---|
| LambdaMART `model.txt` | released MSLR TRAIN only | grouped cross-validation within TRAIN | none; fixed tree count |
| LambdaMART `model_rank_all.txt` | released MSLR TRAIN only | grouped cross-validation within TRAIN | none; fixed tree count |
| LambdaMART `model_raw.txt` | released MSLR TRAIN only | grouped cross-validation within TRAIN | none; fixed tree count |
| `pointwise_model.pt` | released MSLR TRAIN only | released MSLR DEV | released MSLR DEV |
| `pairwise_model.pt` | released MSLR TRAIN only | released MSLR DEV | released MSLR DEV |
| `recipe.py` | benchmark-specific train and validation files supplied by staff | benchmark validation data for stopping; generic rules based on observed dimensions | benchmark validation data |

The hidden MSLR held-out labels and the hidden C1 benchmark labels were not
used for training, feature selection, hyperparameter selection, or debugging.

## Fusion method and why

`fuse_candidates()` uses hand-written weighted CombMNZ. Each channel is
min-max normalized over the current candidate pool; the normalized channel
scores are combined with fixed channel weights, then multiplied by the number
of channels whose normalized score is positive. The implementation also keeps
hand-written RRF, CombSUM, and CombMNZ functions in `submission/fusion.py`.

The active channel order is:

```text
bm25_whole_document, bm25_title, lmir_dir_whole_document, lmir_jm_whole_document
```

The active weights are:

```text
0.50, 0.25, 0.25, 0.00
```

These weights were selected during local experiments on the released data.
The final report should distinguish any TRAIN-only selection from later DEV
evaluation; DEV results are not treated as evidence of held-out
generalization.

## LambdaMART hyperparameters and feature engineering

The primary submitted model uses 277 features: the original 136 MSLR
features, 136 query-local percentile-rank features, and five query-local
ranker-agreement features. The 272-feature companion uses the original 136
features plus the 136 query-local percentile-rank features. The raw companion
uses the original 136 features.

The submitted artifacts use LightGBM's permitted `lambdarank` objective. The
final primary artifact has 600 trees and 31 leaves per tree. The companion
artifacts have 800 trees/31 leaves and 600 trees/15 leaves respectively. The
training scripts expose and record learning rate, minimum data per leaf,
L2 regularization, feature fraction, bagging fraction, seed, and thread
count; grouped TRAIN cross-validation was used to compare these settings.

Feature engineering is query-local and uses no relevance labels. For each
query, ranks and agreement summaries are computed only from the candidate
feature rows. At inference time, `ranking.py` reconstructs the same derived
features from the supplied 136-dimensional vectors.

The ensemble used by the normal MSLR submission combines the 277-, 272-, and
136-feature model predictions after per-query z-score normalization. Its
default weights are `0.45 / 0.25 / 0.30`; these were selected using grouped
out-of-fold TRAIN predictions. If companion artifacts are unavailable, as in
the isolated C1 recipe run, the code falls back to the generated primary
model.

## Neural models

Both neural artifacts use the same plain MLP scorer. The pointwise model uses
the manually implemented squared-error loss. The pairwise model uses the
manually implemented RankNet logistic loss with an additional separate
per-pair relevance-gap weighting function. Pairs are formed only within a
query, ties are excluded, and training uses relevance labels only. No
LambdaMART-score distillation is used.

## Final configuration and generalization trade-off

The LambdaMART models were trained on TRAIN rather than DEV to preserve a
cleaner estimate of performance on unseen queries. Grouped query-level
cross-validation was used because splitting individual rows would leak
documents from the same query across folds. Feature fraction, bagging,
minimum leaf size, L2 regularization, and tree capacity were used to control
overfitting. DEV was used for neural early stopping and public diagnostics;
this is declared above and makes the public DEV scores optimistic.

The C1 recipe is dimension-agnostic: it reads arbitrary flat-feature
SVMLight input, selects generic capacity rules from the observed dimensions,
uses the supplied validation queries for stopping, refits on train plus
validation, and writes a model loadable by `ranking.py`.

## AI-use disclosure

AI was used as an assistant for explaining the assignment,
reviewing implementation choices, suggesting experiments, debugging code,
and drafting documentation.

## Code provenance

The implementation uses LightGBM and PyTorch as permitted standard libraries,
NumPy, and the starter repository's harness utilities. The fusion formulas,
pointwise loss, pair construction, RankNet loss, and training steps are
implemented in this repository rather than delegated to a ranking library.
The RankNet formulation and interface requirements were based on the
assignment PDF and the supplied repository documentation.
