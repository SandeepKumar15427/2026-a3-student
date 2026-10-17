"""
submission/recipe.py -- your training recipe (assignment Section 4.1, Track C1).

Staff call train() ONCE per hidden benchmark (two undisclosed flat-feature LTR datasets) on that
benchmark's train and vali files, in an isolated directory, then evaluate the model it writes on the benchmark's
held-out split through YOUR prepare() / rerank_with_ltr(). Contract:

  * signature: train(train_path, vali_path, out_model_path, seed=0) -> None
  * the files are flat-feature SVMLight (label qid:N 1:v ... D:v), D unknown (at most ~300), labels integers
    0..4 (possibly fewer levels), at most ~500k training rows; feature meanings are NOT given
  * write a LightGBM text model (or whatever your prepare() loads) to out_model_path
  * 20 minutes wall-clock PER BENCHMARK on 4 CPU threads; no network; read only the two input files; deterministic given seed
  * a recipe that raises, times out, or writes a model prepare() cannot load scores 0 on that benchmark

The provided default is a fixed-hyperparameter LambdaMART. Replace the hyperparameters with choices that
transfer: the point of Track C1 is that settings picked by cross-validation on MSLR TRAIN should carry over to
a different dataset -- e.g. express tree count / leaf size / min_data_in_leaf as functions of the number of
queries and features rather than hard-coding numbers tuned on MSLR, or run a small CV inside the time limit.

Test it locally on the synthetic transfer sets:  python scripts/run_recipe.py
"""
from harness.feature_io import read_svmlight_ltr, stack


def _group_sizes(qids):
    sizes, last = [], object()
    for q in qids:
        if q == last:
            sizes[-1] += 1
        else:
            sizes.append(1)
            last = q
    return sizes


def train(train_path: str, vali_path: str, out_model_path: str, seed: int = 0) -> None:
    import lightgbm as lgb

    X, y, q = stack(read_svmlight_ltr(train_path))
    params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [10],
        "num_leaves": 15, "learning_rate": 0.05, "min_data_in_leaf": 50, "lambda_l2": 1.0,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": seed,
        "deterministic": True, "force_row_wise": True, "verbose": -1, "num_threads": 4,
    }
    booster = lgb.train(params, lgb.Dataset(X, label=y, group=_group_sizes(q)), num_boost_round=100)
    booster.save_model(out_model_path)
