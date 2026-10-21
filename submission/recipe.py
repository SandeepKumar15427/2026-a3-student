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

The recipe below adapts model capacity to the input shape, uses the supplied validation
queries for early stopping, and finally refits using all labelled train+validation
queries. It deliberately does not assume MSLR's 136 features or feature meanings.

Test it locally on the synthetic transfer sets:  python scripts/run_recipe.py
"""
import numpy as np

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
    Xv, yv, qv = stack(read_svmlight_ltr(vali_path))

    n_rows, n_features = X.shape
    # Capacity rules depend only on generic dataset properties.  They keep
    # narrow/small transfer sets from being over-parameterised while allowing
    # wider datasets enough tree capacity.
    if n_features <= 60:
        num_leaves = 7
    elif n_features <= 150:
        num_leaves = 15
    else:
        num_leaves = 31
    min_data_in_leaf = max(20, min(200, n_rows // 2000))
    feature_fraction = 0.8 if n_features > 100 else 1.0

    params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [10],
        "num_leaves": num_leaves, "learning_rate": 0.03,
        "min_data_in_leaf": min_data_in_leaf, "lambda_l2": 1.0,
        "feature_fraction": feature_fraction, "seed": seed,
        "deterministic": True, "force_row_wise": True, "verbose": -1, "num_threads": 4,
    }
    train_set = lgb.Dataset(X, label=y, group=_group_sizes(q))
    vali_set = lgb.Dataset(Xv, label=yv, group=_group_sizes(qv), reference=train_set)

    # Let each benchmark choose its effective tree count from its own
    # validation curve instead of imposing an MSLR-specific constant.
    probe = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[vali_set],
        valid_names=["vali"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    best_rounds = max(1, int(probe.best_iteration or 100))

    # Use every labelled query available to the recipe after selecting the
    # stopping point. The held-out benchmark queries remain unseen.
    X_all = np.vstack([X, Xv])
    y_all = np.concatenate([y, yv])
    q_all = list(q) + list(qv)
    final_set = lgb.Dataset(X_all, label=y_all, group=_group_sizes(q_all))
    final = lgb.train(params, final_set, num_boost_round=best_rounds)
    final.save_model(out_model_path)
