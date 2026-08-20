"""
Nested cross-validation training for the three settings of the paper.

All three share one core (`_search_and_refit`):

  inner loop   the training subjects are split into 5 subject-level folds; every
               point of the hyperparameter grid is scored by IPCW-weighted AUROC
               on the pooled out-of-fold validation predictions
  threshold    the decision threshold is the point of equal weighted sensitivity
               and specificity on those same pooled predictions
  refit        the winning hyperparameters are refitted on the full training set

The settings differ only in how the training frame and the test frames are built:

  within  10 outer subject folds of one dataset; train and test both from it
  joint   the 10 outer training folds of all three datasets are pooled; the
          model is evaluated on each dataset's held-out fold separately
  cross   trained on all of one dataset, evaluated on all of another; there is no
          outer fold, so the 10 repeats vary the seed instead

Class imbalance is handled by class_weight='balanced' in the forest, on top of
the IPCW sample weights.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

from config import OUTCOME_VARIABLE
from data import (build_dataset, create_subject_folds, drop_affected_at_baseline,
                  scale_dict, select_subjects, set_paradigm, subject_kfold)
from ipcw import fit_censoring_model, ipcw_weights
from metrics import evaluate, weighted_auc, weighted_confusion_metrics
from models import BetaCalibratedClassifierCV

# The hyperparameter grid the runs search: 54 points. Note this is not the grid
# tabulated in the manuscript's supplement, which lists wider ranges.
PARAM_GRID = {
    'n_estimators': [200, 400, 600],
    'max_depth': [None, 20],
    'max_features': ['sqrt', 0.75, 1.0],
    'min_samples_leaf': [5, 10, 15],
    'min_samples_split': [5],
}

THRESHOLD_GRID = np.linspace(0.01, 0.9, 100)
INNER_FOLDS = 5
OUTER_FOLDS = 10


def _usable_fields(config, frames):
    """Config feature order, restricted to columns present in every frame."""
    fields = list(config.X_fields)
    for df in frames:
        fields = [f for f in fields if f in df.columns]
    dropped = [f for f in config.X_fields if f not in fields]
    if dropped:
        print('Dropped fields (absent from the data):', dropped)
    print('X fields:', fields)
    return fields


def _align_columns(frames):
    """
    Give every frame the union of the frames' columns, filling absent ones with
    NaN, while keeping each frame's own rows. Used for the joint setting, where
    the cohorts do not all record the same features.
    """
    combined = pd.concat(frames, join='outer', ignore_index=True)
    aligned, start = [], 0
    for df in frames:
        end = start + len(df)
        aligned.append(combined.iloc[start:end].reset_index(drop=True))
        start = end
    return aligned


def _design(df, fields, to_scale, scaler):
    X = df[fields].values.astype(float)
    if scaler is not None:
        X[:, to_scale] = scaler.transform(X[:, to_scale])
    return X, df[OUTCOME_VARIABLE].values


def _new_model(params, fields, seed):
    forest = RandomForestClassifier(class_weight='balanced', random_state=seed, **params)
    return BetaCalibratedClassifierCV(base_estimator=forest, X_fields=fields,
                                      cv=5, seed=seed)


def _train_split(df, fields, to_scale, seed, time_horizon):
    """
    Fit the censoring model on `df`, drop its censored rows, and return the
    scaled design matrix, labels, IPCW weights and the fitted scaler + censoring
    model (the latter is reused for the matching validation / test rows).

    Order matters here: the censored rows are dropped BEFORE the weights are
    computed, so the weight vector lines up row-for-row with the design matrix.
    Computing the weights first yields a longer vector, and the estimator indexes
    it positionally without checking the length, so the mismatch is silent.
    """
    censoring_model, df = fit_censoring_model(df, time_horizon=time_horizon)
    df = df[df['CENSORED'] == 0]
    weights = ipcw_weights(df, censoring_model, time_horizon=time_horizon)

    X_raw = df[fields].values.astype(float)
    scaler = StandardScaler().fit(X_raw[:, to_scale]) if any(to_scale) else None
    X, y = _design(df, fields, to_scale, scaler)
    return X, y, weights, scaler, censoring_model


def _search_and_refit(config, df_train, fields, to_scale, time_horizon):
    """Inner-loop hyperparameter search, then refit on all of `df_train`."""
    subjects = df_train['USUBJID'].unique()
    train_folds, val_folds = subject_kfold(subjects, INNER_FOLDS, config.seed)

    best = {'score': -np.inf, 'params': None, 'threshold': None}
    grid = list(ParameterGrid(PARAM_GRID))

    # Visit selection, the Cox censoring fit, the IPCW weights and the scaling are
    # all independent of the hyperparameters, so they are computed once per inner
    # fold instead of once per (fold, grid point). Purely a speedup: nothing here
    # consumes the global RNG, and the estimators take an explicit random_state.
    inner_folds = []
    for train_subjects, val_subjects in zip(train_folds, val_folds):
        df_fit = set_paradigm(select_subjects(df_train, train_subjects), config.cutoff)
        df_val = set_paradigm(select_subjects(df_train, val_subjects), config.cutoff)

        X, y, w, scaler, censoring_model = _train_split(
            df_fit, fields, to_scale, config.seed, time_horizon)

        # Censored validation rows are kept but carry weight 0, so every
        # weighted metric ignores them.
        w_val = ipcw_weights(df_val, censoring_model, time_horizon=time_horizon)
        X_val, y_val = _design(df_val, fields, to_scale, scaler)

        inner_folds.append((X, y, w, X_val, y_val, w_val))

    for i, params in enumerate(grid, start=1):
        probs, truth, val_weights = [], [], []

        for X, y, w, X_val, y_val, w_val in inner_folds:
            model = _new_model(params, fields, config.seed).fit(X, y, sample_weight=w)
            probs.append(model.predict_proba(X_val)[:, 1])
            truth.append(y_val)
            val_weights.append(w_val)

        probs = np.concatenate(probs)
        truth = np.concatenate(truth)
        val_weights = np.concatenate(val_weights)

        gaps = []
        for threshold in THRESHOLD_GRID:
            m = weighted_confusion_metrics(truth, (probs >= threshold).astype(int), val_weights)
            gaps.append(abs(m['Sensitivity'] - m['Specificity']))
        threshold = THRESHOLD_GRID[int(np.argmin(gaps))]

        score = weighted_auc(truth, probs, val_weights)[0]
        if score > best['score']:
            best.update(score=score, params=params, threshold=threshold)

        if i % 10 == 0:
            print(f'seed {config.seed} parameter set {i}/{len(grid)}')

    print('Best params:', best['params'], '| threshold %.3f' % best['threshold'],
          '| val AUROC %.4f' % best['score'])

    df_fit = set_paradigm(df_train, config.cutoff)
    X, y, w, scaler, censoring_model = _train_split(
        df_fit, fields, to_scale, config.seed, time_horizon)
    model = _new_model(best['params'], fields, config.seed).fit(X, y, sample_weight=w)

    config.thresh = best['threshold']
    return model, scaler, censoring_model, best


def _test_arrays(df_test, config, fields, to_scale, scaler, censoring_model, time_horizon):
    df_test = set_paradigm(df_test, config.cutoff)
    df_test = df_test[df_test['CENSORED'] == 0]
    weights = ipcw_weights(df_test, censoring_model, time_horizon=time_horizon)
    X, y = _design(df_test, fields, to_scale, scaler)
    return (X, y), weights


# ---------------------------------------------------------------------------
# Setting 1: within-dataset
# ---------------------------------------------------------------------------
def run_within(config, fold, time_horizon=3.0):
    np.random.seed(config.seed)

    df, metadata = build_dataset(config, time_horizon=time_horizon)
    df = drop_affected_at_baseline(df, config.cutoff)

    folds = create_subject_folds(df['USUBJID'].unique().tolist(),
                                n_folds=OUTER_FOLDS, seed=config.seed)
    df_train = select_subjects(df, folds[fold]['train'])
    df_test = select_subjects(df, folds[fold]['test'])

    fields = _usable_fields(config, [df])
    to_scale = [scale_dict[f] for f in fields]

    model, scaler, censoring_model, _ = _search_and_refit(
        config, df_train, fields, to_scale, time_horizon)

    test_set, test_weights = _test_arrays(df_test, config, fields, to_scale,
                                          scaler, censoring_model, time_horizon)
    results, proba, truth = evaluate(test_set, model, config.thresh, test_weights)
    return {'results': results, 'model': model, 'scaler': scaler, 'metadata': metadata,
            'fields': fields, 'proba': proba, 'labels': truth,
            'testdata': test_set[0], 'weights': test_weights}


# ---------------------------------------------------------------------------
# Setting 2: joint-dataset
# ---------------------------------------------------------------------------
def run_joint(config_list, fold, time_horizon=3.0):
    np.random.seed(config_list[0].seed)
    config = config_list[0]

    train_parts, test_frames, metadata = [], [], None
    for cfg in config_list:
        df, metadata = build_dataset(cfg, time_horizon=time_horizon)
        df = drop_affected_at_baseline(df, cfg.cutoff)
        folds = create_subject_folds(df['USUBJID'].unique().tolist(),
                                    n_folds=OUTER_FOLDS, seed=config.seed)
        train_parts.append(select_subjects(df, folds[fold]['train']))
        test_frames.append(select_subjects(df, folds[fold]['test']))

    # NET-PD LS-1 does not record HY / Insomnia / Gait / Sleepiness / MOCA. The
    # outer join gives every cohort the union of columns, filling those five with
    # NaN for that cohort, so the joint model keeps the full 22-feature vector
    # and leans on the forest's native missing-value handling. The test frames
    # are concatenated the same way and sliced back, so they carry the same
    # columns as the training frame.
    df_train = pd.concat(train_parts, join='outer', ignore_index=True)
    test_frames = _align_columns(test_frames)

    # Interleave the three cohorts so the inner folds are not cohort-ordered.
    np.random.seed(config.seed)
    df_train = df_train.groupby('USUBJID').apply(lambda x: x).sample(frac=1).reset_index(drop=True)

    fields = _usable_fields(config, [df_train])
    to_scale = [scale_dict[f] for f in fields]

    model, scaler, censoring_model, _ = _search_and_refit(
        config, df_train, fields, to_scale, time_horizon)

    for cfg in config_list:
        cfg.thresh = config.thresh

    results_list, proba_list, truth_list, testdata_list, weights_list = [], [], [], [], []
    for cfg, df_test in zip(config_list, test_frames):
        # The joint model reuses the pooled training censoring model.
        test_set, weights = _test_arrays(df_test, cfg, fields, to_scale,
                                         scaler, censoring_model, time_horizon)
        results, proba, truth = evaluate(test_set, model, cfg.thresh, weights)
        results_list.append(results)
        proba_list.append(proba)
        truth_list.append(truth)
        testdata_list.append(test_set[0])
        weights_list.append(weights)

    return {'results': results_list, 'model': model, 'scaler': scaler, 'metadata': metadata,
            'fields': fields, 'proba': proba_list, 'labels': truth_list,
            'testdata': testdata_list, 'weights': weights_list}


# ---------------------------------------------------------------------------
# Setting 3: cross-dataset
# ---------------------------------------------------------------------------
def run_cross(config_train, config_test, time_horizon=3.0):
    np.random.seed(config_train.seed)

    df_train, metadata = build_dataset(config_train, time_horizon=time_horizon)
    df_train = drop_affected_at_baseline(df_train, config_train.cutoff)

    df_test, _ = build_dataset(config_test, time_horizon=time_horizon)
    df_test = drop_affected_at_baseline(df_test, config_test.cutoff)

    fields = _usable_fields(config_train, [df_train, df_test])
    to_scale = [scale_dict[f] for f in fields]

    model, scaler, _, _ = _search_and_refit(
        config_train, df_train, fields, to_scale, time_horizon)
    config_test.thresh = config_train.thresh

    # The censoring distribution is dataset-specific, so the test cohort gets its
    # own censoring model rather than the training cohort's.
    df_test_sel = set_paradigm(df_test, config_test.cutoff)
    test_censoring_model, df_test_sel = fit_censoring_model(df_test_sel,
                                                           time_horizon=time_horizon)
    df_test_sel = df_test_sel[df_test_sel['CENSORED'] == 0]
    weights = ipcw_weights(df_test_sel, test_censoring_model, time_horizon=time_horizon)
    X_test, y_test = _design(df_test_sel, fields, to_scale, scaler)

    results, proba, truth = evaluate((X_test, y_test), model, config_test.thresh, weights)
    return {'results': results, 'model': model, 'scaler': scaler, 'metadata': metadata,
            'fields': fields, 'proba': proba, 'labels': truth,
            'testdata': X_test, 'weights': weights}
