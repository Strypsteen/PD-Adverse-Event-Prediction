"""
Result accumulation.

Each of the 10 folds (or 10 seeds, for the cross-dataset setting) is a separate
process. They append to shared files under Results/<name>/ behind a file lock:

    <name>_output.json        dict of metric -> list of per-fold values
    <name>_proba.npz          arr_i : predicted probabilities for fold i
    <name>_label.npz          arr_i : labels for fold i
    <name>_IPCW_weights.npz   arr_i : IPCW test weights for fold i
    <name>_testdata.npz       arr_i : scaled test design matrix for fold i
    <name>_estimator.pkl      list of the 10 fitted models
    <name>_scaler.pkl         list of the 10 fitted scalers

When the tenth fold lands, the same process writes

    <name>_output_aggregated.json

holding mean +- sd across folds for the threshold-dependent metrics and AUROC,
and a bootstrap point estimate + 95% CI for ECE / slope / intercept computed on
the pooled out-of-fold predictions. A single fold holds too few visits for a
stable calibration estimate, which is why calibration is reported that way.

Per-fold metrics are stored rounded to 3 decimals, and the reported means are
means of those rounded values; this matches the published pipeline.
"""
import json
import os
import pickle

import numpy as np
from filelock import FileLock

METRIC_KEYS = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1', 'AUC',
               'ECE', 'Intercept', 'Slope']

# Metrics reported as mean +- sd over the folds. ECE / slope / intercept are
# instead reported from the pooled bootstrap, so they are not listed here.
FOLD_METRICS = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1', 'AUC']


def _append_json(path, new_dict):
    try:
        with open(path + '.json') as f:
            payload = json.load(f)
    except (OSError, ValueError):
        payload = {k: [] for k in METRIC_KEYS}
    for key, value in new_dict.items():
        payload[key].append(round(float(value), 3))
    with open(path + '.json', 'w') as f:
        json.dump(payload, f, indent=2)


def _append_npz(path, array):
    try:
        with np.load(path + '.npz') as data:
            arrays = [data[f'arr_{i}'] for i in range(len(data.files))]
    except (OSError, ValueError):
        arrays = []
    arrays.append(array)
    np.savez(path + '.npz', **{f'arr_{i}': a for i, a in enumerate(arrays)})


def _append_pkl(path, obj):
    try:
        with open(path + '.pkl', 'rb') as f:
            payload = pickle.load(f)
    except (OSError, ValueError, EOFError):
        payload = []
    payload.append(obj)
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(payload, f, pickle.HIGHEST_PROTOCOL)


def _load_all(path):
    with np.load(path + '.npz') as data:
        return np.concatenate([data[f'arr_{i}'] for i in range(len(data.files))], axis=0)


def _n_runs(path):
    with np.load(path + '.npz') as data:
        return len(data.files)


def _append_order(path, run_id):
    try:
        with open(path + '.json') as f:
            order = json.load(f)
    except (OSError, ValueError):
        order = []
    order.append(run_id)
    with open(path + '.json', 'w') as f:
        json.dump(order, f)


def save_fold(savepath, name, results, proba, labels, weights, testdata,
              estimator=None, scaler=None, fields=None, run_id=None,
              n_expected=10, seed=0):
    """
    Append one run's artefacts and, once `n_expected` runs are present, write the
    aggregated JSON. Returns the aggregate dict, or None if runs are outstanding.
    """
    os.makedirs(savepath, exist_ok=True)
    stem = os.path.join(savepath, name)

    # The feature list actually used, which is not always the config's: NET-PD
    # LS-1 lacks five columns, so its within- and cross-dataset runs train on
    # fewer features. Recording it here keeps the analysis scripts honest about
    # what the columns of _testdata.npz mean.
    if fields is not None and not os.path.exists(stem + '_fields.json'):
        with open(stem + '_fields.json', 'w') as f:
            json.dump(list(fields), f, indent=2)

    _append_json(stem + '_output', results)
    # Rows land in job-completion order, not fold order, so record which fold
    # each row came from. Without this, two runs cannot be compared fold by fold.
    if run_id is not None:
        _append_order(stem + '_fold_order', run_id)
    _append_npz(stem + '_proba', proba)
    _append_npz(stem + '_label', labels)
    _append_npz(stem + '_IPCW_weights', weights)
    _append_npz(stem + '_testdata', testdata)
    if estimator is not None:
        _append_pkl(stem + '_estimator', estimator)
    if scaler is not None:
        _append_pkl(stem + '_scaler', scaler)

    done = _n_runs(stem + '_proba')
    print(f'{name}: {done}/{n_expected} runs complete')
    if done != n_expected:
        return None
    return _write_aggregate(stem, seed=seed)


def _write_aggregate(stem, seed=0):
    from metrics import calibration_bootstrap_ci

    with open(stem + '_output.json') as f:
        per_fold = json.load(f)

    # The run is gated on the number of saved prediction folds but the means are
    # taken over the metric rows. If a stale row survives from an earlier attempt
    # those two disagree and the published mean silently averages the wrong set,
    # so refuse rather than aggregate. Delete the run directory and start over.
    n_rows, n_folds = len(per_fold['AUC']), _n_runs(stem + '_proba')
    if n_rows != n_folds:
        raise ValueError(
            f'{os.path.basename(stem)}: {n_rows} metric rows in _output.json but '
            f'{n_folds} folds in _proba.npz. The directory holds results from more '
            f'than one attempt; remove it and re-run.')

    aggregate = {}
    for metric in FOLD_METRICS:
        values = np.array(per_fold[metric], dtype=float)
        aggregate[metric] = [float(values.mean()), float(values.std())]

    calibration = calibration_bootstrap_ci(
        _load_all(stem + '_proba'), _load_all(stem + '_label'),
        _load_all(stem + '_IPCW_weights'),
        n_bins=10, n_bootstraps=1000, ci=95, seed=seed)
    for key, label in (('ece', 'ECE'), ('intercept', 'Intercept'), ('slope', 'Slope')):
        aggregate[label] = [calibration[key]['point'], *calibration[key]['ci']]

    with open(stem + '_output_aggregated.json', 'w') as f:
        json.dump(aggregate, f, indent=2)
    print('Wrote', stem + '_output_aggregated.json')
    return aggregate


def save_shared(savepath, name, estimator, scaler, fields):
    """
    Store artefacts that belong to the run as a whole rather than to one
    evaluation cohort. The joint setting fits a single model and scaler and then
    evaluates it on three cohorts, so these are appended once per fold under the
    run name instead of once per cohort.
    """
    os.makedirs(savepath, exist_ok=True)
    stem = os.path.join(savepath, name)
    _append_pkl(stem + '_estimator', estimator)
    _append_pkl(stem + '_scaler', scaler)
    if not os.path.exists(stem + '_fields.json'):
        with open(stem + '_fields.json', 'w') as f:
            json.dump(list(fields), f, indent=2)


def lock_for(name):
    """Cross-process lock so concurrent runs cannot interleave their appends."""
    return FileLock(name + '_global_lock.lock', timeout=300)
