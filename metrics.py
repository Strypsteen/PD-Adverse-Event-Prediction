"""
IPCW-weighted evaluation metrics.

Every metric the manuscript reports is computed with the IPCW weights, so that
censored visits (weight 0) drop out and short-follow-up subjects are corrected
for. Rows with zero weight are removed before the AUROC and calibration
computations rather than contributing zero-weight terms.

Reported per fold, then averaged over the 10 folds:
    balanced accuracy, sensitivity, specificity, precision, AUROC
Reported once on the pooled out-of-fold predictions, with a bootstrap CI:
    expected calibration error (ECE), calibration slope and intercept
"""
import numpy as np
import statsmodels.api as sm

EPS = 1e-10


def weighted_confusion_metrics(y_true, y_pred, weights):
    """Weighted sensitivity / specificity / precision / balanced accuracy / F1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)

    tp = np.sum(weights[(y_true == 1) & (y_pred == 1)])
    fn = np.sum(weights[(y_true == 1) & (y_pred == 0)])
    tn = np.sum(weights[(y_true == 0) & (y_pred == 0)])
    fp = np.sum(weights[(y_true == 0) & (y_pred == 1)])

    sensitivity = tp / (tp + fn + EPS)
    specificity = tn / (tn + fp + EPS)
    precision = tp / (tp + fp + EPS)

    return {
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'Accuracy': (sensitivity + specificity) / 2.0,     # balanced accuracy
        'F1': (2 * tp) / (2 * tp + fp + fn + EPS),
    }


def weighted_auc(y_true, y_score, weights):
    """Weighted AUROC by trapezoidal integration of the weighted ROC."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    weights = np.asarray(weights)

    mask = weights > 0
    y_true, y_score, weights = y_true[mask], y_score[mask], weights[mask]

    order = np.argsort(-y_score)
    y_true, weights = y_true[order], weights[order]

    w_pos = np.sum(weights[y_true == 1])
    w_neg = np.sum(weights[y_true == 0])

    tpr = np.r_[0, np.cumsum(weights * (y_true == 1)) / w_pos]
    fpr = np.r_[0, np.cumsum(weights * (y_true == 0)) / w_neg]
    return np.trapz(tpr, fpr), fpr, tpr


def weighted_quantile(values, weights, quantiles):
    values = np.asarray(values)
    weights = np.asarray(weights)
    sorter = np.argsort(values)
    values, weights = values[sorter], weights[sorter]
    cumulative = np.cumsum(weights)
    cumulative = cumulative / cumulative[-1]
    return np.interp(quantiles, cumulative, values)


def weighted_calibration_metrics(y_true, y_prob, weights, n_bins=10):
    """
    Weighted ECE (quantile binning, `n_bins` bins) plus the calibration slope
    and intercept from a weighted logistic recalibration on the logit scale.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    weights = np.asarray(weights)

    mask = weights > 0
    y_true, y_prob, weights = y_true[mask], y_prob[mask], weights[mask]
    total_weight = np.sum(weights)

    bin_edges = weighted_quantile(y_prob, weights, np.linspace(0, 1, n_bins + 1))
    bin_edges[0], bin_edges[-1] = 0.0, 1.0
    binids = np.digitize(y_prob, bin_edges) - 1

    ece = 0.0
    for b in range(n_bins):
        in_bin = binids == b
        if not np.any(in_bin):
            continue
        w = weights[in_bin]
        bin_weight = np.sum(w)
        observed = np.sum(w * y_true[in_bin]) / bin_weight
        predicted = np.sum(w * y_prob[in_bin]) / bin_weight
        ece += (bin_weight / total_weight) * np.abs(predicted - observed)

    eps = 1e-15
    clipped = np.clip(y_prob, eps, 1 - eps)
    logits = np.log(clipped / (1 - clipped))
    glm = sm.GLM(y_true, sm.add_constant(logits),
                 family=sm.families.Binomial(), freq_weights=weights).fit()
    intercept, slope = glm.params[0], glm.params[1]

    return ece, slope, intercept


def calibration_bootstrap_ci(y_prob, y_true, weights, n_bins=10,
                             n_bootstraps=1000, ci=95, seed=0):
    """
    Point estimate plus percentile bootstrap CI for ECE / slope / intercept on
    the pooled out-of-fold predictions. A single fold has too few visits for a
    stable calibration estimate, which is why the manuscript reports a CI here
    rather than a mean +- sd across folds.
    """
    point_ece, point_slope, point_intercept = weighted_calibration_metrics(
        y_true, y_prob, weights, n_bins=n_bins)

    # The original code drew these resamples from the ambient numpy random
    # state; seeding explicitly makes the interval reproducible across reruns.
    rng = np.random.default_rng(seed)
    n = len(y_prob)
    draws = {'ece': [], 'slope': [], 'intercept': []}
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        try:
            e, s, i = weighted_calibration_metrics(
                y_true[idx], y_prob[idx], weights[idx], n_bins=n_bins)
        except Exception:
            continue
        draws['ece'].append(e)
        draws['slope'].append(s)
        draws['intercept'].append(i)

    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2

    def summarise(point, samples):
        if not samples:
            return {'point': float(point), 'ci': [float('nan'), float('nan')]}
        return {'point': float(point),
                'ci': [float(np.percentile(samples, lo)), float(np.percentile(samples, hi))]}

    return {'ece': summarise(point_ece, draws['ece']),
            'slope': summarise(point_slope, draws['slope']),
            'intercept': summarise(point_intercept, draws['intercept'])}


def evaluate(test_set, estimator, threshold, weights):
    """
    Score one fold. Returns (metrics dict, predicted probabilities, labels);
    the probabilities are stored so the pooled ROC / calibration curves and the
    bootstrap CI can be computed once all folds have finished.
    """
    X_test, y_test = test_set
    proba = estimator.predict_proba(X_test)[:, 1]

    results = weighted_confusion_metrics(y_test, (proba > threshold).astype(int), weights)
    results['AUC'] = weighted_auc(y_test, proba, weights)[0]
    ece, slope, intercept = weighted_calibration_metrics(y_test, proba, weights, n_bins=10)
    results.update({'ECE': ece, 'Slope': slope, 'Intercept': intercept})

    return results, proba, y_test
