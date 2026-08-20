"""
Inverse probability of censoring weighting (IPCW).

Predicting N years ahead means that for a visit whose outcome has not occurred
yet, a subject who leaves the study before the horizon cannot contribute a
definite negative. Those rows are flagged CENSORED by data.set_time_horizon and
receive weight 0; the remaining rows are up-weighted by the inverse of their
estimated probability of staying uncensored to the horizon, which corrects the
under-representation of short-follow-up subjects.

The censoring process is modelled with a Cox proportional-hazards model on the
time from the current visit to the subject's last visit, with the study day
(QSDY) as the single covariate. The model is fitted on training data only and
then applied to validation / test data.
"""
import numpy as np
from lifelines import CoxPHFitter

# Weights are clipped to this range before use, so that a near-zero survival
# probability cannot let a handful of rows dominate the fit or the metrics.
WEIGHT_CLIP = (0.1, 10.0)


def fit_censoring_model(train_df, time_horizon=3.0):
    """
    Fit the censoring model on `train_df`.

    Returns (model, train_df) - the frame is returned because the two helper
    columns it gains ('time_to_event', 'event') are written in place.
    """
    train_df['time_to_event'] = train_df['AVAILABLE_TIME'] / 365.0
    train_df['event'] = (train_df['AVAILABLE_TIME'] < time_horizon * 365.0).astype(int)

    cox_df = train_df[['time_to_event', 'event', 'QSDY']].dropna()
    model = CoxPHFitter()
    model.fit(cox_df, duration_col='time_to_event', event_col='event')
    return model, train_df


def ipcw_weights(df, censoring_model, time_horizon=3.0):
    """
    IPCW weight per row: 1 / P(uncensored until the horizon | QSDY), clipped,
    with censored rows forced to 0.
    """
    surv = censoring_model.predict_survival_function(df[['QSDY']].copy(),
                                                     times=[time_horizon])
    prob_uncensored = surv.values.flatten()

    weights = 1.0 / (prob_uncensored + 1e-6)
    weights = np.clip(weights, *WEIGHT_CLIP)
    weights[df['CENSORED'].values == 1] = 0.0
    return weights
