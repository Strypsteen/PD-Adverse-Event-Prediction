"""
Dataset construction: raw per-visit CSV -> first-incidence, N-year-horizon
modelling table.

Pipeline order (identical for the within-, cross- and joint-dataset experiments):

    get_data          read Data/FINAL_<OUT>_ON_<DS>.csv
    add_fields        derive PD_age / PD_time / BMI / LEDD_years, drop screening
                      visits and any visit without a following visit
    set_status        binarise the next-visit outcome at the outcome's cut-off
    set_time_horizon  build the training label (outcome within N years) and the
                      CENSORED flag, then drop each subject's last visit
    drop_affected_at_baseline
                      remove subjects who already have the outcome at their
                      first visit
    set_paradigm      first incidence: keep visits up to and including the first
                      one whose horizon label is positive

`cutoff` (from Config) is the ordinal score the ordinal outcome column must
EXCEED to count as present: 0 for dyskinesia and hallucinations, 1 for daytime
sleepiness. Threading it through is what lets a single code path serve all three
outcomes.

Note on column naming: the outcome column is called `DYSK` in every file
(dyskinesia, hallucinations and sleepiness alike) and `FUTURE_DYSK` is its value
at the next visit. That naming is inherited from the stored CSVs.
"""
import warnings

import numpy as np
import pandas as pd

from config import OUTCOME_VARIABLE

warnings.filterwarnings("ignore", 'This pattern has match groups')

# Which raw feature columns the pipeline standardises. Binary / categorical
# columns are passed to the forest unscaled.
scale_dict = {
    'DYSK': False,
    'SEX': False,
    'AGE': True,
    'PD_age': True,
    'PD_time': True,
    'BMI': True,
    'UPDRS_III': True,
    'UPDRS_II': True,
    'UPDRS_I': True,
    'HY': True,
    'ON_OFF': False,
    'MOCA': True,
    'LEVO': True,
    'FUTURE_LEVO': True,
    'FUTURE_MedClass1': False,
    'FUTURE_MedClass2': False,
    'FUTURE_MedClass3': False,
    'FUTURE_MedClass4': False,
    'FUTURE_MedClass5': False,
    'LEDD_years': True,
    'FUTURE_LEDD_years': True,
    'Hallucinations': False,
    'Insomnia': False,
    'Gait': False,
    'Sleepiness': False,
}


def get_data(config):
    return pd.read_csv(config.get_path())


def add_fields(df):
    """
    Derive the composite features and apply the two structural exclusions that
    are independent of the outcome:
      - visits with no recorded study day or no following visit (screening and
        last visits) are dropped;
      - visits whose following visit is not strictly later are dropped.
    """
    # Remove screening / symptomatic-therapy visits and visits with no successor
    df = df.dropna(subset=['QSDY', 'FUTURE_DAY'])

    # Age at PD onset, in years
    df.insert(2, "PD_age", df['AGE'] - (df['PD_DUR'] / 365.0), True)

    # Total disease duration at this visit, in years
    df.insert(2, "PD_time", (df['QSDY'] / 365.0) + (df['PD_DUR'] / 365.0), True)

    # Height/weight -> BMI
    df.insert(2, "BMI", df['WEIGHT'].values / ((0.01 * df['HEIGHT'].values) ** 2), True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = df.replace({'SEX': {'M': 1, 'F': -1}})

    # LEDD years: total LEDD integrated over the time it was taken, cumulated
    # over the subject's visits. LEDD_years is the value carried INTO the visit,
    # i.e. FUTURE_LEDD_years shifted by one visit.
    def return_LEDD_years(x):
        x['interm'] = x['LEVO'] * (x['FUTURE_DAY'] / 365.0 - x['QSDY'] / 365.0)
        return x['interm'].cumsum()

    df['FUTURE_LEDD_years'] = df.groupby(['USUBJID']).apply(
        lambda x: return_LEDD_years(x)).reset_index(level=0, drop=True)
    df_out = df.groupby(['USUBJID'], dropna=False)[['FUTURE_LEDD_years']].apply(
        lambda x: x.shift(periods=1)).reset_index(level=0)
    df = df.join(df_out, how='left', rsuffix='_y')
    df = df.rename(columns={'FUTURE_LEDD_years_y': 'LEDD_years'})

    # FUTURE_DAY becomes the gap to the next visit; keep only forward-looking rows
    df['FUTURE_DAY'] = df['FUTURE_DAY'] - df['QSDY']
    df = df[df['FUTURE_DAY'] > 0]

    return df.sort_values(by=['USUBJID', 'QSDY'])


def set_status(df, cutoff):
    """Binarise the next-visit outcome at `cutoff`, leaving missing values missing."""
    positive = df['FUTURE_DYSK'] > cutoff
    df.loc[positive, 'FUTURE_DYSK'] = 1.0
    df.loc[~positive & df['FUTURE_DYSK'].notna(), 'FUTURE_DYSK'] = 0.0
    return df.sort_values(by=['USUBJID', 'QSDY'])


def set_time_horizon(df, time_horizon=3.0, cutoff=0.0):
    """
    Build the training label and the censoring flag.

    OUTCOME_VARIABLE is 1 if any later visit of the same subject within
    `time_horizon` years records the outcome above `cutoff`.

    CENSORED is 1 when the label is 0 but the subject's remaining follow-up is
    shorter than the horizon, so a 0 cannot be asserted. Those rows get IPCW
    weight 0 and are dropped from training and evaluation; the weighting of the
    surviving rows is what compensates for them (see ipcw.py).
    """
    def outcome_within_horizon(row, subject_df):
        future = subject_df[
            (subject_df['QSDY'] > row['QSDY']) &
            (subject_df['QSDY'] - row['QSDY'] < 365.0 * time_horizon) &
            (subject_df['DYSK'] > cutoff)
        ]
        return 0 if future.empty else 1

    df[OUTCOME_VARIABLE] = df.groupby('USUBJID', group_keys=False).apply(
        lambda subject_df: subject_df.apply(
            lambda row: outcome_within_horizon(row, subject_df), axis=1))

    df['AVAILABLE_TIME'] = df.groupby('USUBJID')['QSDY'].transform(lambda x: x.max() - x)
    df['CENSORED'] = np.logical_and(df['AVAILABLE_TIME'] < time_horizon * 365.0,
                                    df[OUTCOME_VARIABLE] == 0).astype(int)

    # Each subject's last visit has no follow-up at all
    df = df[df['AVAILABLE_TIME'] > 0]
    return df.dropna(subset=[OUTCOME_VARIABLE])


def drop_affected_at_baseline(df, cutoff):
    """Drop every subject whose first retained visit already shows the outcome."""
    def keep(x):
        if x.iloc[0]['DYSK'] > cutoff:
            return x[0:0]
        return x[:]

    return df.groupby(['USUBJID']).apply(
        lambda x: keep(x), include_groups=False).reset_index(level=0)


def set_paradigm(df, cutoff):
    """
    First-incidence visit selection: for each subject keep the visits up to and
    including the first one whose next-visit outcome is positive, so the outcome
    is absent at every visit a prediction is made for.
    """
    df = df.sort_values(by=['USUBJID', 'QSDY'])

    def visits_before_onset(x):
        if x.iloc[0]['DYSK'] > cutoff:
            return x[0:0]
        if not (x['FUTURE_DYSK'] > 0.0).any():
            return x[:]
        return x.iloc[:1 + np.where(x['FUTURE_DYSK'] > 0.0)[0][0]]

    df = df.groupby(['USUBJID']).apply(
        lambda x: visits_before_onset(x), include_groups=False).reset_index(level=0)
    return df[df['DYSK'] <= cutoff]


def build_dataset(config, time_horizon=3.0):
    """get_data -> add_fields -> set_status -> set_time_horizon, plus metadata."""
    df = add_fields(get_data(config))
    df = set_status(df, config.cutoff)
    metadata = get_metadata(df)
    df = set_time_horizon(df, time_horizon=time_horizon, cutoff=config.cutoff)
    return df, metadata


def get_metadata(df):
    return {
        'N_visits': len(df),
        'N_subjects': len(df.USUBJID.unique()),
        'N_visits_currentoutcome_1': len(df[df['DYSK'] > 0]),
        'N_subjects_currentoutcome_1': len(df[df['DYSK'] > 0].USUBJID.unique()),
        'N_visits_nextoutcome_1': len(df[df['FUTURE_DYSK'] > 0]),
        'N_subjects_nextoutcome_1': len(df[df['FUTURE_DYSK'] > 0].USUBJID.unique()),
        'N_visits_transition_01': len(df[np.logical_and(df['FUTURE_DYSK'] > 0, df['DYSK'] == 0)]),
        'Median day differential': df['FUTURE_DAY'].median(),
        'Mean levodopa differential': (df['FUTURE_LEVO'] - df['LEVO']).mean(),
        'Max levodopa': (df['LEVO']).max(),
    }


def create_subject_folds(subjects, n_folds=10, seed=42):
    """
    Reproducible subject-level k-fold split. Subjects are sorted before
    shuffling so the split depends only on the seed, not on row order.
    """
    import random as _random
    rng = _random.Random(seed)
    np.random.seed(seed)
    subjects_sorted = sorted(subjects)
    rng.shuffle(subjects_sorted)

    folds = []
    fold_size = len(subjects_sorted) // n_folds
    remainder = len(subjects_sorted) % n_folds
    start = 0
    for i in range(n_folds):
        end = start + fold_size + (1 if i < remainder else 0)
        test_subjects = subjects_sorted[start:end]
        train_subjects = np.setdiff1d(subjects_sorted, test_subjects).tolist()
        folds.append({'train': train_subjects, 'test': test_subjects})
        start = end
    return folds


def subject_kfold(subject_indices, K, seed):
    """Inner-loop split of training subjects into K validation folds."""
    rng = np.random.default_rng(seed)
    indices = np.array(subject_indices)
    rng.shuffle(indices)
    test_folds = np.array_split(indices, K)
    train_indices, val_indices = [], []
    for k in range(K):
        val_indices.append(test_folds[k])
        train_indices.append(np.concatenate([f for i, f in enumerate(test_folds) if i != k]))
    return train_indices, val_indices


def select_subjects(df, subjects):
    return pd.concat([df[df['USUBJID'] == s] for s in subjects])
