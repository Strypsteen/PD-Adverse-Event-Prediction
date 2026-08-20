"""
Table 2: cohort characteristics and per-outcome dataset sizes after preprocessing.

Two blocks.

Baseline characteristics, per cohort. Age, disease duration, sex and LEDD are read
from each subject's TRUE baseline visit in Data/BASELINE_DYSK_ON_<cohort>.csv,
restricted to the subjects who survive into the dyskinesia modelling dataset. The
extra row is that cohort's mean LEVO across all of their modelling visits, which
is not a baseline quantity.

Counts, per cohort and outcome. Built by applying only the CLINICAL exclusions:
subjects with fewer than two recorded visits, and subjects who already show the
outcome at their first visit. Healthy/prodromal PPMI participants and DBS
patients are excluded upstream, before these CSVs. The time-horizon censoring is
deliberately NOT applied here, so the counts describe the dataset rather than the
subset a 3-year label can be asserted for.

    python table_cohort.py
"""
import argparse
import csv
import os

import pandas as pd

from config import DATASETS, OUTCOMES, Config
from data import add_fields, get_data

OUTCOME_ROWS = [('dyskinesia', 'Dyskinesia'),
                ('hallucination', 'Hallucinations'),
                ('sleep', 'Daytime sleepiness')]
MIN_VISITS = 2


def _config(dataset, output):
    return Config({'dataset': dataset, 'output': output, 'X_fields': ['DYSK']})


def modelling_frame(dataset, output):
    """
    Per-outcome dataset after the clinical exclusions only (no censoring):
    at least MIN_VISITS recorded visits, and the outcome absent at the first visit.
    """
    cfg = _config(dataset, output)
    raw = get_data(cfg)

    visits = raw.groupby('USUBJID').size()
    raw = raw[raw['USUBJID'].isin(visits[visits >= MIN_VISITS].index)]

    df = add_fields(raw)
    kept = [g for _, g in df.groupby('USUBJID') if g.iloc[0]['DYSK'] <= cfg.cutoff]
    return pd.concat(kept) if kept else df.iloc[0:0]


def dyskinesia_subjects(dataset):
    """Subjects present in the dyskinesia modelling dataset, plus that frame."""
    df = add_fields(get_data(_config(dataset, 'dyskinesia')))
    return set(df['USUBJID'].unique()), df


def baseline_visits(dataset, data_dir='Data'):
    baseline = pd.read_csv(os.path.join(data_dir, f'BASELINE_DYSK_ON_{dataset}.csv'))
    baseline = baseline[baseline['VISIT'].astype(str).str.upper().str.contains('BASELINE')]
    return baseline.sort_values('QSDY').groupby('USUBJID', as_index=False).first()


def mean_sd(series, decimals):
    values = pd.to_numeric(series, errors='coerce').dropna()
    return f'{values.mean():.{decimals}f} ± {values.std():.{decimals}f}'


def sex_split(series):
    initial = series.astype(str).str.upper().str[0]
    return f"{(initial == 'M').mean() * 100:.1f}% / {(initial == 'F').mean() * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_dir', default='Data')
    ap.add_argument('--outdir', default='paper')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cohorts = list(DATASETS.items())
    demographics, counts = {}, {}

    for dataset, label in cohorts:
        print(f'building {label} ({dataset}) ...')
        subjects, dysk_frame = dyskinesia_subjects(dataset)
        baseline = baseline_visits(dataset, args.data_dir)
        baseline = baseline[baseline['USUBJID'].isin(subjects)]

        demographics[dataset] = {
            'Number of subjects': str(baseline['USUBJID'].nunique()),
            'Age [years]': mean_sd(baseline['AGE'], 1),
            'PD duration [years]': mean_sd(baseline['PD_DUR'] / 365.0, 1),
            'Sex [M / F]': sex_split(baseline['SEX']),
            'LEDD at baseline [mg]': mean_sd(baseline['LEVO'], 0),
            'Mean LEDD over trial [mg]':
                mean_sd(dysk_frame[dysk_frame['USUBJID'].isin(subjects)]['LEVO'], 0),
        }

        for output, _ in OUTCOME_ROWS:
            df = modelling_frame(dataset, output)
            # "subjects with the outcome" counts any non-zero score, including for
            # daytime sleepiness, where the modelling label needs a score of 2.
            # This reproduces the published table; the stricter count is printed
            # below for comparison.
            counts[(dataset, output)] = (
                len(df),
                df['USUBJID'].nunique(),
                df[df['DYSK'] > 0]['USUBJID'].nunique(),
            )
            strict = df[df['DYSK'] > OUTCOMES[output]['cutoff']]['USUBJID'].nunique()
            if strict != counts[(dataset, output)][2]:
                print(f'  note: {label} {output}: {counts[(dataset, output)][2]} subjects '
                      f'with a score >0, but only {strict} reach the modelling '
                      f'threshold of >{OUTCOMES[output]["cutoff"]:g}')

    header = [''] + [label for _, label in cohorts]
    rows = [['Baseline characteristics'] + [''] * len(cohorts), header]
    for field in demographics[cohorts[0][0]]:
        rows.append([field] + [demographics[ds][field] for ds, _ in cohorts])

    rows.append([''] * len(header))
    rows.append(['Dataset sizes after preprocessing'] + [''] * len(cohorts))
    rows.append(header)
    for output, outcome_label in OUTCOME_ROWS:
        rows.append([outcome_label] + [''] * len(cohorts))
        for i, field in enumerate(['Number of visits', 'Number of subjects',
                                   'Number of subjects with the outcome']):
            rows.append(['  ' + field] +
                        [str(counts[(ds, output)][i]) for ds, _ in cohorts])

    path = os.path.join(args.outdir, 'table2_cohorts.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(rows)
    print('Wrote', path)
    print()
    for row in rows:
        print(' | '.join(row))


if __name__ == '__main__':
    main()
