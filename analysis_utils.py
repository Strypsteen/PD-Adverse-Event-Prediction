"""
Shared naming and loading helpers for the reporting scripts.

Reads only the aggregated metrics the training runs write under Results/<name>/,
and imports nothing from training.py or ipcw.py, so reporting works without
lifelines installed and without touching the source CSVs.
"""
import json
import os

OUTCOME_LABELS = [('DYSK', 'Dyskinesia'), ('HAL', 'Hallucinations'),
                  ('SLEEP', 'Daytime sleepiness')]
COHORT_LABELS = [('PD_1000', 'PPMI'), ('PD_1003', 'PRoBaND'), ('PD_1017', 'NET-PD LS-1')]


def within_name(cohort, outcome, feature_set='All', prefix=''):
    return f'{prefix}{cohort}_{outcome}_{feature_set}_TH3'


def joint_name(outcome, prefix=''):
    return f'{prefix}JOINT_{outcome}_TH3'


def cross_name(outcome, train_on, evaluate_on, prefix=''):
    return f'{prefix}CROSS_{outcome}_{train_on}to{evaluate_on}_TH3'


def load_aggregated(results_dir, name, stem=None):
    """Read a <name>_output_aggregated.json, or None if the run is incomplete."""
    stem = stem or name
    path = os.path.join(results_dir, name, stem + '_output_aggregated.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
