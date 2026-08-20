"""
Result tables, written as CSV (UTF-8 with BOM, so Excel and Word paste cleanly).

  Supplementary Table 3  within-dataset performance, 3 cohorts x 3 outcomes x 3
                         feature sets
  Supplementary Table 4  for each evaluation cohort: the within-dataset model, the
                         two cross-dataset transfer models, and the joint model

Threshold-dependent metrics and AUROC are printed as mean +- sd over the 10 runs;
ECE as the pooled bootstrap point estimate with its 95% CI. All values are
percentages. Runs whose aggregated JSON is missing print as NA, so the tables can
be built while jobs are still finishing.

    python tables.py
"""
import argparse
import csv
import os

import analysis_utils as A

FEATURE_SETS = [('All', 'All'), ('Risk', 'Risk set'), ('Levo', 'LEDD')]
METRIC_COLUMNS = ['Bal. Acc.', 'Spec.', 'Sens.', 'Prec.', 'AUROC', 'ECE']


def _pct(value):
    return 'NA' if value is None else f'{value[0] * 100:.1f} ± {value[1] * 100:.1f}'


def _ece(value):
    return 'NA' if value is None else \
        f'{value[0] * 100:.1f}, [{value[1] * 100:.1f}, {value[2] * 100:.1f}]'


def _cells(aggregate):
    if aggregate is None:
        return ['NA'] * len(METRIC_COLUMNS)
    return [_pct(aggregate['Accuracy']), _pct(aggregate['Specificity']),
            _pct(aggregate['Sensitivity']), _pct(aggregate['Precision']),
            _pct(aggregate['AUC']), _ece(aggregate['ECE'])]


def _write(path, rows):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(rows)
    print('Wrote', path)


def table_within(results_dir, outdir, prefix=''):
    header = ['Dataset', 'Feature set'] + METRIC_COLUMNS
    rows = []
    for outcome, outcome_label in A.OUTCOME_LABELS:
        rows.append([outcome_label] + [''] * (len(header) - 1))
        rows.append(header)
        for cohort, cohort_label in A.COHORT_LABELS:
            for i, (fs, fs_label) in enumerate(FEATURE_SETS):
                name = A.within_name(cohort, outcome, fs, prefix)
                rows.append([cohort_label if i == 0 else '', fs_label]
                            + _cells(A.load_aggregated(results_dir, name)))
        rows.append([''] * len(header))
    _write(os.path.join(outdir, 'supp_table3_within_dataset.csv'), rows)
    return rows


def table_cross_joint(results_dir, outdir, prefix=''):
    header = ['Evaluation set', 'Training set'] + METRIC_COLUMNS
    rows = []
    for outcome, outcome_label in A.OUTCOME_LABELS:
        rows.append([outcome_label] + [''] * (len(header) - 1))
        rows.append(header)
        for evaluate_on, eval_label in A.COHORT_LABELS:
            others = [(c, l) for c, l in A.COHORT_LABELS if c != evaluate_on]

            block = [(eval_label, A.load_aggregated(
                results_dir, A.within_name(evaluate_on, outcome, 'All', prefix)))]
            for train_on, train_label in others:
                block.append((train_label, A.load_aggregated(
                    results_dir, A.cross_name(outcome, train_on, evaluate_on, prefix))))
            joint = A.joint_name(outcome, prefix)
            block.append(('All', A.load_aggregated(
                results_dir, joint, stem=f'{joint}_to_{evaluate_on}')))

            for i, (train_label, aggregate) in enumerate(block):
                rows.append([eval_label if i == 0 else '', train_label] + _cells(aggregate))
        rows.append([''] * len(header))
    _write(os.path.join(outdir, 'supp_table4_cross_and_joint.csv'), rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results_dir', default='Results')
    ap.add_argument('--prefix', default='',
                    help="prepended to every run name; use 'CLAUDE_CENSORING_' for the "
                         "original research code's Results/ layout")
    ap.add_argument('--outdir', default='paper')
    ap.add_argument('--quiet', action='store_true', help='do not echo the tables')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for rows in (table_within(args.results_dir, args.outdir, args.prefix),
                 table_cross_joint(args.results_dir, args.outdir, args.prefix)):
        if not args.quiet:
            print()
            for row in rows:
                print(' | '.join(row))


if __name__ == '__main__':
    main()
