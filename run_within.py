"""
Within-dataset experiment (Supplementary Table 3): train and evaluate on the
folds of a single cohort.

Run once per (config, fold); the tenth fold writes the aggregated metrics.

    python run_within.py --config PD_1000_DYSK_All --name PD_1000_DYSK_All_TH3 \
        --seed 109999 --fold 0
"""
import argparse
import os

from config import load_config
from results_io import lock_for, save_fold
from training import run_within


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='stem of a Configs/*.json file')
    ap.add_argument('--name', required=True, help='experiment name; results go to Results/<name>/')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--fold', type=int, required=True, help='outer fold, 0-9')
    ap.add_argument('--time_horizon', type=float, default=3.0)
    args = ap.parse_args()

    cfg = load_config(os.path.join('Configs', args.config + '.json'), args.seed)
    print(f'### {args.name} | {cfg.dataset} {cfg.output} | fold {args.fold} | seed {args.seed}')

    out = run_within(cfg, args.fold, time_horizon=args.time_horizon)

    with lock_for(args.name):
        save_fold(os.path.join('Results', args.name), args.name,
                  out['results'], out['proba'], out['labels'], out['weights'], out['testdata'],
                  estimator=out['model'], scaler=out['scaler'], fields=out['fields'],
                  run_id=args.fold, seed=args.seed)


if __name__ == '__main__':
    main()
