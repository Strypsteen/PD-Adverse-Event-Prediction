"""
Cross-dataset experiment (the transfer rows of Supplementary Table 4): train on
all of one cohort, evaluate on all of another.

There is no outer fold here - the whole training cohort is used - so the ten
repeats vary the seed instead, which resamples the inner validation split.

    python run_cross.py --config_train PD_1000_DYSK_All --config_test PD_1003_DYSK_All \
        --name CROSS_DYSK_PD_1000toPD_1003_TH3 --seed 109999
"""
import argparse
import os

from config import load_config
from results_io import lock_for, save_fold
from training import run_cross


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config_train', required=True)
    ap.add_argument('--config_test', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--time_horizon', type=float, default=3.0)
    args = ap.parse_args()

    cfg_train = load_config(os.path.join('Configs', args.config_train + '.json'), args.seed)
    cfg_test = load_config(os.path.join('Configs', args.config_test + '.json'), args.seed)
    print(f'### {args.name} | train {cfg_train.dataset} -> test {cfg_test.dataset} '
          f'| {cfg_train.output} | seed {args.seed}')

    out = run_cross(cfg_train, cfg_test, time_horizon=args.time_horizon)

    with lock_for(args.name):
        save_fold(os.path.join('Results', args.name), args.name,
                  out['results'], out['proba'], out['labels'], out['weights'], out['testdata'],
                  estimator=out['model'], scaler=out['scaler'], fields=out['fields'],
                  run_id=args.seed, seed=args.seed)


if __name__ == '__main__':
    main()
