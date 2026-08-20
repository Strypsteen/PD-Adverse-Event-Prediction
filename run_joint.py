"""
Joint-dataset experiment (Fig. 2, Fig. 3 and the joint rows of Supplementary
Table 4): pool the training folds of all three cohorts, then evaluate the single
model on each cohort's held-out fold separately.

Results are written to Results/<name>/<name>_to_PD_1000..., one set per cohort.

    python run_joint.py --outcome DYSK --name JOINT_DYSK_TH3 --seed 109999 --fold 0
"""
import argparse
import os

from config import load_config
from results_io import lock_for, save_fold, save_shared
from training import run_joint

COHORTS = ['PD_1000', 'PD_1003', 'PD_1017']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outcome', required=True, choices=['DYSK', 'HAL', 'SLEEP'])
    ap.add_argument('--name', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--fold', type=int, required=True, help='outer fold, 0-9')
    ap.add_argument('--time_horizon', type=float, default=3.0)
    args = ap.parse_args()

    # The joint model always uses the full feature set.
    configs = [load_config(os.path.join('Configs', f'{ds}_{args.outcome}_All.json'), args.seed)
               for ds in COHORTS]
    print(f'### {args.name} | joint {args.outcome} | fold {args.fold} | seed {args.seed}')

    out = run_joint(configs, args.fold, time_horizon=args.time_horizon)

    savepath = os.path.join('Results', args.name)
    with lock_for(args.name):
        # one model and scaler, evaluated on three cohorts
        save_shared(savepath, args.name, out['model'], out['scaler'], out['fields'])
        for i, ds in enumerate(COHORTS):
            save_fold(savepath, f'{args.name}_to_{ds}',
                      out['results'][i], out['proba'][i], out['labels'][i],
                      out['weights'][i], out['testdata'][i],
                      fields=out['fields'], run_id=args.fold, seed=args.seed)


if __name__ == '__main__':
    main()
