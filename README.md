# PD Adverse Event Prediction

Code for *AI-enabled personalized medication management through cross-cohort
prediction of treatment-related adverse events in Parkinson's disease*
(Strypsteen et al.).

Three random-forest models estimate the probability that a person with Parkinson's
Disease who has **not yet** experienced dyskinesia, hallucinations or excessive
daytime sleepiness will develop it **within 3 years** under a given medication
regimen. Visits are selected under the first-incidence paradigm, probabilities are
beta-calibrated, and inverse probability of censoring weighting (IPCW) corrects for
subjects whose follow-up is shorter than the prediction horizon.

## Install

```sh
pip install -r requirements.txt
```

## Data

Visit-level records from three cohorts of the Critical Path for Parkinson's (CPP)
Integrated Database: PPMI (`PD-1000`), PRoBaND / Tracking Parkinson's (`PD-1003`)
and NET-PD LS-1 (`PD-1017`).

**The data is not in this repository and cannot be redistributed.** Access runs
through a formal application to the Critical Path for Parkinson's. Place the
extracted CSVs in `Data/`:

```
Data/FINAL_{DYSK,HAL,SLEEP}_ON_PD-{1000,1003,1017}.csv    9 files, training
Data/BASELINE_DYSK_ON_PD-{1000,1003,1017}.csv             3 files, table_cohort.py only
```

One row per visit per subject. The outcome column is named `DYSK` in every file —
in `FINAL_HAL_*` it holds the hallucination score, in `FINAL_SLEEP_*` the
sleepiness score — and `FUTURE_DYSK` is its value at the next visit. Columns used:
`USUBJID`, `QSDY`, `FUTURE_DAY`, `DYSK`, `FUTURE_DYSK`, `AGE`, `SEX`, `PD_DUR`,
`WEIGHT`, `HEIGHT`, `UPDRS_I/II/III`, `HY`, `ON_OFF`, `MOCA`, `LEVO`,
`FUTURE_LEVO`, `FUTURE_MedClass1..5`, `Hallucinations`, `Insomnia`, `Gait`,
`Sleepiness`.

## Training

Three settings, one script each. Each invocation is a single job; the ten jobs of a
run append to shared files under `Results/<name>/` behind a file lock, so they can
execute concurrently, and whichever finishes last writes the aggregated metrics.

```sh
# within-dataset: one cohort, 10 outer subject folds
python run_within.py --config PD_1000_DYSK_All --name PD_1000_DYSK_All_TH3 \
    --seed 109999 --fold 0

# joint: training folds of all three cohorts pooled, evaluated on each separately
python run_joint.py --outcome DYSK --name JOINT_DYSK_TH3 --seed 109999 --fold 0

# cross-dataset: train on one cohort, evaluate on another. There is no held-out
# split, so the ten repeats vary the seed rather than a fold.
python run_cross.py --config_train PD_1000_DYSK_All --config_test PD_1003_DYSK_All \
    --name CROSS_DYSK_PD_1000toPD_1003_TH3 --seed 109999
```

`--config` names a file in `Configs/`, which holds 27 combinations of cohort ×
outcome × feature set (`All`, `Risk`, `Levo`). `--time_horizon` defaults to 3.0
years.

The full matrix is 480 jobs — 270 within, 180 cross, 30 joint:

```sh
./run_all.sh                                             # sequential, local
./run_all.sh joint                                       # one stage only
SUBMIT="sbatch slurm/within.slurm" ./run_all.sh within   # via a scheduler
```

**Expect hours per job.** Each searches 54 hyperparameter points × 5 inner folds,
and every fit is a 5-member calibrated forest ensemble. Run this on a cluster and
adapt the wrappers in `slurm/` to your environment.

## Reporting

```sh
./make_paper.sh              # both, into paper/
```

or individually:

```sh
python table_cohort.py       # cohort characteristics and dataset sizes
python tables.py             # within-, cross- and joint-dataset metrics
```

These scripts read only the artefacts training writes under `Results/`, so they
need neither the source CSVs nor `lifelines`. Pass `--results_dir` and `--prefix`
to point them at a differently named results tree.

## Layout

```
config.py            cohort/outcome definitions, feature order, severity cut-offs
data.py              CSV -> first-incidence, N-year-horizon modelling table
ipcw.py              Cox censoring model and the IPCW weights
models.py            beta-calibrated random forest
metrics.py           IPCW-weighted metrics, calibration, bootstrap CI
training.py          nested CV; the within / cross / joint settings
results_io.py        artefact accumulation and aggregation across jobs
analysis_utils.py    run-name helpers and aggregated-metric loading
tables.py            within-, cross- and joint-dataset metric tables
table_cohort.py      cohort characteristics
Configs/             27 experiment configs
slurm/               scheduler wrappers
```

Each run writes to `Results/<name>/`: per-fold metrics and predictions, the fitted
models and scalers, the feature list used, and `<name>_output_aggregated.json` once
all folds are in.

## Citation and licence

Code: MIT. Source data: governed by the CPP Integrated Database data-sharing
agreement, not by this licence.

Funded by the European Union, Horizon Europe grant 101080581 (AI-PROGNOSIS).
