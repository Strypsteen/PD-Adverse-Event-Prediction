#!/bin/sh
# =============================================================================
# The complete experiment matrix behind the paper.
#
#   within  27 runs (3 cohorts x 3 outcomes x 3 feature sets) x 10 folds  = 270 jobs
#   cross   18 runs (3 outcomes x 6 ordered cohort pairs)     x 10 seeds  = 180 jobs
#   joint    3 runs (3 outcomes)                              x 10 folds  =  30 jobs
#                                                                    total  480 jobs
#
# Every job searches 54 hyperparameter points x 5 inner folds, each fitting a
# 5-member beta-calibrated forest ensemble, so a single job takes hours. Run them
# on a cluster; SUBMIT below is the hook for that. The default runs them
# sequentially in the foreground, which is only sensible for a smoke test.
#
#   ./run_all.sh                     # sequential, local
#   SUBMIT="sbatch slurm/within.slurm" ./run_all.sh within
#   ./run_all.sh joint               # one stage only: within | cross | joint
#
# The jobs of one run append to shared files under Results/<name>/ behind a file
# lock, so they may run concurrently; the last one to finish writes the
# aggregated metrics.
# =============================================================================
set -e

SEED=109999
TH=3.0
COHORTS="PD_1000 PD_1003 PD_1017"
OUTCOMES="DYSK HAL SLEEP"
FEATURE_SETS="All Risk Levo"

STAGES="${1:-within cross joint}"

run_within() {
    for cohort in $COHORTS; do
        for outcome in $OUTCOMES; do
            for fs in $FEATURE_SETS; do
                for fold in 0 1 2 3 4 5 6 7 8 9; do
                    ${SUBMIT:-python -u run_within.py} \
                        --config "${cohort}_${outcome}_${fs}" \
                        --name "${cohort}_${outcome}_${fs}_TH3" \
                        --seed $SEED --fold $fold --time_horizon $TH
                done
            done
        done
    done
}

run_cross() {
    for outcome in $OUTCOMES; do
        for train in $COHORTS; do
            for test in $COHORTS; do
                [ "$train" = "$test" ] && continue
                # cross-dataset training uses the whole training cohort, so the ten
                # repeats vary the seed rather than an outer fold
                i=0
                while [ $i -lt 10 ]; do
                    ${SUBMIT:-python -u run_cross.py} \
                        --config_train "${train}_${outcome}_All" \
                        --config_test "${test}_${outcome}_All" \
                        --name "CROSS_${outcome}_${train}to${test}_TH3" \
                        --seed $((SEED + i)) --time_horizon $TH
                    i=$((i + 1))
                done
            done
        done
    done
}

run_joint() {
    for outcome in $OUTCOMES; do
        for fold in 0 1 2 3 4 5 6 7 8 9; do
            ${SUBMIT:-python -u run_joint.py} \
                --outcome "$outcome" \
                --name "JOINT_${outcome}_TH3" \
                --seed $SEED --fold $fold --time_horizon $TH
        done
    done
}

for stage in $STAGES; do
    echo "===== stage: $stage ====="
    case "$stage" in
        within) run_within ;;
        cross)  run_cross ;;
        joint)  run_joint ;;
        *) echo "unknown stage '$stage' (expected within, cross or joint)" >&2; exit 1 ;;
    esac
done
