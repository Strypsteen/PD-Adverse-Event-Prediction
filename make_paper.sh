#!/bin/sh
# Rebuild the result tables into paper/ from a finished Results/ tree.
set -e
python table_cohort.py "$@"        # cohort characteristics and dataset sizes
python tables.py "$@"              # within-, cross- and joint-dataset metrics
