"""
Experiment configuration.

One Config object describes one (dataset, outcome, feature set) combination.
Everything that was a switch in the original research code but takes a single
value in the paper runs is fixed here instead of being configurable:

    visit selection      first incidence  (severe first incidence for sleepiness)
    label binarisation   >=1 present      (>=2 for sleepiness)
    inclusion criteria   all visits that survive add_fields()
    ON/OFF state         ON
    calibration          beta calibration, 5-fold
    prediction horizon   3 years (still a CLI argument, since it is the paper's N)
"""

DATASETS = {
    'PD-1000': 'PPMI',
    'PD-1003': 'PRoBaND',
    'PD-1017': 'NET-PD LS-1',
}

# outcome -> (csv stem, MDS-UPDRS-derived severity cut-off for a positive label)
#
# The stored `DYSK` column always holds the outcome of that file as a 0-4
# ordinal clinician score. `cutoff` is the value the score must EXCEED for the
# outcome to count as present, which reproduces the manuscript definitions:
#   dyskinesia     MDS-UPDRS 4.2 >= 1   -> cutoff 0
#   hallucinations MDS-UPDRS 1.2 >= 1   -> cutoff 0
#   sleepiness     MDS-UPDRS 1.8 >= 2   -> cutoff 1
OUTCOMES = {
    'dyskinesia':    {'stem': 'FINAL_DYSK',  'cutoff': 0.0},
    'hallucination': {'stem': 'FINAL_HAL',   'cutoff': 0.0},
    'sleep':         {'stem': 'FINAL_SLEEP', 'cutoff': 1.0},
}

DATA_DIR = 'Data'

# Column holding the label the models are trained on: "did the outcome occur in
# any visit within the prediction horizon". Built by data.set_time_horizon().
OUTCOME_VARIABLE = 'FUTURE_DYSK_TIMEHORIZON'


class Config:
    """dataset + outcome + feature order for one experiment."""

    def __init__(self, dictionary):
        self.dataset = dictionary['dataset']
        self.output = dictionary['output']
        self.X_fields = list(dictionary['X_fields'])
        self.seed = dictionary.get('seed', 0)
        self.thresh = None          # set by training, used by evaluation

        if self.dataset not in DATASETS:
            raise ValueError(f'unknown dataset {self.dataset!r}; expected one of {list(DATASETS)}')
        if self.output not in OUTCOMES:
            raise ValueError(f'unknown outcome {self.output!r}; expected one of {list(OUTCOMES)}')

    @property
    def cutoff(self):
        """Ordinal score the outcome must exceed to count as present."""
        return OUTCOMES[self.output]['cutoff']

    def get_path(self):
        stem = OUTCOMES[self.output]['stem']
        return f'{DATA_DIR}/{stem}_ON_{self.dataset}.csv'


def load_config(config_path, seed):
    """Read a Configs/*.json and stamp the seed the run was launched with."""
    import json
    with open(config_path) as f:
        cfg = Config(json.load(f))
    cfg.seed = seed
    return cfg
