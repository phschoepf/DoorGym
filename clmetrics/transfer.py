import argparse
import os
import sqlite3
import yaml

import numpy as np
import DoorGym.enjoy as enjoy
from DoorGym.a2c_ppo_acktr.arguments import add_common_args
from DoorGym.deterministic import set_seed

from continuousworld import run_eval

def make_accuracy_matrix(config):
    """Make the accuracy matrix of a given run. The accuracy matrix is an NxN array for N tasks trained on the hnet.
    amatrix[A,B] is the accuracy of task B, evaluated on the network after training on task A."""

    experiments, worlds = zip(*[(os.path.join(os.path.expanduser(config['checkpoint_root']), run['checkpoint']),
                                 os.path.join(os.path.expanduser(config['world_root']), run['world']))
                                for run in config['runs']])

    assert len(experiments) == len(worlds), "Number of experiments and world paths has to be the same"
    for file in experiments:
        assert os.path.isfile(file), f'Checkpoint {os.path.basename(file)} does not exist'
    for dir_ in worlds:
        assert os.path.isdir(dir_) and len(os.listdir(dir_)) != 0, f'World folder {os.path.basename(dir_)} does not exist or is empty'

    accuracy_mat = np.zeros((len(experiments), len(experiments)))

    # iterate over checkpoint files
    for train_tid, checkpoint in enumerate(experiments):
        # iterate over tasks
        for eval_tid, world in enumerate(worlds):

            # we can't give tids higher than the train_tid to enjoy.py, thus we clip it to the highest tid for all other tasks
            # evaluating on not-yet-trained tasks can reveal zero-shot capabilities
            tid_to_evaluate = min(train_tid, eval_tid)
            print(f'evaluating task {eval_tid} on checkpoint of task {train_tid} (world {os.path.basename(world)})')

            opening_rate = run_eval(checkpoint, world, tid_to_evaluate, db_connection, args)
            accuracy_mat[train_tid, eval_tid] = opening_rate

    return accuracy_mat


class CLMetric:
    def __init__(self, amatrix: np.ndarray):
        self.accuracy_matrix = amatrix
        self.n = self.accuracy_matrix.shape[0]

    def __call__(self, *args, **kwargs):
        return self.all_metrics()

    def __str__(self):
        return str(self.__call__())

    def accuracy(self):
        """from https://arxiv.org/abs/1810.13166"""
        acc_triangle = np.tril(self.accuracy_matrix, k=0)
        return np.sum(acc_triangle) * 2 / (self.n * (self.n + 1))

    def forward_transfer(self):
        """from https://arxiv.org/abs/1810.13166"""
        fwt_matrix = np.triu(self.accuracy_matrix, k=1)
        return np.sum(fwt_matrix) * 2 / (self.n * (self.n-1))

    def _backward_transfer(self):
        """from https://arxiv.org/abs/1810.13166. This is the BWT metric."""
        bwt_matrix = np.tril(self.accuracy_matrix - np.diag(self.accuracy_matrix), k=-1)
        return np.sum(bwt_matrix) * 2 / (self.n * (self.n-1))

    def backward_transfer_pos(self):
        """from https://arxiv.org/abs/1810.13166. This is the BWT+ metric."""
        return max(0, self._backward_transfer())

    def remembering(self):
        """from https://arxiv.org/abs/1810.13166. This is the REM metric."""
        return 1 - abs(min(self._backward_transfer(), 0))

    def all_metrics(self):
        return dict(
            accuracy=self.accuracy(),
            forward_transfer=self.forward_transfer(),
            backward_transfer_pos=self.backward_transfer_pos(),
            remembering=self.remembering())


if __name__ == "__main__":
    # Usage example:
    # python3 clmetrics/transfer.py --config clmetrics/template_config.yml
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='config YAML: contains list of checkpoints and worlds')
    args = parser.parse_args()

    db_connection = sqlite3.connect(os.path.join(os.path.dirname(__file__), 'eval_results.sqlite'),
                                    detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)

    with open(args.config) as cf:
        config = yaml.safe_load(cf)
        set_seed(config['seed'], cuda_deterministic=True)
        accmatrix = make_accuracy_matrix(config)
    print(accmatrix, "\n")
    print(CLMetric(accmatrix))
