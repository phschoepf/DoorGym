import argparse
import json
import os
import yaml

import numpy as np
import DoorGym.enjoy as enjoy
from DoorGym.a2c_ppo_acktr.arguments import add_common_args
from DoorGym.deterministic import set_seed

def make_accuracy_matrix(config):
    """Make the accuracy matrix of a given run. The accuracy matrix is an NxN array for N tasks trained on the hnet.
    amatrix[A,B] is the accuracy of task B, evaluated on the network after training on task A."""

    experiments, worlds = zip(*[(os.path.join(os.path.expanduser(config['checkpoint_root']), run['checkpoint']),
                                 os.path.join(os.path.expanduser(config['world_root']), run['world']))
                                for run in config['runs']])

    assert len(experiments) == len(worlds), "Number of experiments and world paths has to be the same"
    accuracy_mat = np.zeros((len(experiments), len(experiments)))

    # iterate over checkpoint files
    for train_tid, checkpoint in enumerate(experiments):
        # iterate over tasks
        for eval_tid, world in enumerate(worlds):

            # we can't give tids higher than the train_tid to enjoy.py, thus we clip it to the highest tid for all other tasks
            # evaluating on not-yet-trained tasks can reveal zero-shot capabilities
            tid_to_evaluate = min(train_tid, eval_tid)

            env_kwargs = dict(port=args.port,
                              visionnet_input=args.visionnet_input,
                              unity=args.unity,
                              world_path=world)

            print(f'evaluating task {eval_tid} on checkpoint of task {train_tid} (world {os.path.basename(world)})')
            opening_rate, opening_timeavg, episode_rewards_avg = enjoy.onpolicy_inference(
                seed=args.seed,
                env_name=args.env_name,
                det=True,
                load_name=checkpoint,
                evaluation=True,
                render=False,
                knob_noisy=args.knob_noisy,
                visionnet_input=args.visionnet_input,
                env_kwargs=env_kwargs,
                actor_critic=None,
                verbose=False,
                pos_control=args.pos_control,
                step_skip=args.step_skip,
                task_id=tid_to_evaluate)
            accuracy_mat[train_tid, eval_tid] = opening_rate

    return accuracy_mat


class CLMetric:
    def __init__(self, amatrix: np.ndarray = None):
        if amatrix is None:
            self.accuracy_matrix = make_accuracy_matrix()
        else:
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
    parser.add_argument(
        '-m', '--matrix',
        type=str,
        help='precalculated accuracy matrix')
    args = parser.parse_args()

    # if matrix is given, just print the CL metrics
    if args.matrix:
        with open(args.matrix) as mf:
            accmatrix = np.array(json.load(mf))

    # if not, calculate the matrix (slow!) and save it for later
    else:
        with open(args.config) as cf:
            config = yaml.safe_load(cf)
            set_seed(config['seed'], cuda_deterministic=True)
            accmatrix = make_accuracy_matrix(config)
        with open(args.config.replace("config.yml", "accuracy_matrix.json"), mode='w') as mf:
            print(f'writing matrix to {mf.name}')
            json.dump(accmatrix.tolist(), mf)

    print(CLMetric(accmatrix))
