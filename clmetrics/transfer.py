import os
import re

import numpy as np
import DoorGym.enjoy as enjoy
from DoorGym.a2c_ppo_acktr.arguments import get_args_enjoy


def parse_loadname(load_name: str):
    """
    Find all folders with a given prefix, and returns information about their experiment.

    Returns:
        list of tuples (full_folderpath, correspondig_world_name, task_id) for all experiments with the given prefix
    """
    folderpat = re.compile(r'.*?-task([0-9])-([a-z]+).*?')
    info = []
    dirname, prefix = os.path.split(load_name)
    for folder in os.listdir(dirname):
        if folder.startswith(prefix):
            full_path = os.path.join(dirname, folder)
            m = re.match(folderpat, folder)
            tid = int(m.group(1))
            worldname = m.group(2) + "_blue_gripper"
            info.append((full_path, worldname, tid))
    return info


def make_accuracy_matrix():
    args = get_args_enjoy()

    experiments = parse_loadname(args.load_name)
    accuracy_mat = np.zeros((len(experiments), len(experiments)))
    for folder, world, train_tid in sorted(experiments, key=lambda e: e[2]):
        # iterate over checkpoint files
        checkpoint_to_load = os.path.join(folder,
                                          sorted(os.listdir(folder))[-1])  # get the latest checkpoint from the folder

        for eval_tid in range(len(experiments)):
            # iterate over tasks

            # load the world corresponding to this task
            world_to_load = os.path.join(args.world_path, experiments[eval_tid][1])

            # we can't give tids higher than the train_tid to enjoy.py, thus we clip it to the highest tid for all other tasks
            # evaluating on not-yet-trained tasks can reveal zero-shot capabilities
            tid_to_evaluate = min(train_tid, eval_tid)

            env_kwargs = dict(port=args.port,
                              visionnet_input=args.visionnet_input,
                              unity=args.unity,
                              world_path=world_to_load)

            print(
                f'evaluating task {eval_tid} on checkpoint of task {train_tid} (world {world_to_load.split("/")[-1]})')
            opening_rate, opening_timeavg = enjoy.onpolicy_inference(
                seed=args.seed,
                env_name=args.env_name,
                det=True,
                load_name=checkpoint_to_load,
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

    def backward_transfer(self):
        """from https://arxiv.org/abs/1810.13166. This is the BWT+ metric."""
        bwt_matrix = np.tril(self.accuracy_matrix - np.diag(self.accuracy_matrix), k=-1)
        return max(0, np.sum(bwt_matrix) * 2 / (self.n * (self.n-1)))

    def remembering(self):
        """from https://arxiv.org/abs/1810.13166. This is the REM metric."""
        bwt_matrix = np.tril(self.accuracy_matrix - np.diag(self.accuracy_matrix), k=-1)
        return 1 - abs(min(np.sum(bwt_matrix) * 2 / (self.n * (self.n-1)), 0))

    def all_metrics(self):
        return dict(
            accuracy=self.accuracy(),
            forward_transfer=self.forward_transfer(),
            backward_transfer_pos=self.backward_transfer(),
            remembering=self.remembering())


if __name__ == "__main__":
    # Usage example:
    # python3 clmetrics/transfer.py --env-name doorenv-v0 --world-path ~/Desktop/schoepf-bachelor-thesis/DoorGym/world_generator/world  --load-name trained_models/hnppo/doorenv-v0_ppo-hn7
    accmatrix = make_accuracy_matrix()

    # for debug purposes
    testmatrix = np.array([[1.0, 0.5, 0.0],
                           [1.0, 0.9, 0.1],
                           [0.6, 0.7, 0.5]]
                          )

    print(CLMetric(accmatrix))
