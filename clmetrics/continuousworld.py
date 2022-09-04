import argparse
import json
import logging
import math
import numpy as np
import os
import re
import yaml

from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import DoorGym.enjoy as enjoy
from DoorGym.a2c_ppo_acktr.arguments import add_common_args
from DoorGym.deterministic import set_seed


"""https://arxiv.org/pdf/2105.10919.pdf, Section 4.1"""

logger = logging.getLogger(__name__)

def get_iternum(chp):
    """get the iteration number from checkpoint path"""
    match = re.match(r".*\.(\d+)\.pt$", chp)
    return int(match.group(1))


def get_all_checkpoints(base_dir, maxiter=None) -> Dict[int, str]:
    """Get all checkpoints in a folder. If maxiter is specified, filter to only those up to
     and including a given iteration

     Returns:
         dict with iternums as keys and path to checkpoint as values"""

    if maxiter is None:
        maxiter = math.inf

    base_dir = os.path.expanduser(base_dir)
    all_files = [os.path.join(base_dir, file)
                 for file in os.listdir(base_dir)
                 if get_iternum(file) <= maxiter]
    d = {}
    for f in all_files:
        d[get_iternum(f)] = f
    return d


def run_eval(checkpoint, world, task_id):
    if MOCK:
        return int(np.random.uniform(0, 100))
    env_kwargs = dict(port=args.port,
                      visionnet_input=args.visionnet_input,
                      unity=args.unity,
                      world_path=world)

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
        task_id=task_id)
    logging.debug(json.dumps({"checkpoint": checkpoint, "world": world, "task_id": task_id, "opening_rate": opening_rate}))
    return opening_rate


@dataclass
class CLTimepoint:
    task_id: int
    checkpoint: int


class CLSeries:
    def __init__(self, config):
        # cl runs (hnppo with previous learned tasks)
        cl_checkpoint_folders = [os.path.join(config['checkpoint_root'], os.path.dirname(run['checkpoint']))
                                 for run in config['runs']]
        cl_highest_iters = [get_iternum(run['checkpoint']) for run in config['runs']]
        # list of dicts of checkpoints, one dict per trained task
        self.cl_checkpoints = [get_all_checkpoints(checkpoint_folder, maxiter=highest)
                               for checkpoint_folder, highest in zip(cl_checkpoint_folders, cl_highest_iters)]

        # reference runs (hnppo with fresh networks)
        try:
            ref_checkpoint_folders = [os.path.join(config['checkpoint_root'], os.path.dirname(run['checkpoint']))
                                      for run in config['reference_runs']]
            ref_highest_iters = [get_iternum(run['checkpoint']) for run in config['reference_runs']]
            self.reference_checkpoints = [get_all_checkpoints(checkpoint_folder, maxiter=highest)
                                          for checkpoint_folder, highest in zip(ref_checkpoint_folders, ref_highest_iters)]
            assert len(self.cl_checkpoints) == len(self.reference_checkpoints)
        except KeyError:
            # Reference checkpoints are only needed for forward transfer metric, so not giving them is fine for the rest
            self.reference_checkpoints = None

        self.worlds = [os.path.join(os.path.expanduser(config['world_root']), run['world'])
                       for run in config['runs']]

    def avg_performance(self, t: CLTimepoint):
        """Average success rate of all tasks at time t. The latest task id is used for tasks that have not
        been trained at this point."""

        checkpoint = self.cl_checkpoints[t.task_id][t.checkpoint]

        accuracies = []
        for eval_tid, world in enumerate(self.worlds):
            tid_to_evaluate = min(t.task_id, eval_tid)

            print(f'AVG: evaluating task {eval_tid} on checkpoint of task {t.task_id} (world {os.path.basename(world)})')
            opening_rate = run_eval(checkpoint, world, tid_to_evaluate)
            accuracies.append(np.clip(opening_rate / 100, a_min=0, a_max=1))
        return np.mean(accuracies)

    def forward_transfer(self):
        def _get_auc(checkpoints, task_id):
            accuracies = []
            for checkpoint in checkpoints:
                print(f'FT: evaluating task {task_id} on checkpoint {os.path.basename(checkpoint)} (world {os.path.basename(self.worlds[task_id])})')
                opening_rate = run_eval(checkpoint, self.worlds[task_id], task_id)
                accuracies.append(np.clip(opening_rate / 100, a_min=0, a_max=1))
            return np.mean(accuracies)

        if self.reference_checkpoints is None:
            logging.warning('Forward transfer was skipped, no reference runs given')
            return -1
        task_fts = []
        # Task-wise forward transfers
        for task_id in range(len(self.cl_checkpoints)):
            cl_auc = _get_auc(self.cl_checkpoints[task_id].values(), task_id)
            ref_auc = _get_auc(self.reference_checkpoints[task_id].values(), task_id)

            task_fts.append((cl_auc-ref_auc) / (1-ref_auc))
        logging.debug(f'Task-wise forward transfers: {task_fts}')
        return np.mean(task_fts)

    def forgetting(self):
        task_forgettings = []
        # Task-wise forgetting metric
        for task_id in range(len(self.cl_checkpoints)):
            last_chp_task = self.cl_checkpoints[task_id][max(self.cl_checkpoints[task_id])]
            last_chp_total = self.cl_checkpoints[-1][max(self.cl_checkpoints[-1])]
            print(f'Forgetting: evaluating task {task_id} on checkpoint after training (world {os.path.basename(self.worlds[task_id])})')
            after_training = run_eval(last_chp_task, self.worlds[task_id], task_id)
            print(f'Forgetting: evaluating task {task_id} on final checkpoint (world {os.path.basename(self.worlds[task_id])})')
            at_end = run_eval(last_chp_total, self.worlds[task_id], task_id)

            task_forgettings.append(np.clip(after_training / 100, a_min=0, a_max=1) -
                                    np.clip(at_end / 100, a_min=0, a_max=1))
        logging.debug(f'Task-wise forgetting: {task_forgettings}')
        return np.mean(task_forgettings)

    def all_metrics(self, t: CLTimepoint = None):
        # default is last checkpoint of last task
        if t is None:
            t = CLTimepoint(task_id=len(self.cl_checkpoints)-1,
                            checkpoint=max(self.cl_checkpoints[-1]))
        return dict(
            avg_performance=self.avg_performance(t),
            forward_transfer=self.forward_transfer(),
            forgetting=self.forgetting())

    def __call__(self, *args, **kwargs):
        return self.all_metrics(kwargs.get('t'))


if __name__ == "__main__":
    MOCK = True
    # Usage example:
    # python3 clmetrics/transfer.py --config clmetrics/template_config.yml
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='config YAML: contains list of checkpoints and worlds')
    args = parser.parse_args()

    with open(args.config) as cf:
        config = yaml.safe_load(cf)
        set_seed(config['seed'], cuda_deterministic=True)
        series = CLSeries(config)

    print(series.all_metrics())
    #print(series.all_metrics(t=CLTimepoint(task_id=0, checkpoint=20)))
