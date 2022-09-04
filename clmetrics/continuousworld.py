import argparse
import json
import logging
import math
import numpy as np
import os
import re
import sqlite3
import yaml

from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import DoorGym.enjoy as enjoy
from DoorGym.a2c_ppo_acktr.arguments import add_common_args
from DoorGym.deterministic import set_seed


"""https://arxiv.org/pdf/2105.10919.pdf, Section 4.1"""

logger = logging.getLogger()

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


def run_eval(checkpoint, world, task_id, db_connection: sqlite3.Connection, args: argparse.Namespace):
    # Check DB for preexiting value, reuse if it exists
    cur = db_connection.cursor()
    res = cur.execute("SELECT opening_rate FROM evals WHERE checkpoint = ? AND world = ? AND task_id = ?",
                      (checkpoint, world, task_id))
    opening_rate = res.fetchone()
    if opening_rate is not None:
        logger.info(f"using cached opening rate for {os.path.basename(checkpoint)}")
        return opening_rate[0]

    # if value does not exist, run evaluation and save in DB
    else:
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
        opening_rate_normalized = np.clip(opening_rate / 100, a_min=0, a_max=1)
        # Doorgym opening rate is in percent, convert it to a ratio. Also, clip to [0,1] because sometimes the
        # reported opening rate will be something like 101%, which is a bug due to asynchronous calls to mujoco

        timestamp = datetime.now()
        cur.execute("INSERT INTO evals(checkpoint, world, task_id, opening_rate, opening_timeavg, episode_rewards_avg, ts) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?)",
                    (checkpoint, world, task_id, opening_rate_normalized, opening_timeavg, float(episode_rewards_avg), timestamp))
        db_connection.commit()
        return opening_rate_normalized


@dataclass
class CLTimepoint:
    task_id: int
    checkpoint: int


class CLSeries:
    def __init__(self, config, db: sqlite3.Connection, doorgym_args: argparse.Namespace):
        # cl runs (hnppo with previous learned tasks)
        cl_checkpoint_folders = [os.path.join(config['checkpoint_root'], os.path.dirname(run['checkpoint']))
                                 for run in config['runs']]
        cl_highest_iters = [get_iternum(run['checkpoint']) for run in config['runs']]
        # list of dicts of checkpoints, one dict per trained task
        self.cl_checkpoints = [get_all_checkpoints(checkpoint_folder, maxiter=highest)
                               for checkpoint_folder, highest in zip(cl_checkpoint_folders, cl_highest_iters)]

        # reference runs (hnppo with fresh networks)
        try:
            ref_checkpoint_folders = [os.path.join(config['checkpoint_root'], os.path.dirname(run['ref_checkpoint']))
                                      for run in config['runs']]
            ref_highest_iters = [get_iternum(run['ref_checkpoint']) for run in config['runs']]
            self.reference_checkpoints = [get_all_checkpoints(checkpoint_folder, maxiter=highest)
                                          for checkpoint_folder, highest in zip(ref_checkpoint_folders, ref_highest_iters)]
            assert len(self.cl_checkpoints) == len(self.reference_checkpoints)
        except KeyError:
            # Reference checkpoints are only needed for forward transfer metric, so not giving them is fine for the rest
            self.reference_checkpoints = None

        self.worlds = [os.path.join(os.path.expanduser(config['world_root']), run['world'])
                       for run in config['runs']]
        self.db = db
        self.args = doorgym_args

    def avg_performance(self, t: CLTimepoint):
        """Average success rate of all tasks at time t. The latest task id is used for tasks that have not
        been trained at this point."""

        checkpoint = self.cl_checkpoints[t.task_id][t.checkpoint]

        accuracies = []
        for eval_tid, world in enumerate(self.worlds):
            tid_to_evaluate = min(t.task_id, eval_tid)

            logger.info(f'AVG: evaluating task {eval_tid} on checkpoint of task {t.task_id} (world {os.path.basename(world)})')
            opening_rate = run_eval(checkpoint, world, tid_to_evaluate, self.db, self.args)
            accuracies.append(opening_rate)
        return np.mean(accuracies)

    def forward_transfer(self):
        def _get_auc(checkpoints, task_id):
            accuracies = []
            for checkpoint in checkpoints:
                logger.info(f'FT: evaluating task {task_id} on checkpoint {os.path.basename(checkpoint)} (world {os.path.basename(self.worlds[task_id])})')
                opening_rate = run_eval(checkpoint, self.worlds[task_id], task_id, self.db, self.args)
                accuracies.append(opening_rate)
            return np.mean(accuracies)

        if self.reference_checkpoints is None:
            logger.warning('Forward transfer was skipped, no reference runs given')
            return None
        task_fts = []
        # Task-wise forward transfers
        for task_id in range(len(self.cl_checkpoints)):
            cl_auc = _get_auc(self.cl_checkpoints[task_id].values(), task_id)
            ref_auc = _get_auc(self.reference_checkpoints[task_id].values(), task_id)

            task_fts.append((cl_auc-ref_auc) / (1-ref_auc))
        logger.debug(f'Task-wise forward transfers: {task_fts}')
        return np.mean(task_fts)

    def forgetting(self):
        task_forgettings = []
        # Task-wise forgetting metric
        for task_id in range(len(self.cl_checkpoints)):
            last_chp_task = self.cl_checkpoints[task_id][max(self.cl_checkpoints[task_id])]
            logger.info(f'Forgetting: evaluating task {task_id} on checkpoint after training (world {os.path.basename(self.worlds[task_id])})')
            after_training = run_eval(last_chp_task, self.worlds[task_id], task_id, self.db, self.args)

            last_chp_total = self.cl_checkpoints[-1][max(self.cl_checkpoints[-1])]
            logger.info(f'Forgetting: evaluating task {task_id} on final checkpoint (world {os.path.basename(self.worlds[task_id])})')
            at_end = run_eval(last_chp_total, self.worlds[task_id], task_id, self.db, self.args)

            task_forgettings.append(after_training - at_end)
        logger.debug(f'Task-wise forgetting: {task_forgettings}')
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
    # Usage example:
    # python3 clmetrics/continuousworld.py --config clmetrics/template_config.yml
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='config YAML: contains list of checkpoints and worlds')
    args = parser.parse_args()

    logging.basicConfig(filename=os.path.join(os.path.dirname(__file__), 'log', os.path.splitext(os.path.basename(args.config))[0]+'.log'),
                        filemode='w',
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    db_connection = sqlite3.connect(os.path.join(os.path.dirname(__file__), 'eval_results.sqlite'),
                                    detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)

    with open(args.config) as cf:
        config = yaml.safe_load(cf)
        set_seed(config['seed'], cuda_deterministic=True)
        series = CLSeries(config, db_connection, args)

    res = series.all_metrics()
    #print(series.all_metrics(t=CLTimepoint(task_id=0, checkpoint=20)))
    logger.info(res)
    print(res)
