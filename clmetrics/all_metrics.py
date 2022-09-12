import argparse
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


@dataclass
class CLTimepoint:
    task_id: int
    checkpoint: int


class CLSeries:
    """Base class for evaluations. Represents a set of CL runs and their corresponding baseline runs.
    Also has a DB connection for saving results and the interface to DoorGym."""

    def __init__(self, config, db: sqlite3.Connection, doorgym_args: argparse.Namespace):
        # cl runs (hnppo with previous learned tasks)
        self.cl_checkpoints = []
        self.reference_checkpoints = []
        for run in config['runs']:
            cl_folder = os.path.join(config['checkpoint_root'], os.path.dirname(run['checkpoint']))
            cl_highest_iter = get_iternum(run['checkpoint'])
            cl_checkpoint_dict = get_all_checkpoints(cl_folder, maxiter=cl_highest_iter)
            self.cl_checkpoints.append(cl_checkpoint_dict)

            try:
                ref_folder = os.path.join(config['checkpoint_root'], os.path.dirname(run['ref_checkpoint']))
                ref_highest_iter = get_iternum(run['ref_checkpoint'])
                ref_checkpoint_dict = get_all_checkpoints(ref_folder, maxiter=ref_highest_iter)
                self.reference_checkpoints.append(ref_checkpoint_dict)

            except KeyError:
                # Reference checkpoints are only needed for forward transfer metric, so not giving them is fine for the rest
                logger.warning(f'Missing reference checkpoints for {run.get("world")}')

        # Sanity check if all reference dirs exist and checkpoint numbers (=dict keys) are the same. Otherwise this
        # will produce inaccurate forward transfer metric.
        if not (
                len(self.cl_checkpoints) == len(self.reference_checkpoints) and
                all(cl.keys() == ref.keys() for cl, ref in zip(self.cl_checkpoints, self.reference_checkpoints))
        ):
            self.reference_checkpoints = None

        self.worlds = [os.path.join(os.path.expanduser(config['world_root']), run['world'])
                       for run in config['runs']]
        self.db = db
        self.args = doorgym_args

    def run_eval(self, checkpoint, world, task_id: int):
        """Run DoorGym evaluation and save the result in a database. Will re-use saved results on subsequent calls with
           the same (checkpoint, world, task_id).
        Args:
            checkpoint: Path to the model file to evaluate
            world: World folder to sample worlds from
            task_id: ID of the task to evaluate
        Returns:
            door opening success rate [0,1]"""

        args = self.args
        logger.info(f'evaluating {os.path.basename(checkpoint)} on world {os.path.basename(world)} with tid {task_id}')

        # Check DB for preexiting value, reuse if it exists
        cur = self.db.cursor()
        res = cur.execute("SELECT opening_rate FROM evals WHERE checkpoint = ? AND world = ? AND task_id = ?",
                          (checkpoint, world, task_id))
        opening_rate = res.fetchone()
        if opening_rate is not None:
            logger.debug(f'reusing saved result')
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
            cur.execute(
                "INSERT INTO evals(checkpoint, world, task_id, opening_rate, opening_timeavg, episode_rewards_avg, ts) "
                "VALUES(?, ?, ?, ?, ?, ?, ?)",
                (checkpoint, world, task_id, opening_rate_normalized, opening_timeavg, float(episode_rewards_avg),
                 timestamp))
            self.db.commit()
            return opening_rate_normalized


class CWMetrics(CLSeries):
    """Implements the metrics from the Contiual World paper https://arxiv.org/pdf/2105.10919.pdf, Section 4.1"""

    def __init__(self, config, db: sqlite3.Connection, doorgym_args: argparse.Namespace):
        super().__init__(config, db, doorgym_args)

    def avg_performance(self, t: CLTimepoint, **kwargs):
        """Average success rate of all tasks at time t. The latest task id is used for tasks that have not
        been trained at this point."""

        checkpoint = self.cl_checkpoints[t.task_id][t.checkpoint]

        accuracies = []
        for eval_tid, world in enumerate(self.worlds):
            tid_to_evaluate = min(t.task_id, eval_tid)
            opening_rate = self.run_eval(checkpoint, world, tid_to_evaluate)
            accuracies.append(opening_rate)
        return np.mean(accuracies)

    def forward_transfer(self, ignore_missing_ref=False, **kwargs):
        """Forward transfer of the learning curve. Compares AUC of the CL learning curve with the reference method.

        Args:
            ignore_missing_ref: Continue even if the reference runs are missing/incomplete. Will not
              return a valid forward transfer metric, but already evaluates all possible runs and caches
              them to speed up re-computation once all runs are ready."""
        def _get_auc(checkpoints, task_id):
            accuracies = []
            for checkpoint in checkpoints:
                opening_rate = self.run_eval(checkpoint, self.worlds[task_id], task_id)
                accuracies.append(opening_rate)
            return np.mean(accuracies)

        if self.reference_checkpoints is None:
            logger.warning('Forward transfer was skipped, no reference runs given')
            if ignore_missing_ref:
                for task_id in range(len(self.cl_checkpoints)):
                    logger.warning(f'evaluating forward transfer in caching-only mode: task {task_id}')
                    _ = _get_auc(self.cl_checkpoints[task_id].values(), task_id)
            return None
        task_fts = []
        # Task-wise forward transfers
        for task_id in range(len(self.cl_checkpoints)):
            logger.info(f'evaluating forward transfer: task {task_id}')
            cl_auc = _get_auc(self.cl_checkpoints[task_id].values(), task_id)
            ref_auc = _get_auc(self.reference_checkpoints[task_id].values(), task_id=0)

            task_fts.append((cl_auc-ref_auc) / (1-ref_auc))
        logger.debug(f'Task-wise forward transfers: {task_fts}')
        return np.mean(task_fts)

    def forgetting(self):
        task_forgettings = []
        # Task-wise forgetting metric
        for task_id in range(len(self.cl_checkpoints)):
            logger.info(f'evaluating forgetting: task {task_id}')
            last_chp_task = self.cl_checkpoints[task_id][max(self.cl_checkpoints[task_id])]
            after_training = self.run_eval(last_chp_task, self.worlds[task_id], task_id)

            last_chp_total = self.cl_checkpoints[-1][max(self.cl_checkpoints[-1])]
            at_end = self.run_eval(last_chp_total, self.worlds[task_id], task_id)

            task_forgettings.append(after_training - at_end)
        logger.debug(f'Task-wise forgetting: {task_forgettings}')
        return np.mean(task_forgettings)

    def all_metrics(self, **kwargs):
        # default is last checkpoint of last task
        if kwargs.get('t') is None:
            kwargs['t'] = CLTimepoint(task_id=len(self.cl_checkpoints)-1,
                                      checkpoint=max(self.cl_checkpoints[-1]))
        return dict(
            avg_performance=self.avg_performance(**kwargs),
            forward_transfer=self.forward_transfer(**kwargs),
            forgetting=self.forgetting())


class MTFMetrics(CLSeries):
    """Implements the metrics from the More Than Forgetting paper https://arxiv.org/abs/1810.13166"""

    def __init__(self, config, db: sqlite3.Connection, doorgym_args: argparse.Namespace):
        super().__init__(config, db, doorgym_args)
        self.accuracy_matrix = self._make_accuracy_matrix()
        self.n = self.accuracy_matrix.shape[0]

    def _make_accuracy_matrix(self):
        """Make the accuracy matrix of a given run. The accuracy matrix is an NxN array for N tasks trained on the hnet.
        amatrix[A,B] is the accuracy of task B, evaluated on the network after training on task A."""

        experiments = [chp[max(chp)] for chp in self.cl_checkpoints]

        accuracy_mat = np.zeros((len(experiments), len(experiments)))

        # iterate over checkpoint files
        for train_tid, checkpoint in enumerate(experiments):
            # iterate over tasks
            for eval_tid, world in enumerate(self.worlds):
                # we can't give tids higher than the train_tid to enjoy.py, thus we clip it to the highest tid for all other tasks
                # evaluating on not-yet-trained tasks can reveal zero-shot capabilities
                tid_to_evaluate = min(train_tid, eval_tid)

                opening_rate = self.run_eval(checkpoint, world, tid_to_evaluate)
                accuracy_mat[train_tid, eval_tid] = opening_rate

        return accuracy_mat

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


class CLTimelinePlot(CLSeries):
    """Generate evaluations for the timeline plot. Does not output anything, just rus the necessary evals and
    stores them in the DB."""
    def __init__(self, config, db: sqlite3.Connection, doorgym_args: argparse.Namespace):
        super().__init__(config, db, doorgym_args)

    def run_evals(self, chp_skip=1):
        for task_id, world in enumerate(self.worlds):
            for chp_dict in self.cl_checkpoints[task_id:]:
                total_step = 0
                for chp in chp_dict.values():
                    if total_step % chp_skip == 0:
                        self.run_eval(chp, world, task_id)
                    total_step += 1


if __name__ == "__main__":
    # Usage example:
    # python3 clmetrics/all_metrics.py --config clmetrics/template_config.yml
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

    """logger.info('Calculating MTF metrics....')
    mtf = MTFMetrics(config, db_connection, args)
    mtf_res = mtf.all_metrics()
    logger.info(mtf_res)
    print('MTF metrics:', mtf_res)

    logger.info('Calculating CW metrics....')
    cw = CWMetrics(config, db_connection, args)
    cw_res = cw.all_metrics(ignore_missing_ref=True)
    logger.info(cw_res)
    print('CW metrics:', cw_res)"""

    clplot_eval_runner = CLTimelinePlot(config, db_connection, args)
    clplot_eval_runner.run_evals(chp_skip=4)

    #print(series.all_metrics(t=CLTimepoint(task_id=0, checkpoint=20)))
