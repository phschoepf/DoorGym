from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer

from clfd.imitation_cl.model.hypernetwork import calc_delta_theta, calc_fix_target_reg, get_current_targets


class SACTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            log_alpha=None,
            alpha_optimizer=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.target_entropy = target_entropy

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if log_alpha:
            print(log_alpha._grad)
            # self.log_alpha = ptu.ones(1, requires_grad=True)
            # self.log_alpha = ptu.ones(1, requires_grad=True) * log_alpha.item()
            self.log_alpha = log_alpha
            print("trained log_alpha has been loaded",log_alpha, " alpha:",self.log_alpha.exp())
        else:
            self.log_alpha = ptu.zeros(1, requires_grad=True)

        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )
            # print(self.alpha_optimizer)
            if alpha_optimizer:
                self.alpha_optimizer.load_state_dict(alpha_optimizer.state_dict())
        else:
            self.alpha_optimizer = alpha_optimizer 

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
            # print("alpha",alpha.item())
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
            # self.eval_statistics['Target Entropy'] = self.target_entropy
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
            log_alpha=self.log_alpha,
            alpha_optimizer=self.alpha_optimizer,
        )


class HNSACTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            hnet,
            weight_shapes,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            task_id: int,
            beta: int = 5e-3,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            log_alpha=None,
            alpha_optimizer=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.target_entropy = target_entropy

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if log_alpha:
            print(log_alpha._grad)
            # self.log_alpha = ptu.ones(1, requires_grad=True)
            # self.log_alpha = ptu.ones(1, requires_grad=True) * log_alpha.item()
            self.log_alpha = log_alpha
            print("trained log_alpha has been loaded",log_alpha, " alpha:",self.log_alpha.exp())
        else:
            self.log_alpha = ptu.zeros(1, requires_grad=True)

        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )
            # print(self.alpha_optimizer)
            if alpha_optimizer:
                self.alpha_optimizer.load_state_dict(alpha_optimizer.state_dict())
        else:
            self.alpha_optimizer = alpha_optimizer

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.hnet = hnet
        self.weight_shapes = weight_shapes  # dict of the target network shapes
        self.task_id = task_id  # id of the active task
        self.beta = beta

        # generate new task embeddings if the task has not been seen by the HNs before
        if self.task_id > self.tasks_trained - 1:
            self.add_task()

        # calculate targets of past tasks for regularization
        if self.beta > 0:
            self.targets = get_current_targets(self.task_id, self.hnet)
        else:
            self.targets = None

        self.theta_optimizer = optimizer_class(list(self.hnet.theta), lr=policy_lr) # TODO different learning rates for policy and qf?
        # self.nonreg_optimizer = optimizer_class(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=qf_lr) # TODO for non-hnet qf (maybe later)
        self.emb_optimizer = optimizer_class([self.hnet.get_task_emb(self.task_id)], lr=policy_lr)

        # initialize target weights
        self._update_tnet_weights(initialize=True)

    @property
    def tasks_trained(self):
        return len(self.hnet.task_embs)

    def add_task(self) -> int:
        """
        Add a task to the hnet.

        Returns:
            new number of tasks know to the hnet.
        """
        self.hnet.gen_new_task_emb()
        return self.tasks_trained

    def set_active_task(self, task_id: int):
        self.task_id = task_id

    def _update_tnet_weights(self, initialize=False):
        """
        Forward the hnet to get weights and feed the weights into all target networks

        Args:
             initialize: If true, also sets the target_qf networks. Should only be used on initialization of the trainer.
        """
        generated_weights = self.hnet(self.task_id)
        # slice the weight list into chunks for each target network that the hnet manages
        sliced_weights = {}
        index = 0
        for key, shape in self.weight_shapes.items():
            sliced_weights[key] = generated_weights[index:index + len(shape)]
            index += len(shape)

        self.qf1.set_weights(sliced_weights['qf1'])
        self.qf2.set_weights(sliced_weights['qf2'])
        self.policy.set_weights([sliced_weights['fcs'], sliced_weights['last_fc'], sliced_weights['last_fc_logstd']])
        if initialize:
            self.target_qf1.set_weights(sliced_weights['qf1'])
            self.target_qf2.set_weights(sliced_weights['qf2'])

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        calc_reg = self.task_id > 0 and self.beta > 0

        """
        Set target network weights
        """
        self._update_tnet_weights()

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
            # print("alpha",alpha.item())
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(torch.cat((obs, new_obs_actions), dim=1)),
            self.qf2(torch.cat((obs, new_obs_actions), dim=1)),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(torch.cat((obs, actions), dim=1))
        q2_pred = self.qf2(torch.cat((obs, actions), dim=1))
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        target_q_values = torch.min(
            self.target_qf1(torch.cat((next_obs, new_next_actions), dim=1)),
            self.target_qf2(torch.cat((next_obs, new_next_actions), dim=1)),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.theta_optimizer.zero_grad()
        #self.nonreg_optimizer.zero_grad()
        self.emb_optimizer.zero_grad()

        # retain_graph has to be true since the computational graphs are connected via the hnet
        qf1_loss.backward(retain_graph=True, create_graph=False)
        qf2_loss.backward(retain_graph=True, create_graph=False)
        policy_loss.backward(retain_graph=True, create_graph=False)

        self.emb_optimizer.step()

        # Initialize the regularization loss
        loss_reg = 0

        # Initialize dTheta, the candidate change in the hnet parameters
        dTheta = None

        if calc_reg:
            # Find out the candidate change (dTheta) in trainable parameters (theta) of the hnet
            # This function just computes the change (dTheta), but does not apply it
            dTheta = calc_delta_theta(self.theta_optimizer, use_sgd_change=False, detach_dt=True)

            # Calculate the regularization loss using dTheta
            # This implements the second part of equation 2
            loss_reg = calc_fix_target_reg(self.hnet, self.task_id, targets=self.targets, dTheta=dTheta)

            # Multiply the regularization loss with the scaling factor
            loss_reg *= self.beta

            # Backpropagate the regularization loss
            loss_reg.backward()

        # Update the hnet params using the current task loss and the regularization loss
        self.theta_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to_tnet(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to_tnet(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
            # self.eval_statistics['Target Entropy'] = self.target_entropy
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.hnet,
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            hnet=self.hnet,
            weight_shapes=self.weight_shapes,
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
            log_alpha=self.log_alpha,
            alpha_optimizer=self.alpha_optimizer,
        )
