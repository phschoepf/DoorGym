import numpy as np
import torch
from torch import nn as nn

from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.core import eval_np
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.networks import Mlp

from clfd.imitation_cl.model.hypernetwork import TargetNetwork

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class TanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.visionnet_input = False
        self.visionmodel = None
        self.nn = 6
        self.knob_noisy = False
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )

class MakeDeterministic(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy
        self.visionnet_input = stochastic_policy.visionnet_input
        self.visionmodel = stochastic_policy.visionmodel
        self.nn = stochastic_policy.nn

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)


class TanhGaussianHnetPolicy(nn.Module, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__()
        self.visionnet_input = False
        self.visionmodel = None
        self.nn = 6
        self.knob_noisy = False
        self.log_std = None
        self.std = std

        last_hidden_size = obs_dim
        if len(hidden_sizes) > 0:
            last_hidden_size = hidden_sizes[-1]

        # Common layers
        self.fcs = TargetNetwork(n_in=obs_dim,
                                 n_out=last_hidden_size,
                                 hidden_layers=hidden_sizes[:-1],
                                 no_weights=True,
                                 bn_track_stats=False,
                                 activation_fn=torch.nn.ReLU(),
                                 out_fn=torch.nn.ReLU(),
                                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.last_fc = TargetNetwork(n_in=last_hidden_size,
                                     n_out=action_dim,
                                     hidden_layers=[],
                                     no_weights=True,
                                     bn_track_stats=False,
                                     activation_fn=None,
                                     out_fn=None,
                                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if std is None:
            self.last_fc_log_std = TargetNetwork(n_in=last_hidden_size,
                                                 n_out=action_dim,
                                                 hidden_layers=[],
                                                 no_weights=True,
                                                 bn_track_stats=False,
                                                 activation_fn=None,
                                                 out_fn=None,
                                                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def set_weights(self, weights):
        """
        Set weights of the policy's target networks. 3 weights must be provided:

        * fcs_weights [[obs_dim x hidden_size] ..layers.. [hidden_size]]
        * last_fc_weights [[hidden_size x action_dim], [action_dim]]
        * last_fc_logstd_weights [[hidden_size x action_dim], [action_dim]]
        """
        self.fcs.set_weights(weights[0])
        self.last_fc.set_weights(weights[1])
        self.last_fc_log_std.set_weights(weights[2])

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h, _ = self.fcs(obs)
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )
