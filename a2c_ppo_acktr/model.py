import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from clfd.imitation_cl.model.hypernetwork import HyperNetwork, ChunkedHyperNetwork, TargetNetwork

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian, FunctionalDiagGaussian
from a2c_ppo_acktr.utils import init
from typing import List

logits_input = False
knob_pos_smoothening = False

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

def gen_layers(num_inputs: int, hidden_size: List[int], activation_fn=nn.Tanh):
    """Generate a list of modules for given input size, hidden sizes and activation function. Output layer is omitted."""
    layer_size = [num_inputs] + hidden_size
    modules_list = []
    for i in range(len(layer_size) - 1):
        modules_list.append(init_(nn.Linear(layer_size[i], layer_size[i+1]))),
        modules_list.append(activation_fn())
    return modules_list

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.tt = 0
        self.nn = 0
        self.visionmodel = None
        self.knob_target_hist = torch.zeros(1,3).cuda()
        self.base = base(obs_shape[0], **base_kwargs)

        if type(self.base) == HNBase:
            if not action_space.__class__.__name__ == "Box":
                raise NotImplementedError("Hypernetwork dist only implemented for Box")
            self.dist = self.base.dist
        else:
            if action_space.__class__.__name__ == "Discrete":
                num_outputs = action_space.n
                self.dist = Categorical(self.base.output_size, num_outputs)
            elif action_space.__class__.__name__ == "Box":  # this is being used in doorenv-v0
                num_outputs = action_space.shape[0]
                self.dist = DiagGaussian(self.base.output_size, num_outputs)
            elif action_space.__class__.__name__ == "MultiBinary":
                num_outputs = action_space.shape[0]
                self.dist = Bernoulli(self.base.output_size, num_outputs)
            else:
                raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

    def obs2img_vec(self, inputs):
        img_size = 256
        joints_nn = self.nn
        joints = inputs[:,:joints_nn*2]
        finger_tip_target = inputs[:,joints_nn*2:joints_nn*2+3]
        img_front = inputs[:,joints_nn*2+3:-3*img_size*img_size].view(-1, 3, img_size, img_size)
        img_top   = inputs[:,-3*img_size*img_size:].view(-1, 3, img_size, img_size)
        return joints, finger_tip_target, img_front, img_top

    def obs2inputs(self, inputs, epoch):
        joints, finger_tip_target, img_front, img_top = self.obs2img_vec(inputs)

        with torch.no_grad():
            pp, hm1, hm2 = self.visionmodel(img_top, img_front)        
        knob_target = pp
        dist_vec = finger_tip_target - knob_target

        inputs = torch.cat((joints, dist_vec), 1)
        return inputs

class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size: List[int]):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size[0])
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size[0]
        return 1

    @property
    def output_size(self):
        return self._hidden_size[-1]

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=[64, 64], **kwargs):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size[0]

        self.actor = nn.Sequential(*gen_layers(num_inputs, hidden_size))
        self.critic = nn.Sequential(*gen_layers(num_inputs, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size[-1], 1))
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class HNBase(NNBase):
    def __init__(self, num_inputs, action_space, hnet:HyperNetwork, recurrent=False, hidden_size=[64, 64], te_dim=8):
        """
        Args:
            num_inputs: Number of input (env observations)
            action_space: Action space of the env. Needed to infer output shape.
            hnet: Class of hypernet to use. Must be a subclass of "HyperNetwork". Currently only HN or ChunkedHN.
            recurrent: Always false for HN
            hidden_size: Hidden units per layer in the target network.
        """
        super(HNBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size[0]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dims_a = TargetNetwork.weight_shapes(n_in=num_inputs, n_out=hidden_size[-1], hidden_layers=hidden_size[:-1])
        num_outputs = action_space.shape[0]
        self.output_dims_dist = [[num_outputs, self.output_size], [num_outputs], [num_outputs]]

        # Common param dict for the hypernetwork. We filter what we need for the particular hypernetwork types.
        hparams = {
            'target_shapes': self.output_dims_a + self.output_dims_dist,
            'layers': [hidden_size[0] * 10] * 2,  # hnet is currently fixed at 2 layers deep
            'te_dim': te_dim,
            'chunk_dim': 1000,
            'ce_dim': 5,
            'device': device
        }

        if hnet == HyperNetwork:
            self.hnet = hnet(
                hparams['target_shapes'],
                layers=hparams['layers'],
                te_dim=hparams['te_dim'],
                device=hparams['device'])

        elif hnet == ChunkedHyperNetwork:
            self.hnet = hnet(
                hparams['target_shapes'],
                layers=hparams['layers'],
                te_dim=hparams['te_dim'],
                chunk_dim=hparams['chunk_dim'],
                ce_dim=hparams['ce_dim'],
                device=hparams['device'])

        else:
            raise NotImplementedError(f"Unkown hnet class {hnet.__class__.__name__}")

        self.actor = TargetNetwork(
                         n_in=num_inputs,
                         n_out=hidden_size[-1],
                         hidden_layers=hidden_size[:-1],
                         no_weights=True,
                         bn_track_stats=False,
                         activation_fn=torch.nn.Tanh(),
                         out_fn=torch.nn.Tanh(),
                         device=device)

        # critic is just a normal nn.Sequential, only used for training (copied from MLPBase)
        self.critic = nn.Sequential(*gen_layers(num_inputs, hidden_size), init_(nn.Linear(hidden_size[-1], 1)))
        # dist was also moved here from policy so we can populate it with weights from the HN
        self.dist = FunctionalDiagGaussian(self.output_size, num_outputs)

        self.tasks_trained = 0
        self.active_task = None
        self.train()

    def add_task(self) -> int:
        """
        Add a task to the hnet.

        Returns:
            new number of tasks know to the hnet.
        """
        self.tasks_trained += 1
        self.hnet.gen_new_task_emb()
        return self.tasks_trained

    def reset_critic(self):
        print("Critic reset")
        def _res(module):
            if type(module) == nn.Linear:
                module = init_(module)

        self.critic.apply(_res)

    def set_active_task(self, task_id: int):
        # reset critic if the task id changes (i.e. on training next task)
        if task_id != self.active_task and self.active_task is not None:
           self.reset_critic()
        self.active_task = task_id

    def forward(self, inputs, rnn_hxs, masks):
        # generate weights for both networks, as a list, then split the list to populate the networks' parameters
        generated_weights = self.hnet(self.active_task)
        self.actor.set_weights(generated_weights[:len(self.output_dims_a)])
        self.dist.set_weights(generated_weights[len(self.output_dims_a):])

        hidden_critic = self.critic(inputs)
        hidden_actor, _ = self.actor(inputs)
        # we do not forward the dist here, this is done in Policy.act()

        return hidden_critic, hidden_actor, rnn_hxs
