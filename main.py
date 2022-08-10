import os
import re
import sys
import time
import pickle
from collections import deque
import gym
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from trained_visionmodel.visionmodel import VisionModelXYZ
from enjoy import onpolicy_inference, offpolicy_inference
from util import add_vision_noise, add_joint_noise,load_visionmodel, prepare_trainer, prepare_env

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy, HNBase
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import doorenv
import doorenv2

from clfd.imitation_cl.model.hypernetwork import HyperNetwork, ChunkedHyperNetwork
import logging

logging.basicConfig(level='WARN')
logger = logging.getLogger()

### DEBUG ###
# torch.autograd.set_detect_anomaly(True)

def set_seed(seed=1000, cuda_deterministic=False):
    """
    Sets the seed for reproducability

    Args:
        seed (int, optional): Input seed. Defaults to 1000.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cuda_available = args.cuda and torch.cuda.is_available()

    if cuda_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        if cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def onpolicy_main():
    print("onpolicy main")

    set_seed(args.seed, cuda_deterministic=args.cuda_deterministic)
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    writer = SummaryWriter(os.path.join(args.log_dir, args.algo, f'{args.env_name}_{args.save_name}'))
    writer.add_text('hyperparameters',
                    pd.DataFrame([[key, str(value)] for key, value in args.__dict__.items()], columns=['param', 'value'])
                    .to_markdown())
    writer.flush()

    # Make vector env
    envs = make_vec_envs(args.env_name,
                         args.seed,
                         args.num_processes,
                         args.gamma, 
                         args.log_dir, 
                         device, 
                         False, 
                         env_kwargs=env_kwargs,)

    # agly ways to access to the environment attirubutes
    if args.env_name.find('doorenv')>-1:
        if args.num_processes>1:
            visionnet_input = envs.venv.venv.visionnet_input
            nn = envs.venv.venv.nn
            env_name = envs.venv.venv.xml_path
        else:
            visionnet_input = envs.venv.venv.envs[0].env.env.env.visionnet_input
            nn = envs.venv.venv.envs[0].env.env.env.nn
            env_name = envs.venv.venv.envs[0].env.env.env.xml_path
        dummy_obs = np.zeros(nn*2+3)
    else:
        dummy_obs = envs.observation_space
        visionnet_input = None
        nn = None

    assert not (args.pretrained_policy_load and args.resume), "cannot finetune and resume training at the same time"
    if args.pretrained_policy_load:
        j_start = 0
        print("loading pretrained model from", args.pretrained_policy_load)
        actor_critic, ob_rms = torch.load(args.pretrained_policy_load)
    elif args.resume:
        # resume from last checkpoint
        savename_regex = f".*?{args.save_name}\\.([0-9]+)\\.pt"
        j_start = int(re.match(savename_regex, args.resume).group(1)) + 1
        print("loading checkpoint from", args.resume)
        actor_critic, ob_rms = torch.load(args.resume)
    else:
        # start from episode 0 with new policy
        j_start = 0
        actor_critic = Policy(
            dummy_obs.shape,
            envs.action_space,
            base=(HNBase if args.algo.find("hnppo") > -1 else None),
            base_kwargs={'recurrent': args.recurrent_policy,
                         'action_space': envs.action_space,
                         'hnet': HyperNetwork if args.algo == "hnppo" else ChunkedHyperNetwork if args.algo == "chnppo" else None,
                         'hidden_size': args.network_size,
                         'te_dim': args.te_dim})
    
    if visionnet_input: 
            visionmodel = load_visionmodel(args.save_name, args.visionmodel_path, VisionModelXYZ())  
            actor_critic.visionmodel = visionmodel.eval()
    actor_critic.nn = nn
    actor_critic.to(device)

    #only hypernetwork algos needs task_id
    if args.task_id is not None:
        actor_critic.base.set_active_task(args.task_id)

    #disable normalizer
    vec_norm = get_vec_normalize(envs)
    vec_norm.eval()
    
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo.find('hnppo') > -1:
        agent = algo.ppo.HNPPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            task_id=args.task_id,
            beta=args.beta)
    else:
        raise ValueError(f"Unknown algo {args.algo}")

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              dummy_obs.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    full_obs = envs.reset()
    initial_state = full_obs[:,:envs.action_space.shape[0]]

    if args.env_name.find('doorenv')>-1 and visionnet_input:
        obs = actor_critic.obs2inputs(full_obs, 0)
    else:
        if knob_noisy:
            obs = add_vision_noise(full_obs, 0)
        elif obs_noisy:
            obs = add_joint_noise(full_obs)
        else:
            obs = full_obs

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(j_start, num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates, args.lr)

        # total_switches = 0
        # prev_selection = ""
        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
                next_action = action 

            if args.pos_control:
                # print("main step_skip",args.step_skip)
                if step%(512/args.step_skip-1)==0: current_state = initial_state
                next_action = current_state + next_action
                for kk in range(args.step_skip):
                    full_obs, reward, done, infos = envs.step(next_action)
                    
                current_state = full_obs[:,:envs.action_space.shape[0]]
            else:
                for kk in range(args.step_skip):
                    full_obs, reward, done, infos = envs.step(next_action)

            # convert img to obs if door_env and using visionnet 
            if args.env_name.find('doorenv')>-1 and visionnet_input:
                obs = actor_critic.obs2inputs(full_obs, j)
            else:
                if knob_noisy:
                    obs = add_vision_noise(full_obs, j)
                elif obs_noisy:
                    obs = add_joint_noise(full_obs)
                else:
                    obs = full_obs

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
            
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        # Get total number of timesteps
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        # number of timesteps in this training run (since resumption)
        current_num_steps = (j - j_start + 1) * args.num_processes * args.num_steps

        writer.add_scalar("Value loss", value_loss, j)
        writer.add_scalar("action loss", action_loss, j)
        writer.add_scalar("dist entropy loss", dist_entropy, j)
        writer.add_scalar("Episode rewards", np.mean(episode_rewards), j)
        if isinstance(agent.actor_critic.base, HNBase):
            # log target networks' weights
            for name, param in enumerate(agent.actor_critic.base.hnet.task_embs):
                writer.add_histogram(f'emb.{name}', param.clone().detach().cpu().numpy(), j)
            for name, param in enumerate(agent.actor_critic.base.actor.weights):
                writer.add_histogram(f'actor.{name}', param.clone().detach().cpu().numpy(), j)
            for name, param in agent.actor_critic.base.critic.named_parameters():
                writer.add_histogram(f'critic.{name}', param.clone().detach().cpu().numpy(), j)
            for name, param in agent.actor_critic.base.dist.weights.items():
                writer.add_histogram(f'dist.{name}', param.clone().detach().cpu().numpy(), j)

        writer.add_scalar("rollout_value/max", rollouts.value_preds.flatten().max(), j)
        writer.add_scalar("rollout_value/min", rollouts.value_preds.flatten().min(), j)
        writer.add_scalar("rollout_value/mean", rollouts.value_preds.flatten().mean(), j)
        writer.add_scalar("rollout_value/variance", rollouts.value_preds.flatten().var(), j)
        #writer.add_custom_scalars_multilinechart(['max_val', 'min_val', 'mean_val'])

        writer.add_scalar("rollout_reward/max", rollouts.rewards.flatten().max(), j)
        writer.add_scalar("rollout_reward/min", rollouts.rewards.flatten().min(), j)
        writer.add_scalar("rollout_reward/mean", rollouts.rewards.flatten().mean(), j)
        writer.add_scalar("rollout_reward/variance", rollouts.rewards.flatten().var(), j)
        #writer.add_custom_scalars_multilinechart(['max_reward', 'min_reward', 'mean_reward'])

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo, f"{args.env_name}_{args.save_name}")
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, f"{args.save_name}.{j}.pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(current_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):

            opening_rate, opening_timeavg = onpolicy_inference(
                                                seed=args.seed, 
                                                env_name=args.env_name, 
                                                det=True, 
                                                load_name=args.save_name, 
                                                evaluation=True, 
                                                render=False, 
                                                knob_noisy=args.knob_noisy, 
                                                visionnet_input=args.visionnet_input, 
                                                env_kwargs=env_kwargs_val,
                                                actor_critic=actor_critic,
                                                verbose=False,
                                                pos_control=args.pos_control,
                                                step_skip=args.step_skip,
                                                task_id=args.task_id)

            print("{}th update. {}th timestep. opening rate {}%. Average time to open is {}.".format(j, total_num_steps, opening_rate, opening_timeavg))
            writer.add_scalar("Opening rate per envstep", opening_rate, total_num_steps)
            writer.add_scalar("Opening rate per update", opening_rate, j)

        DR=True #Domain Randomization
        ################## for multiprocess world change ######################
        if DR:
            print("changing world")

            envs.close_extras()
            envs.close()
            del envs

            envs = make_vec_envs(args.env_name,
                        args.seed,
                        args.num_processes,
                        args.gamma, 
                        args.log_dir, 
                        device, 
                        False, 
                        env_kwargs=env_kwargs,)

            full_obs = envs.reset()
            if args.env_name.find('doorenv')>-1 and visionnet_input:
                obs = actor_critic.obs2inputs(full_obs, j)
            else:
                obs = full_obs
        #######################################################################


def offpolicy_main(variant):
    print("offpolicy main")  

    setup_logger('{0}_{1}'.format(args.env_name, args.save_name), variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=True)

    expl_env, eval_env, env_obj = prepare_env(args.env_name, args.visionmodel_path, **env_kwargs)
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size    
    expl_policy, eval_policy, trainer = prepare_trainer(args.algo, expl_env, obs_dim, action_dim, args.pretrained_policy_load, variant)

    if args.env_name.find('doorenv')>-1:
        expl_policy.knob_noisy = eval_policy.knob_noisy = args.knob_noisy
        expl_policy.nn = eval_policy.nn = env_obj.nn
        expl_policy.visionnet_input = eval_policy.visionnet_input = env_obj.visionnet_input

    if args.visionnet_input:
        visionmodel = load_visionmodel(expl_env._wrapped_env.xml_path, args.visionmodel_path, VisionModelXYZ())
        visionmodel.to(ptu.device)
        expl_policy.visionmodel = visionmodel.eval()
    else:
        expl_policy.visionmodel = None

    # print("intput stepskip:", args.step_skip)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
        doorenv=args.env_name.find('doorenv')>-1,
        pos_control=args.pos_control,
        step_skip=args.step_skip,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
        doorenv=args.env_name.find('doorenv')>-1,
        pos_control=args.pos_control,
        step_skip=args.step_skip,
    )

    if not args.replaybuffer_load:
        replay_buffer = EnvReplayBuffer(
            variant['replay_buffer_size'],
            expl_env,
        )
    else:
        replay_buffer = pickle.load(open(args.replaybuffer_load,"rb"))
        replay_buffer._env_info_keys = replay_buffer.env_info_sizes.keys()
        print("Loaded the replay buffer that has length of {}".format(replay_buffer.get_diagnostics()))

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )

    algorithm.save_interval = args.save_interval
    algorithm.save_dir = args.save_dir
    algorithm.algo = args.algo
    algorithm.env_name = args.env_name
    algorithm.save_name = args.save_name
    algorithm.env_kwargs = env_kwargs
    algorithm.env_kwargs_val = env_kwargs_val
    algorithm.eval_function = offpolicy_inference
    algorithm.eval_interval = args.eval_interval
    algorithm.knob_noisy = knob_noisy
    algorithm.visionnet_input = args.visionnet_input
    algorithm.pos_control = args.pos_control
    algorithm.step_skip = args.step_skip
    algorithm.max_path_length = variant['algorithm_kwargs']['max_path_length']

    writer = SummaryWriter(os.path.join(args.log_dir, args.algo, f'{args.env_name}_{args.save_name}'))
    algorithm.writer = writer

    algorithm.to(ptu.device)
    algorithm.train()


def parse(args):
    import datetime
    opt = args
    args = vars(opt)
    verbose = True
    if verbose:
        print('------------ Options -------------')
        print("start time:", datetime.datetime.now())
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
    # save to the disk
    expr_dir = os.path.join(opt.params_log_dir)
    file_name = os.path.join(expr_dir, '{}.txt'.format(opt.env_name))
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        opt_file.write('start time:' + str(datetime.datetime.now()))
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

if __name__ == "__main__":
    args = get_args()

    knob_noisy = args.knob_noisy
    obs_noisy = args.obs_noisy
    env_kwargs = dict(port = args.port,
                    visionnet_input = args.visionnet_input,
                    unity = args.unity,
                    world_path = args.world_path,
                    pos_control = args.pos_control)

    env_kwargs_val = env_kwargs.copy()
    if args.val_path: env_kwargs_val['world_path'] = args.val_path

    if args.algo == 'sac':
        variant = dict(
            algorithm=args.algo,
            version="normal",
            layer_size=100,
            algorithm_kwargs=dict(
                num_epochs=6000,
                num_eval_steps_per_epoch=512, #512
                num_trains_per_train_loop=1000, #1000
                num_expl_steps_per_train_loop=512, #512
                min_num_steps_before_training=512, #1000
                max_path_length=512, #512
                batch_size=128,
                ),
            trainer_kwargs=dict(
                discount=0.99,
                soft_target_tau=5e-3,
                target_update_period=1,
                policy_lr=1E-3,
                qf_lr=1E-3,
                reward_scale=0.1,
                use_automatic_entropy_tuning=True,
                ),
            replay_buffer_size=int(1E6),
        )
        # args_variant = {**vars(args), **variant}
        # parse(args_variant)
        offpolicy_main(variant)

    elif args.algo == 'hnsac':
        variant = dict(
            algorithm=args.algo,
            version="normal",
            layer_size=100,
            algorithm_kwargs=dict(
                num_epochs=6000,
                num_eval_steps_per_epoch=512,  # 512
                num_trains_per_train_loop=1000,  # 1000
                num_expl_steps_per_train_loop=512,  # 512
                min_num_steps_before_training=512,  # 1000
                max_path_length=512,  # 512
                batch_size=128,
            ),
            trainer_kwargs=dict(
                discount=0.99,
                soft_target_tau=5e-3,
                target_update_period=1,
                policy_lr=1E-3,
                qf_lr=1E-3,
                reward_scale=0.1,
                use_automatic_entropy_tuning=False,
                beta=args.beta,
                task_id=args.task_id,
            ),
            replay_buffer_size=int(1E6),
        )
        offpolicy_main(variant)

    elif args.algo == 'td3':
        variant = dict(
            algorithm=args.algo,
            algorithm_kwargs=dict(
                num_epochs=3000,
                num_eval_steps_per_epoch=512,
                num_trains_per_train_loop=1000,
                num_expl_steps_per_train_loop=512,
                min_num_steps_before_training=512,
                max_path_length=512,
                batch_size=128,
            ),
            trainer_kwargs=dict(
                discount=0.99,
                policy_learning_rate=1e-3,
                qf_learning_rate=1e-3,
                policy_and_target_update_period=2,
                tau=0.005,
            ),
            qf_kwargs=dict(
                hidden_sizes=[100, 100],
            ),
            policy_kwargs=dict(
                hidden_sizes=[100, 100],
            ),
            replay_buffer_size=int(1E6),
        )
        # args_variant = {**args, **variant}
        # parse(args_variant)
        offpolicy_main(variant)
    elif args.algo in ['a2c', 'ppo', 'hnppo', 'chnppo']:
        # parse(args_variant)
        onpolicy_main()
    else:
        raise Exception("unknown algorithm")