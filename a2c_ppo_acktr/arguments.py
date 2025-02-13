import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    add_common_args(parser)
    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: a2c | ppo | sac | td3')
    parser.add_argument(
        '--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--beta',
        type=float,
        default=5e-3,
        help='hypernetwork regularization coefficient (default: 5e-3')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_false',
        default=True,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=8,
        help='how many training CPU processes to use (default: 8)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=4096,
        help='number of forward steps (default: 4096)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=10,
        help='number of ppo epochs (default: 10)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=256,
        help='number of batches for ppo (default: 256)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1,
        help='log interval, one log per n updates (default: 1)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=5,
        help='save interval, one save per n updates (default: 5)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=20,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e8,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--log-dir',
        default='./logs/',
        help='directory to save agent logs (default: ./logs/)')
    parser.add_argument(
        '--params-log-dir',
        default='./params_logs/',
        help='directory to save params logs (default: ./params_logs/)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_false',
        default=True,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--save-name',
        type=str,
        required=True,
        help='name for changing the log and model name')
    parser.add_argument(
        '--obs-noisy',
        action='store_true',
        default=False,
        help='add noise to entire observation signal')
    parser.add_argument(
        '--pretrained-policy-load',
        type=str,
        default=None,
        help='which pretrained model to load')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='resume training from a checkpointed model, preserves the epoch number')
    parser.add_argument(
        '--replaybuffer-load',
        type=str,
        default=None,
        help='load the replay buffer')
    parser.add_argument(
        '--val-path',
        type=str,
        default=None,
        help='load the vision network model')
    parser.add_argument(
        '--network-size',
        nargs='+',
        type=int,
        default=None,
        help='List of widths of each hidden layer in the policy network. If given, overrides --network-width and '
             '--network-depth. Currently customizable network size is only implemented for PPO and HNPPO. '
        # TODO make available to other algos
    )

    parser.add_argument(
        '--network-width',
        type=int,
        default=64,
        help='Width of hidden layers. Currently customizable network size is only implemented for PPO and HNPPO. (default: 64)'
    )
    parser.add_argument(
        '--network-depth',
        type=int,
        default=2,
        help='Number of hidden layers. Currently customizable network size is only implemented for PPO and HNPPO. (default: 2)'
    )
    parser.add_argument(
        '--te-dim',
        type=int,
        default=8,
        help='Hypernetwork task embedding dimension'
    )
    parser.add_argument(
        '--freshcritic',
        type=bool,
        default=True,
        help='Whether to use fresh critic (non-HN) or HN-based critic network. Default is freshcritic, i.e. True'
    )

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'sac', 'td3', 'hnppo', 'chnppo', 'hnsac']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for sac'

    if args.unity:
        assert args.visionnet_input, \
            'Visionnet_input should be True when Unity is True'

    if args.algo.find('hn') > -1 and args.task_id is None:
        parser.error('task_id is required for hypernetwork algorithms')
    if args.algo.find('hn') <= -1 and args.task_id is not None:
        parser.error('Unexpected argument "task_id" for non-hypernetwork algorithm')

    # if network_size is not given, determine it from width and depth
    if not args.network_size:
        args.network_size = [args.network_width] * args.network_depth

    return args


def get_args_enjoy():
    """
    Argparser for evaluation/inference script enjoy.py
    """
    parser = argparse.ArgumentParser(description='RL')
    add_common_args(parser)
    parser.add_argument(
        '--non-det',
        action='store_true',
        default=False,
        help='whether to use a non-deterministic policy')
    parser.add_argument(
        '--load-name',
        type=str,
        required=True,
        help='which model to load')
    parser.add_argument(
        '--eval',
        action='store_true',
        default=False,
        help="Measure the opening ratio among 100 trials")
    parser.add_argument(
        '--render',
        action='store_true',
        default=False,
        help="force rendering")
    args = parser.parse_args()

    return args


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Arguments used for both training and eval go here. """
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--env-name',
        default='doorenv-v0',
        help='environment to train on (default: doorenv-v0)')
    parser.add_argument(
        '--knob-noisy',
        action='store_true',
        default=False,
        help='add noise to knob position to resemble the noise from the visionnet')
    parser.add_argument(
        '--visionnet-input',
        action="store_true",
        default=False,
        help='Use vision net for knob position estimation')
    parser.add_argument(
        '--unity',
        action="store_true",
        default=False,
        help='Use unity for an input of a vision net')
    parser.add_argument(
        '--port',
        type=int,
        default=1050,
        help='Unity connection port (Only for off-policy)')
    parser.add_argument(
        '--visionmodel-path',
        type=str,
        default="./trained_visionmodel/",
        help='load the vision network model')
    parser.add_argument(
        '--world-path',
        type=str,
        help='Folder containing the world files')
    parser.add_argument(
        '--pos-control',
        action="store_true",
        default=False,
        help='Turn on pos control')
    parser.add_argument(
        '--step-skip',
        type=int,
        default=4,
        help='number of step skip in pos control')
    parser.add_argument(
        '--task-id',
        type=int,
        default=None,
        help='task id for continual learning')
