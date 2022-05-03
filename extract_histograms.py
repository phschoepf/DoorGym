import re
import sys
import os
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter

basedir = sys.argv[1]
writer = SummaryWriter(os.path.join('logs', 'extracted', os.path.basename(basedir.rstrip('/'))))
filepat = re.compile(r'.*\.([0-9]+)\.pt')
folderpat = re.compile(r'.*?-task([0-9]).*?')
task_id = int(re.match(folderpat, basedir).group(1))

for checkpoint in sorted(os.listdir(basedir), key=lambda s: int(re.match(filepat, s).group(1))):
    actor_critic, ob_rms = torch.load(os.path.join(basedir, checkpoint))
    j = int(re.match(filepat, checkpoint).group(1))
    print(f'processing iteration {j}')

    actor_critic.base.set_active_task(task_id)
    try:
        # we're not interested in actually forwarding sth, only in setting the weights of the target network
        actor_critic.base(None, None, None)
    except AttributeError:
        # thus we ignore the attribute error thrown by forwarding a NoneType
        pass

    actor_critic.base.actor.set_weights(nn.ParameterList(nn.Parameter(t) for t in actor_critic.base.actor.weights))
    actor_critic.base.critic.set_weights(nn.ParameterList(nn.Parameter(t) for t in actor_critic.base.critic.weights))
    actor_critic.base.dist.weights = nn.ParameterDict(dict([(name, nn.Parameter(t)) for name, t in actor_critic.base.dist.weights.items()]))

    # log embeddings to see how they change with the task
    all_embs = torch.cat([emb.clone().cpu().data.expand(1, -1) for emb in actor_critic.base.hnet.task_embs])
    # writer.add_embedding(mat=all_embs, tag=f'hnet.embeddings', global_step=j)

    for name, param in enumerate(actor_critic.base.hnet.task_embs):
        writer.add_histogram(f'emb.{name}', param.clone().cpu().data.numpy(), j)
    # log histograms of target network weights
    for name, param in actor_critic.base.actor.named_parameters(prefix='actor'):
        writer.add_histogram(name, param.clone().cpu().data.numpy(), j)
    for name, param in actor_critic.base.critic.named_parameters(prefix='critic'):
        writer.add_histogram(name, param.clone().cpu().data.numpy(), j)
    for name, param in actor_critic.base.dist.named_parameters(prefix='dist'):
        writer.add_histogram(name, param.clone().cpu().data.numpy(), j)
