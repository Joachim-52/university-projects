"""DQN training driver for the report.

Execs the (graded) code cells of dqn_atari.ipynb verbatim so the experiment is
faithful to the notebook, then runs a wall-clock-bounded training loop on Atari
Breakout, periodically saving metrics, a checkpoint and a learning-curve figure.

Usage: python dqn_train.py [max_seconds] [device]
"""
import json, sys, time, os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, 'dqn_atari.ipynb')
MAX_SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 2100.0
FORCE_DEVICE = sys.argv[2] if len(sys.argv) > 2 else None

nb = json.load(open(NB))


def strip_magics(src):
    return '\n'.join(l for l in src.split('\n') if not l.lstrip().startswith(('%', '!')))


g = {'__name__': '__main__'}
for idx in [1, 2, 3, 4, 5, 6]:
    src = strip_magics(''.join(nb['cells'][idx]['source']))
    if FORCE_DEVICE and idx == 3:
        src += f"\ndevice = torch.device('{FORCE_DEVICE}')\nprint('FORCED device:', device)\n"
    exec(compile(src, f'<cell{idx}>', 'exec'), g)

# ---- Hyperparameters (mirror notebook cell 7) ----
batch_size = 32
alpha = 1e-4
gamma = 0.99
eps, eps_decay, min_eps = 1.0, 0.99995, 0.1
burn_in_phase = 5_000
sync_target = 1_000
max_train_frames = 5_000
run_as_ddqn = False

DeepQNet, AtariAgent = g['DeepQNet'], g['AtariAgent']
ExperienceReplayMemory = g['ExperienceReplayMemory']
online_dqn, target_dqn = g['online_dqn'], g['target_dqn']
device = g['device']

buffer = ExperienceReplayMemory(10_000)
optimizer = torch.optim.Adam(online_dqn.parameters(), lr=alpha)
criterion = torch.nn.MSELoss()

agent = AtariAgent(buffer=buffer, eps=eps, eps_decay=eps_decay, min_eps=min_eps, gamma=gamma,
                   batch_size=batch_size, online_dqn=online_dqn, target_dqn=target_dqn,
                   run_as_ddqn=run_as_ddqn, optimizer=optimizer, criterion=criterion,
                   device=device, max_train_frames=max_train_frames,
                   burn_in_phase=burn_in_phase, sync_target=sync_target)
g['agent'] = agent
print(f'device={device} | params={sum(p.numel() for p in online_dqn.parameters()):,} '
      f'| budget={MAX_SECONDS:.0f}s', flush=True)


def save(metrics, tag='dqn'):
    np.savez(os.path.join(HERE, f'{tag}_metrics.npz'),
             reward=np.array(metrics['reward']), loss=np.array(metrics['loss']),
             ep_len=np.array(metrics['ep_len']), current_step=agent.current_step,
             eps=agent.eps)
    torch.save({'online_dqn': online_dqn.state_dict(),
                'target_dqn': target_dqn.state_dict(),
                'eps': agent.eps, 'curr_step': agent.current_step,
                'train_metrics': {k: metrics[k] for k in ('reward', 'loss')}},
               os.path.join(HERE, 'saved_model.pt'))


train_metrics = dict(reward=[], loss=[], ep_len=[])
t0 = time.time()
it = 0
while True:
    ep_start_step = agent.current_step
    m = g['agent'].run_episode(is_training=True)
    train_metrics['reward'].append(m['reward'])
    train_metrics['loss'].append(m['loss'])
    train_metrics['ep_len'].append(agent.current_step - ep_start_step)
    it += 1
    elapsed = time.time() - t0
    if it % 10 == 0 or it <= 3:
        w = train_metrics['reward'][-50:]
        print(f"ep {it:4d} | step {agent.current_step:7d} | eps {agent.eps:5.3f} | "
              f"R(50) {np.mean(w):5.2f} | R {m['reward']:4.0f} | len {train_metrics['ep_len'][-1]:4d} | "
              f"loss {m['loss']:.4f} | {elapsed/60:4.1f}min | {agent.current_step/max(elapsed,1):4.0f} fps",
              flush=True)
    if it % 25 == 0:
        save(train_metrics)
    if elapsed > MAX_SECONDS:
        print(f'Wall-clock budget reached after {it} episodes / {agent.current_step} frames.', flush=True)
        break

save(train_metrics)
print(f'DONE: {it} episodes, {agent.current_step} frames, final eps={agent.eps:.3f}', flush=True)

# learning-curve figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def running_mean(x, w):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    w = max(1, min(w, len(x)))
    return np.convolve(x, np.ones(w) / w, mode='valid')


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
r = train_metrics['reward']; l = train_metrics['loss']
axes[0].plot(r, color='tab:blue', alpha=0.3, label='per episode')
sm = running_mean(r, 20); axes[0].plot(np.arange(len(sm)) + (len(r) - len(sm)), sm, 'tab:blue', lw=2, label='running mean (20)')
axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Accumulated reward'); axes[0].set_title('DQN training: reward per episode'); axes[0].legend()
axes[1].plot(l, color='tab:red', alpha=0.3, label='per episode')
sm = running_mean(l, 20); axes[1].plot(np.arange(len(sm)) + (len(l) - len(sm)), sm, 'tab:red', lw=2, label='running mean (20)')
axes[1].set_xlabel('Episode'); axes[1].set_ylabel('Mean TD loss'); axes[1].set_title('DQN training: loss per episode'); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'dqn_train_curves.png'), dpi=130, bbox_inches='tight')
print('Saved dqn_train_curves.png', flush=True)
