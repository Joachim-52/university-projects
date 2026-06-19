"""Regenerate dqn_train_curves.png from the saved real-training metrics
(dqn_metrics.npz), independent of the training process. Same style as the
notebook's plotting cell."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, 'dqn_metrics.npz'))
r, l = d['reward'].astype(float), d['loss'].astype(float)
print(f'episodes={len(r)} frames={int(d["current_step"])} eps={float(d["eps"]):.3f} '
      f'first50={np.mean(r[:50]):.2f} last50={np.mean(r[-50:]):.2f} max={r.max():.0f}')


def rm(x, w):
    w = max(1, min(w, len(x)))
    return np.convolve(x, np.ones(w) / w, mode='valid')


fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(r, color='tab:blue', alpha=0.3, label='per episode')
s = rm(r, 20); ax[0].plot(np.arange(len(s)) + (len(r) - len(s)), s, 'tab:blue', lw=2, label='running mean (20)')
ax[0].set_xlabel('Episode'); ax[0].set_ylabel('Accumulated reward')
ax[0].set_title('DQN training: reward per episode'); ax[0].legend()
ax[1].plot(l, color='tab:red', alpha=0.3, label='per episode')
s = rm(l, 20); ax[1].plot(np.arange(len(s)) + (len(l) - len(s)), s, 'tab:red', lw=2, label='running mean (20)')
ax[1].set_xlabel('Episode'); ax[1].set_ylabel('Mean TD loss')
ax[1].set_title('DQN training: loss per episode'); ax[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'dqn_train_curves.png'), dpi=130, bbox_inches='tight')
print('Saved dqn_train_curves.png')
