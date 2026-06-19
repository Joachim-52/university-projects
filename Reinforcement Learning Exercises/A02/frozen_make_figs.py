"""Generate the six FrozenLake result figures (one per algorithm x gamma) using
the selected hyperparameters. Saves the official PDF (as produced by
plot_frozenlake_model_free_results) and a PNG copy for the report, and prints a
metrics table + value-function grids for the write-up."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from rl_frozen_lake import ModelFreeAgent, RLAlgorithm, BEST_HP
from frozen_lake_utils import plot_frozenlake_model_free_results

SEED = 0
GAMMAS = [0.95, 1.0]
ALGOS = [RLAlgorithm.SARSA, RLAlgorithm.Q_LEARNING, RLAlgorithm.EXPECTED_SARSA]

print(f"{'algorithm':16s} {'gamma':6s} {'alpha':6s} {'eps_decay':9s} "
      f"{'train_R':8s} {'test_R':7s}")
rows = []
for gamma in GAMMAS:
    for algo in ALGOS:
        alpha, eps_decay = BEST_HP[(algo, gamma)]
        agent = ModelFreeAgent(algorithm=algo, alpha=alpha, eps=1.0, gamma=gamma,
                               eps_decay=eps_decay, num_train_episodes=10_000,
                               num_test_episodes=5_000, max_episode_length=200)
        # match train_test_agent(seed=...) so report PNGs == submission PDFs
        np.random.seed(SEED)
        agent.env.reset(seed=SEED)
        agent.env.action_space.seed(SEED)
        agent.train()
        agent.test()
        train_r, test_r = float(np.mean(agent.train_reward)), float(np.mean(agent.test_reward))
        rows.append((algo, gamma, alpha, eps_decay, train_r, test_r))
        print(f"{algo.value:16s} {gamma:<6} {alpha:<6} {eps_decay:<9} "
              f"{train_r:<8.4f} {test_r:<7.4f}")

        # figure (PNG for report + PDF for submission)
        plot_frozenlake_model_free_results(agent, gamma, savefig=False)
        base = f"frozenlake_{algo.value.replace(' ', '_')}_gamma_{gamma}"
        plt.savefig(base + '.png', dpi=140, bbox_inches='tight')
        plt.savefig(f"FrozenLake_{algo.value.replace(' ', '_')}_gamma_{gamma}.pdf",
                    bbox_inches='tight')
        plt.close('all')

        v = np.max(agent.Q, axis=1).reshape(4, 4)
        print('   v(s) =')
        for r in v:
            print('     ', '  '.join(f'{x:5.3f}' for x in r))

print('\nDone. Figures saved (PNG + PDF).')
