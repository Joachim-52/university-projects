"""Hyperparameter grid search for the FrozenLake model-free agents.

Runs a grid over (alpha, eps_decay) for every (algorithm, gamma) combination,
averaging the mean test reward over a few training seeds, and prints the best
configuration plus a ranked table per case. Pure search utility -- not graded.
"""
import itertools
import time
import numpy as np

from rl_frozen_lake import ModelFreeAgent, RLAlgorithm

ALPHAS = [0.01, 0.05, 0.1, 0.2, 0.5]
EPS_DECAYS = [0.998, 0.999, 0.9993, 0.9995, 0.9997]
GAMMAS = [0.95, 1.0]
ALGOS = [RLAlgorithm.SARSA, RLAlgorithm.Q_LEARNING, RLAlgorithm.EXPECTED_SARSA]
SEEDS = [0, 1, 2]
NUM_TRAIN = 10_000
NUM_TEST = 2_000


def evaluate(algo, gamma, alpha, eps_decay, seed):
    np.random.seed(seed)
    agent = ModelFreeAgent(algorithm=algo, alpha=alpha, eps=1.0, gamma=gamma,
                           eps_decay=eps_decay, num_train_episodes=NUM_TRAIN,
                           num_test_episodes=NUM_TEST, max_episode_length=200)
    agent.train()
    agent.test()
    return float(np.mean(agent.test_reward))


def main():
    t0 = time.time()
    results = {}  # (gamma, algo) -> list of (mean, std, alpha, eps_decay)
    for gamma in GAMMAS:
        for algo in ALGOS:
            rows = []
            for alpha, eps_decay in itertools.product(ALPHAS, EPS_DECAYS):
                scores = [evaluate(algo, gamma, alpha, eps_decay, s) for s in SEEDS]
                rows.append((np.mean(scores), np.std(scores), alpha, eps_decay))
            rows.sort(key=lambda r: r[0], reverse=True)
            results[(gamma, algo.value)] = rows
            print(f"\n===== {algo.value} | gamma={gamma} "
                  f"(elapsed {time.time()-t0:.0f}s) =====")
            print("  rank  mean_test   std    alpha   eps_decay")
            for i, (m, s, a, d) in enumerate(rows):
                marker = " <-- best" if i == 0 else ""
                print(f"  {i+1:>4d}  {m:8.4f}  {s:6.4f}  {a:6.3f}  {d:8.4f}{marker}")

    print("\n\n########## SUMMARY (best per case) ##########")
    for (gamma, algo), rows in results.items():
        m, s, a, d = rows[0]
        print(f"{algo:15s} | gamma={gamma:4} | alpha={a:5.3f} | "
              f"eps_decay={d:.4f} | mean_test={m:.4f} +/- {s:.4f}")
    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
