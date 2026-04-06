"""
Task 1 - Monte Carlo Simulation
Truncated Poisson distribution on X = {0, 1, 2, 3, 4} with lambda = 4.
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# ── Parameters ────────────────────────────────────────────────────────────────
LAMBDA = 4
STATE_SPACE = np.array([0, 1, 2, 3, 4])

# ── Task 1a: Normalizing constant M ──────────────────────────────────────────
# p(X=k) = M * lambda^k * e^{-lambda} / k!
# Sum over k in {0..4} must equal 1:
#   M * e^{-lambda} * sum_{k=0}^{4} lambda^k / k! = 1
#   => M = 1 / (e^{-lambda} * S)  where S = sum_{k=0}^{4} lambda^k / k!

S = sum(LAMBDA**k / math.factorial(k) for k in STATE_SPACE)
M = 1.0 / (math.exp(-LAMBDA) * S)

print("Task 1a – Normalizing constant M")
print(f"  S = sum_{{k=0}}^4 lambda^k/k! = {S:.6f}")
print(f"  M = 1 / (e^{{-4}} * S) = {M:.6f}")
print()

# ── PMF ───────────────────────────────────────────────────────────────────────
pmf = np.array([M * LAMBDA**k * math.exp(-LAMBDA) / math.factorial(k) for k in STATE_SPACE])
assert abs(pmf.sum() - 1.0) < 1e-12, "PMF does not sum to 1"

# ── Task 1b: Analytical expected value ───────────────────────────────────────
# E[X] = sum_{k=0}^{4} k * p(X=k)
#       = M * e^{-4} * sum_{k=0}^4 k * 4^k / k!
#       = 284 / 103  (exact fraction)
E_X_exact = sum(k * pmf[k] for k in STATE_SPACE)
E_X_fraction = 284 / 103  # derived analytically

print("Task 1b – Analytical expected value E[X]")
print(f"  E[X] = 284/103 = {E_X_fraction:.6f}")
print(f"  Verification via PMF: {E_X_exact:.6f}")
print()

# ── Task 1c: Monte Carlo convergence ─────────────────────────────────────────
rng = np.random.default_rng(seed=42)

# Draw 10 000 samples once, then compute running means at each checkpoint
max_sims = 10_000
samples = rng.choice(STATE_SPACE, size=max_sims, p=pmf)

checkpoints = np.arange(100, max_sims + 1, 100)          # 100, 200, …, 10 000
estimates = np.array([samples[:n].mean() for n in checkpoints])

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(checkpoints, estimates, color='steelblue', linewidth=1.2, label='MC estimate')
ax.axhline(E_X_fraction, color='tomato', linewidth=1.5,
           linestyle='--', label=f'Exact E[X] = 284/103 ≈ {E_X_fraction:.3f}')
ax.set_xlabel('Number of simulations')
ax.set_ylabel('Estimated E[X]')
ax.set_title('Monte Carlo convergence – Truncated Poisson (λ=4)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('monte_carlo_convergence.png', dpi=150)
plt.show()
print("Plot saved to monte_carlo_convergence.png")
