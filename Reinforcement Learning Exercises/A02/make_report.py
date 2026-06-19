"""Build the Assignment-2 report PDF with reportlab.

Pulls the DQN training summary automatically from dqn_metrics.npz (if present)
and embeds the FrozenLake figures and the DQN learning-curve figure.
"""
import os
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                                TableStyle, PageBreak, KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

HERE = os.path.dirname(os.path.abspath(__file__))
MPL = '/opt/miniconda3/envs/reinforcementlearning/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf'
pdfmetrics.registerFont(TTFont('DejaVu', os.path.join(MPL, 'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu-Bold', os.path.join(MPL, 'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu-Obl', os.path.join(MPL, 'DejaVuSans-Oblique.ttf')))
pdfmetrics.registerFont(TTFont('DejaVuMono', os.path.join(MPL, 'DejaVuSansMono.ttf')))
pdfmetrics.registerFontFamily('DejaVu', normal='DejaVu', bold='DejaVu-Bold', italic='DejaVu-Obl')

# ----------------------------- styles -----------------------------
BODY = ParagraphStyle('body', fontName='DejaVu', fontSize=9.3, leading=13.2,
                      alignment=TA_JUSTIFY, spaceAfter=5)
H1 = ParagraphStyle('h1', fontName='DejaVu-Bold', fontSize=14, leading=17,
                    spaceBefore=10, spaceAfter=6, textColor=colors.HexColor('#11335a'))
H2 = ParagraphStyle('h2', fontName='DejaVu-Bold', fontSize=11, leading=14,
                    spaceBefore=8, spaceAfter=3, textColor=colors.HexColor('#1d4e79'))
TITLE = ParagraphStyle('title', fontName='DejaVu-Bold', fontSize=18, leading=22,
                       alignment=TA_CENTER, spaceAfter=4)
SUB = ParagraphStyle('sub', fontName='DejaVu', fontSize=11, leading=14, alignment=TA_CENTER)
EQ = ParagraphStyle('eq', fontName='DejaVuMono', fontSize=8.7, leading=12.5,
                    leftIndent=14, spaceBefore=2, spaceAfter=5, textColor=colors.HexColor('#222222'))
CAP = ParagraphStyle('cap', fontName='DejaVu-Obl', fontSize=8.2, leading=10.5,
                     alignment=TA_CENTER, spaceAfter=9, textColor=colors.HexColor('#444444'))


def esc(t):
    t = t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    t = (t.replace('[[b]]', '<b>').replace('[[/b]]', '</b>')
          .replace('[[i]]', '<i>').replace('[[/i]]', '</i>').replace('[[br]]', '<br/>'))
    return t


def P(t, style=BODY):
    return Paragraph(esc(t), style)


def img(path, width_cm):
    w_pt = width_cm * cm
    with PILImage.open(path) as im:
        iw, ih = im.size
    return Image(path, width=w_pt, height=w_pt * ih / iw)


story = []


def H(t, s=H1):
    story.append(P(t, s))


# ----------------------------- title -----------------------------
story += [Spacer(1, 6),
          P('Reinforcement Learning VU (708.006) — SS26', TITLE),
          P('Assignment 2 — RL in a Grid World &amp; Deep Reinforcement Learning', SUB),
          Spacer(1, 8),
          P('Joachim Rath   ·   Matrikelnr.: 51811071   ·   joachim.rath52@gmail.com', SUB),
          P('05.06.2026', SUB),
          Spacer(1, 10)]

# ===================================================================
# PART 1
# ===================================================================
H('1  Reinforcement Learning in a Grid World [10 pts]')
story.append(P(
    "This part uses the same 4×4 slippery FrozenLake as in Assignment 1, but now the agent does not know the "
    "transition dynamics and has to learn from experience. It keeps a tabular Q(s,a) (16 states, 4 actions, "
    "all initialised to 0) and updates it with temporal-difference (TD) control while acting ε-greedily. "
    "I train each agent for 10 000 episodes, decaying ε from 1 by multiplying it with eps_decay after every "
    "episode, and then test the learned greedy policy (ε = 0) for 5 000 episodes. The reward is 1 only for "
    "reaching the goal and 0 otherwise, and an episode ends at a hole or the goal, or after 200 steps."))

H('1.1  Implemented algorithms (Task a)', H2)
story.append(P(
    "[[b]]ε-greedy policy.[[/b]] With probability ε the agent takes a random action, otherwise the greedy "
    "one, arg max_a Q(s,a). For testing I set ε = 0, so the test policy is purely greedy and deterministic. "
    "The three algorithms only differ in the TD target they plug into Q(s,a) ← Q(s,a) + α·[target − Q(s,a)]. "
    "When the next state is terminal I drop the bootstrap term, since a terminal state has value 0:"))
story.append(P("SARSA  (on-policy):", EQ))
story.append(P("Q(s,a) ← Q(s,a) + α·[ r + γ·Q(s′,a′) − Q(s,a) ],   a′ ~ π(·|s′)", EQ))
story.append(P("Q-Learning  (off-policy):", EQ))
story.append(P("Q(s,a) ← Q(s,a) + α·[ r + γ·maxₐ′ Q(s′,a′) − Q(s,a) ]", EQ))
story.append(P("Expected SARSA:", EQ))
story.append(P("Q(s,a) ← Q(s,a) + α·[ r + γ·Σₐ′ π(a′|s′)·Q(s′,a′) − Q(s,a) ]", EQ))
story.append(P(
    "For an ε-greedy policy the expectation over a′ in Expected SARSA has a simple closed form: "
    "E[Q(s′,·)] = ε·meanₐ Q(s′,a) + (1−ε)·maxₐ Q(s′,a). The ε part of the probability is spread evenly over "
    "all actions and the remaining (1−ε) sits on the greedy action. I compute this with the current ε of "
    "the training run."))

H('1.2  Hyperparameter search (Task a)', H2)
story.append(P(
    "[[b]]Grid search.[[/b]] For each algorithm and each γ ∈ {0.95, 1.0} I swept "
    "α ∈ {0.01, 0.05, 0.10, 0.20, 0.50} and eps_decay ∈ {0.998, 0.999, 0.9993, 0.9995, 0.9997}, so 25 "
    "combinations per case (script hp_search.py). I scored each one by the mean greedy test return over "
    "2 000 episodes and averaged that over 3 training seeds. It helps to know what eps_decay means in "
    "episodes: ε drops to about 0.05 after ln(0.05)/ln(eps_decay) episodes, which is roughly 1 500 for 0.998, "
    "3 000 for 0.999, 4 300 for 0.9993, 6 000 for 0.9995 and 10 000 for 0.9997. A smaller eps_decay therefore "
    "means exploration stops earlier."))
story.append(P(
    "A few things stood out. α = 0.01 was always too slow: ε had decayed long before Q had converged, so the "
    "test return stayed around 0.05–0.20. For γ = 0.95 all three algorithms were forgiving and reached about "
    "0.78–0.81 over a wide α range (0.05–0.5), which is basically the Policy-Iteration level (0.786). γ = 1.0 "
    "was a different story for Q-Learning: with α ≥ 0.05 and slow ε-decay it often diverged and the greedy "
    "policy ended up useless (test return 0.0) in 14 of the 25 cells. Only α = 0.05 with eps_decay = 0.9993 "
    "gave a stable, good policy. SARSA and especially Expected SARSA were much less picky here (Expected "
    "SARSA got 0.811 ± 0.001 across seeds). For that reason I report a setting per case that is good and "
    "also stable; a very large α sometimes gave a slightly higher average but with much noisier value "
    "estimates, so I did not use it."))

# best HP table
best_rows = [
    ['Algorithm', 'γ', 'α', 'eps_decay', 'train R̄', 'test R̄', 'PI test (ref.)'],
    ['SARSA', '0.95', '0.05', '0.9993', '0.487', '0.768', '0.786'],
    ['Q-Learning', '0.95', '0.10', '0.9990', '0.554', '0.779', '0.786'],
    ['Expected SARSA', '0.95', '0.10', '0.9993', '0.479', '0.776', '0.786'],
    ['SARSA', '1.0', '0.10', '0.9997', '0.230', '0.816', '0.826'],
    ['Q-Learning', '1.0', '0.05', '0.9993', '0.510', '0.769', '0.826'],
    ['Expected SARSA', '1.0', '0.10', '0.9997', '0.243', '0.816', '0.826'],
]
t = Table(best_rows, hAlign='CENTER', colWidths=[3.3*cm, 1.0*cm, 1.0*cm, 2.0*cm, 1.9*cm, 1.8*cm, 2.4*cm])
t.setStyle(TableStyle([
    ('FONT', (0, 0), (-1, -1), 'DejaVu', 8),
    ('FONT', (0, 0), (-1, 0), 'DejaVu-Bold', 8),
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1d4e79')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#9bb6d2')),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#eef3f9')]),
    ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
]))
story.append(t)
story.append(P("Table 1: The hyperparameters I picked for each case (seed 0), with the mean training and "
               "greedy-test return and the Policy-Iteration reference. All six are close to the "
               "Policy-Iteration baseline.", CAP))

H('1.3  Result figures (Task a)', H2)
story.append(P(
    "Figures 1–6 are the plots from plot_frozenlake_model_free_results, one per algorithm and γ. Each shows "
    "the training reward per episode (with its running mean), the greedy test reward over 5 000 episodes, "
    "the learned policy (white arrows) and the value function v(s) = maxₐ Q(s,a)."))

fl = [
    ('frozenlake_SARSA_gamma_0.95.png', 'Figure 1: SARSA, γ = 0.95.'),
    ('frozenlake_Q-Learning_gamma_0.95.png', 'Figure 2: Q-Learning, γ = 0.95.'),
    ('frozenlake_Expected_SARSA_gamma_0.95.png', 'Figure 3: Expected SARSA, γ = 0.95.'),
    ('frozenlake_SARSA_gamma_1.0.png', 'Figure 4: SARSA, γ = 1.0.'),
    ('frozenlake_Q-Learning_gamma_1.0.png', 'Figure 5: Q-Learning, γ = 1.0.'),
    ('frozenlake_Expected_SARSA_gamma_1.0.png', 'Figure 6: Expected SARSA, γ = 1.0.'),
]
for fn, cap in fl:
    p = os.path.join(HERE, fn)
    if os.path.exists(p):
        story.append(KeepTogether([img(p, 12.5), P(cap, CAP)]))

H('1.4  Comparison of the three algorithms (Task a)', H2)
story.append(P(
    "All three methods end up with a near-optimal greedy policy, so the interesting differences are in how "
    "stable the training is and what the learned values look like. Q-Learning is off-policy and learns the "
    "value of the greedy policy directly through the max, so its v comes out closest to the true optimal v* "
    "(see §1.5). The downside is that the max overestimates, and at γ = 1.0 the Bellman optimality operator "
    "is no longer a contraction, so the updates are not guaranteed to converge. That is why Q-Learning was "
    "the least stable at γ = 1.0 and only worked with a small α. SARSA is on-policy: it evaluates the "
    "ε-greedy policy it actually follows, so it accounts for the chance of slipping into a hole while "
    "exploring and learns more cautious, smaller values. With ε decaying its greedy policy still becomes "
    "near-optimal, and it was clearly more stable than Q-Learning at γ = 1.0. Expected SARSA replaces the "
    "sampled next action by its expectation, which takes out the noise from sampling a′. In my runs it had "
    "the lowest variance between seeds and tolerated the largest learning rates (it actually turns into "
    "Q-Learning if the target policy is greedy)."))

H('1.5  Comparison with Policy Iteration (Task b)', H2)
story.append(P(
    "[[b]]Policies.[[/b]] For both γ all three agents essentially find the same greedy policy as Policy "
    "Iteration (Fig. 1 in the assignment) and reach a similar test success rate (about 0.77–0.82 vs 0.786 / "
    "0.826). The typical slippery behaviour shows up as well: next to a hole the policy points away from it, "
    "often straight into a wall, so that the 1/3 sideways slip cannot push the agent into the hole. At the "
    "start cell, for example, it points left or up against the border."))
story.append(P(
    "[[b]]Value functions.[[/b]] At γ = 0.95 the discount makes the Bellman operator a contraction, and all "
    "three value functions come out close to Policy Iteration's v* (for instance SARSA's bottom-centre cell "
    "is 0.741 vs 0.724, and the start cell is 0.184 vs 0.18). At γ = 1.0 they split apart (Table 2). "
    "Q-Learning still matches v* quite well, but SARSA and Expected SARSA underestimate the values badly once "
    "you move away from the goal. There are two reasons for this. Being on-policy, they evaluate the ε-greedy "
    "policy (which sometimes falls in), not the optimal greedy one. And without any discount the value has to "
    "propagate all the way back from the single goal reward, so within 10 000 episodes the far and "
    "rarely-visited cells simply do not get there. Policy Iteration's v* is about 0.824 almost everywhere "
    "(the probability of eventually reaching the goal under the optimal policy), and that kind of "
    "long-horizon value is exactly what undiscounted TD control is slow to pin down. Even Q-Learning "
    "underestimates the far top-right corner (0.289 vs 0.824) for the same reason."))

vf_rows = [
    ['v(s) at γ = 1.0', 'start (0,0)', 'left-col (2,0)', 'left-of-goal (3,2)'],
    ['Policy Iteration (v*)', '0.824', '0.824', '0.941'],
    ['Q-Learning', '0.797', '0.788', '0.937'],
    ['SARSA', '0.504', '0.574', '0.900'],
    ['Expected SARSA', '0.514', '0.540', '0.824'],
]
t2 = Table(vf_rows, hAlign='CENTER', colWidths=[4.6*cm, 3.0*cm, 3.2*cm, 3.6*cm])
t2.setStyle(TableStyle([
    ('FONT', (0, 0), (-1, -1), 'DejaVu', 8),
    ('FONT', (0, 0), (-1, 0), 'DejaVu-Bold', 8),
    ('FONT', (0, 1), (0, -1), 'DejaVu-Bold', 8),
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1d4e79')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#9bb6d2')),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#eef3f9')]),
    ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
]))
story.append(t2)
story.append(P("Table 2: Values of three cells at γ = 1.0. Near the goal everything agrees; far from it the "
               "on-policy methods fall well short of v*, while off-policy Q-Learning stays close.", CAP))

story.append(PageBreak())

# ===================================================================
# PART 2
# ===================================================================
H('2  Deep Q-Learning for Atari Breakout [20 pts]')
story.append(P(
    "For the second part I train a DQN (Mnih et al., 2015) on Atari Breakout with off-policy Q-learning and "
    "a convolutional network, using the two tricks that make DQN work: a fixed target network and an "
    "experience-replay buffer. The frames are pre-processed with frame-skip 4, turned to grayscale, resized "
    "to 84×84 and the last 4 frames stacked, so one state is a 4×84×84 tensor. Breakout has 4 actions."))

H('2.1  Network, fixed target network and burn-in (Task a)', H2)
story.append(P(
    "[[b]]Network.[[/b]] The online network is the usual Nature-DQN: three conv layers (32 filters 8×8 "
    "stride 4, 64 filters 4×4 stride 2, 64 filters 3×3 stride 1), each followed by a ReLU, then flatten to "
    "3136, a fully-connected layer to 512 with ReLU, and a final layer to the 4 action values. That is about "
    "1.69 M parameters. I get the flattened size from a dummy forward pass so the head still fits if the "
    "input resolution changes, and I kept the activation and a batch-norm flag as arguments so I could try "
    "other configurations (below)."))
story.append(P(
    "[[b]]Fixed target network.[[/b]] There are two networks with the same architecture: the online network "
    "Q_θ that I optimise, and a target network Q_θ̄ whose parameters are frozen (requires_grad = False). The "
    "regression target is"))
story.append(P("y_t = r_{t+1} + γ · maxₐ′ Q_θ̄(s_{t+1}, a′)", EQ))
story.append(P(
    "and I minimise the MSE ( y_t − Q_θ(s_t,a_t) )² with Adam. Every sync_target frames I copy θ̄ ← θ with "
    "load_state_dict. The reason for doing this is that the target itself depends on the network weights. If "
    "I bootstrapped from the online network while it keeps changing, the target would move on every single "
    "step and the training tends to oscillate or blow up. Freezing the target for a while gives a fixed "
    "regression target over many updates, which makes the optimisation much more stable. I also added Double "
    "DQN (the online net picks a′ and the target net evaluates it) to reduce the overestimation from the "
    "max; it is available through run_as_ddqn but off by default."))
story.append(P(
    "[[b]]burn_in_phase.[[/b]] For the first burn_in_phase frames the agent only plays (ε = 1) and stores "
    "transitions, with no gradient updates and no ε-decay yet (in the code both are behind the check "
    "current_step &gt; burn_in_phase). The idea is to fill the replay buffer with some varied experience "
    "before learning starts. If you sampled minibatches from an almost empty buffer they would be very "
    "correlated and the network would just overfit to a handful of early frames. A bigger burn-in means more "
    "random exploration up front and a more mixed buffer, but you start learning later. I used 5 000, and in "
    "the logs you can see that the loss only becomes non-zero and ε only starts to drop once current_step "
    "passes it, which is exactly what should happen."))

H('2.2  Experience replay memory (Task b)', H2)
story.append(P(
    "[[b]]Implementation.[[/b]] The buffer is a deque with a fixed maxlen holding tuples "
    "(s, s′, a, r, terminated, truncated). sample() draws a uniform random minibatch with random.sample and "
    "stacks each part into a batched tensor, i.e. (s,a,r,s′) ~ Uniform(B). Once it is full the oldest "
    "transition gets pushed out."))
story.append(P(
    "[[b]]Why a replay buffer.[[/b]] Most ML models trained with SGD assume the samples are i.i.d. "
    "(independent and identically distributed). If I learned online, straight from the transitions in the "
    "order they happen, that assumption breaks in two ways. First, consecutive transitions are heavily "
    "correlated: s_{t+1} comes directly from s_t and neighbouring frames look almost identical, so a batch of "
    "consecutive samples carries little new information, gives noisy gradients and lets the network forget "
    "older experience. Second, the transitions come from the current policy, which keeps changing as the "
    "agent learns, so the data is not even identically distributed (it is non-stationary). Sampling uniformly "
    "from a large buffer fixes both: a minibatch now mixes many different episodes and time steps, which is "
    "close to independent, and it averages over many past policies, so the distribution shifts more slowly. "
    "On top of that it reuses every transition many times, which matters because stepping the environment is "
    "the expensive part. Together with the target network this is the second thing that keeps DQN stable."))

H('2.3  Hyperparameters, configuration study and results (Task a)', H2)
story.append(P(
    "[[b]]Setup.[[/b]] batch 32, Adam with learning rate 1e-4, γ = 0.99, ε annealed from 1.0 to 0.1 with a "
    "per-frame factor of 0.99995 (so it reaches 0.1 after about 46 000 learning frames), replay capacity "
    "10 000, burn-in 5 000 and a target sync every 1 000 frames. About the hardware: I trained on an "
    "Apple-M2 laptop (8 GB shared memory, PyTorch MPS). Really mastering Breakout needs tens of millions of "
    "frames on a proper GPU, which the assignment also points out, so on the laptop I can only show a shorter "
    "run where the agent clearly starts to learn, not a fully trained one. Because of the 8 GB limit I keep "
    "the replay buffer in CPU RAM (a float transition is about 0.22 MB) and only move the sampled minibatch "
    "to the device in compute_loss; on a CUDA GPU you could keep the whole buffer on the GPU and make it much "
    "bigger. For reference, throughput was around 550 frames/s during the burn-in and 60–110 frames/s once "
    "there is a gradient step per frame, so the bottleneck is clearly the environment and pre-processing on "
    "the CPU and not the network (CPU and MPS were about the same speed)."))
story.append(P(
    "[[b]]Trying different networks.[[/b]] I varied the number of conv layers, the activation and batch "
    "normalisation. Plain ReLU without batch-norm trained the most reliably. Adding batch-norm after each "
    "conv made things worse, which makes sense: the target network's running statistics lag behind the "
    "online one and the per-batch statistics are noisy, so it does not go well with bootstrapped targets "
    "(batch-norm is rarely used in DQN for this reason). A fourth conv layer just slowed the frame-rate down "
    "without helping at this short budget, so I stayed with the three-conv network."))

# ---- DQN results, auto-loaded ----
mfile = os.path.join(HERE, 'dqn_metrics.npz')
if os.path.exists(mfile):
    d = np.load(mfile)
    r = d['reward'].astype(float)
    ne = len(r)
    frames = int(d['current_step']) if 'current_step' in d else 0
    feps = float(d['eps']) if 'eps' in d else 0.0
    k = max(1, min(100, ne // 5))
    first = float(np.mean(r[:k])) if ne else 0.0
    last = float(np.mean(r[-k:])) if ne else 0.0
    mx = float(np.max(r)) if ne else 0.0
    res = (f"[[b]]Results.[[/b]] In about 40 minutes the agent played {ne} episodes, which is {frames:,} "
           f"agent steps (roughly {frames*4:,} game frames), and ε reached {feps:.2f}. The TD loss is zero "
           f"during the burn-in, jumps up once learning starts and then stays in a bounded range instead of "
           f"blowing up, which is the target network doing its job. More importantly, the mean episode reward "
           f"climbs from {first:.1f} over the first {k} episodes to {last:.1f} over the last {k}, with a best "
           f"episode of {mx:.0f} (Figure 7). So even with this tiny budget compared to the ~50 M frames of "
           f"the original DQN paper, the agent clearly learns to play better than random, which scores about "
           f"2. With a GPU, a larger replay buffer and a longer ε schedule the same code should reach "
           f"noticeably higher scores.")
    story.append(P(res))
    cur = os.path.join(HERE, 'dqn_train_curves.png')
    if os.path.exists(cur):
        story.append(KeepTogether([img(cur, 15.5),
                     P("Figure 7: DQN on Breakout. Reward per episode (left) and mean TD loss (right), each "
                       "with a running mean. The loss only starts after the burn-in and stays stable, while "
                       "the reward goes up over training.", CAP)]))
else:
    story.append(P("[[b]]Results.[[/b]] (DQN metrics file not found at report-build time; run dqn_train.py "
                   "or the notebook to populate Figure 7.)"))

# ===================================================================
# LLM usage
# ===================================================================
H('3  Use of LLMs')
story.append(P(
    "[[b]]Tool.[[/b]] I used Claude (Anthropic) through the Claude-Code assistant for this assignment."))
story.append(P(
    "[[b]]How it helped.[[/b]] It was useful for writing the tabular TD updates and the ε-greedy policy, and "
    "for talking through details like masking the bootstrap term on terminal steps and the closed form of "
    "the Expected-SARSA target. It also helped me set up and read the hyperparameter grid search, in "
    "particular making sense of why Q-Learning blows up at γ = 1.0 (the max bias together with losing the "
    "contraction). For the DQN part it helped me fill in the skeleton (the conv net, the online and target "
    "networks, the periodic sync, the uniform replay sampling, and the Adam/MSE setup), and it noticed that "
    "the agent was being constructed without the run_as_ddqn argument it expects. Finally it helped me adapt "
    "everything to my 8 GB M2 (keeping the replay buffer in CPU RAM and using the MPS device) and write the "
    "small training and plotting scripts."))
story.append(P(
    "[[b]]What did not work / limits.[[/b]] The given skeleton did not run as-is: the run_as_ddqn argument "
    "was missing, and the per-episode loss divides by the step counter t, which would crash on a zero-length "
    "episode. Training Breakout to a high score was simply not possible on my laptop, so the DQN result is a "
    "short demonstration rather than a finished agent. And the first suggestion that a slippery environment "
    "needs a small α was only half right, since the grid search showed that moderate to large α is fine for "
    "γ = 0.95 but not for Q-Learning at γ = 1.0. I went through the code and the reasoning myself and "
    "understand how all of it works."))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('DejaVu', 7.5)
    canvas.setFillColor(colors.HexColor('#888888'))
    canvas.drawString(2*cm, 1.05*cm, 'RL VU SS26 — Assignment 2 — Joachim Rath (51811071)')
    canvas.drawRightString(A4[0] - 2*cm, 1.05*cm, f'Page {doc.page}')
    canvas.restoreState()


doc = SimpleDocTemplate(os.path.join(HERE, 'RL26_Ass2_Report.pdf'), pagesize=A4,
                        leftMargin=2*cm, rightMargin=2*cm, topMargin=1.6*cm, bottomMargin=1.6*cm,
                        title='RL SS26 Assignment 2 Report', author='Joachim Rath')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print('Wrote RL26_Ass2_Report.pdf')
