"""Smoke test: exec the notebook's code cells and run a few mini-episodes to
verify the DQN pipeline runs end-to-end (sampling, target sync, loss, backward,
and the test branch). Also benchmarks frames/sec on the chosen device."""
import json, sys, time, types
import torch

NB = '/Users/Joachim/Downloads/Reinforcement_Learning_Assignment2/dqn_atari.ipynb'
FORCE_DEVICE = sys.argv[1] if len(sys.argv) > 1 else None  # 'cpu' | 'mps' | None

nb = json.load(open(NB))
g = {'__name__': '__main__'}

# exec cells 1..6 (skip cell 0 = %pip magic). Cell 3 builds the env + device.
def strip_magics(src):
    return '\n'.join(l for l in src.split('\n') if not l.lstrip().startswith(('%', '!')))

for idx in [1, 2, 3, 4, 5, 6]:
    src = strip_magics(''.join(nb['cells'][idx]['source']))
    if FORCE_DEVICE and idx == 3:
        # override device selection for benchmarking
        src += f"\ndevice = torch.device('{FORCE_DEVICE}')\nprint('FORCED device:', device)\n"
    exec(compile(src, f'<cell{idx}>', 'exec'), g)

# rebuild nets on the (possibly forced) device
DeepQNet = g['DeepQNet']
import copy
h, w, image_stack, num_actions, device = g['h'], g['w'], g['image_stack'], g['num_actions'], g['device']
online = DeepQNet(h, w, image_stack, num_actions).to(device)
target = copy.deepcopy(online).requires_grad_(False).to(device)
g['online_dqn'], g['target_dqn'] = online, target

n_params = sum(p.numel() for p in online.parameters())
print(f'device={device}  net params={n_params:,}')

# tiny agent to exercise every branch quickly
AtariAgent = g['AtariAgent']
ERM = g['ExperienceReplayMemory']
agent = AtariAgent(buffer=ERM(2000), eps=1.0, eps_decay=0.999, min_eps=0.1, gamma=0.99,
                   batch_size=32, online_dqn=online, target_dqn=target, run_as_ddqn=False,
                   optimizer=torch.optim.Adam(online.parameters(), lr=1e-4),
                   criterion=torch.nn.MSELoss(), device=device,
                   max_train_frames=400, burn_in_phase=64, sync_target=100)
g['agent'] = agent

# run a couple of training episodes (exercise burn-in, sample, sync, backward)
t0 = time.time()
for ep in range(3):
    m = eval('agent.run_episode(is_training=True)', g)
    print(f'  train ep{ep}: reward={m["reward"]:.1f} loss={m["loss"]:.4f} '
          f'step={agent.current_step} eps={agent.eps:.3f}')
steps = agent.current_step
dt = time.time() - t0
print(f'TRAIN  {steps} frames in {dt:.1f}s -> {steps/dt:.0f} frames/s')

# run a test episode (exercise the eval branch + DDQN off path)
agent.eps = 0.0
mt = eval('agent.run_episode(is_training=False)', g)
print(f'  test ep: reward={mt["reward"]:.1f} loss={mt["loss"]:.4f}')

# quick DDQN path check
agent.run_as_ddqn = True
agent.current_step = agent.burn_in_phase + 1
mt2 = eval('agent.run_episode(is_training=True)', g)
print(f'  ddqn train ep: reward={mt2["reward"]:.1f} loss={mt2["loss"]:.4f}')
print('OK: pipeline runs without errors')
