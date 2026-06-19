"""Patch dqn_atari.ipynb: fill in all TODOs, clear stale outputs.

Whole-cell replacements for cells 2, 4, 5, 7, 8 and a targeted edit for the
device selection in cell 3. Run once; safe to re-run (idempotent via markers).
"""
import json

NB = '/Users/Joachim/Downloads/Reinforcement_Learning_Assignment2/dqn_atari.ipynb'

cell2 = r'''class SkipFrame(gym.Wrapper):
    def __init__(self, env, num_skip):
        super().__init__(env)
        self.num_skip = num_skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.num_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.float32)

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        transform = torchvision.transforms.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.float32)

    def observation(self, observation):
        transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(self.shape),
                                                     torchvision.transforms.Normalize(0, 255)])
        return transforms(observation).squeeze(0)


class ExperienceReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def store(self, state, next_state, action, reward, terminated, truncated, device):

        state = torch.as_tensor(state.__array__(), device=device)
        next_state = torch.as_tensor(next_state.__array__(), device=device)
        action = torch.as_tensor(action, device=device, dtype=torch.long)
        reward = torch.as_tensor(reward, device=device)
        terminated = torch.as_tensor(terminated, device=device)
        truncated = torch.as_tensor(truncated, device=device)
        self.memory.append((state, next_state, action, reward, terminated, truncated))


    def sample(self, batch_size):
        # Uniformly sample a minibatch of transitions from the cyclic buffer,
        # i.e. (s, s', a, r, term, trunc) ~ Uniform(B), and stack each component
        # into a single batched tensor. The reward is cast to float so it matches
        # the network output dtype in the MSE loss.
        batch = random.sample(self.memory, batch_size)
        state, next_state, action, reward, terminated, truncated = zip(*batch)
        return (torch.stack(state), torch.stack(next_state), torch.stack(action),
                torch.stack(reward).float(), torch.stack(terminated), torch.stack(truncated))'''

cell4 = r'''class DeepQNet(torch.nn.Module):
    def __init__(self, h, w, image_stack, num_actions,
                 activation=torch.nn.ReLU, use_batchnorm=False):
        super(DeepQNet, self).__init__()
        # Convolutional feature extractor following Mnih et al. (2015). The conv
        # stack maps the (image_stack x h x w) stack of frames to a feature map.
        # `activation` and `use_batchnorm` are exposed so we can easily explore
        # different network configurations (see report).
        def conv_block(in_c, out_c, k, s):
            layers = [torch.nn.Conv2d(in_c, out_c, kernel_size=k, stride=s)]
            if use_batchnorm:
                layers.append(torch.nn.BatchNorm2d(out_c))
            layers.append(activation())
            return layers

        self.features = torch.nn.Sequential(
            *conv_block(image_stack, 32, 8, 4),
            *conv_block(32, 64, 4, 2),
            *conv_block(64, 64, 3, 1),
        )
        # Infer the flattened conv-output size with a dummy forward pass so the
        # fully-connected head adapts automatically to the input resolution.
        with torch.no_grad():
            n_flatten = self.features(torch.zeros(1, image_stack, h, w)).reshape(1, -1).shape[1]

        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_flatten, 512),
            activation(),
            torch.nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


# The online network is optimised by gradient descent; the target network is a
# periodically-synced copy whose parameters carry no gradient.
online_dqn = DeepQNet(h, w, image_stack, num_actions)
target_dqn = copy.deepcopy(online_dqn).requires_grad_(False)
online_dqn.to(device)
target_dqn.to(device)'''

cell5 = r'''def convert(x):
    return torch.tensor(x.__array__()).float()


class AtariAgent:
    def __init__(self, buffer, eps, eps_decay, min_eps, gamma, batch_size,
                 online_dqn, target_dqn, run_as_ddqn,
                 optimizer, criterion, device,
                 max_train_frames, burn_in_phase, sync_target):

        self.buffer = buffer
        self.eps = eps
        self.eps_decay = eps_decay
        self.min_eps = min_eps
        self.gamma = gamma
        self.batch_size = batch_size

        self.online_dqn = online_dqn
        self.target_dqn = target_dqn
        self.run_as_ddqn = run_as_ddqn
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.max_train_frames = max_train_frames
        self.burn_in_phase = burn_in_phase
        self.sync_target = sync_target

        self.current_step = 0
        # Replay transitions are kept in CPU memory (the 8 GB unified memory of
        # the test machine cannot hold the buffer on the GPU); sampled minibatches
        # are moved to self.device inside compute_loss.
        self.buffer_device = torch.device("cpu")


    def policy(self, state, is_training):
        state = convert(state).unsqueeze(0).to(self.device)

        # Epsilon-greedy action selection: during training we explore with
        # probability self.eps, otherwise we act greedily wrt. the online network.
        # During testing (is_training=False) we always take the greedy action.
        if is_training and random.random() < self.eps:
            return torch.tensor(random.randrange(num_actions), dtype=torch.long)
        with torch.no_grad():
            q_values = self.online_dqn(state)
        return q_values.argmax(dim=1).squeeze(0).cpu()


    def compute_loss(self, state, action, reward, next_state, truncated, terminated):
        state, action, reward, next_state, truncated, terminated = [x.to(self.device) for x in
                                (state, action, reward, next_state, truncated, terminated)]
        """ Computes the DQN (or DDQN) target y and with it the loss based on self.criterion """

        done = truncated | terminated # bitwise or
        q_s_a = self.online_dqn(state).gather(1, action.view(-1, 1)).squeeze(1)

        if self.run_as_ddqn:
            # Double DQN: the online net selects the next action, the target net
            # evaluates it -- this reduces the maximisation bias of vanilla DQN.
            next_actions = self.online_dqn(next_state).argmax(dim=1, keepdim=True)
            q_sprime_aprime = self.target_dqn(next_state).gather(1, next_actions).squeeze(1)
        else:
            q_sprime_aprime = torch.max(self.target_dqn(next_state), dim=1).values
        y = reward + (1 - done.float()) * self.gamma * q_sprime_aprime.detach()

        return self.criterion(q_s_a, y)


    def run_episode(self, is_training):
        episode_reward, episode_loss = 0, 0.
        state, _ = env.reset()

        for t in range(self.max_train_frames):
            action = self.policy(state, is_training)
            self.current_step += 1
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward

            if is_training:
                self.buffer.store(state, next_state, action, reward, terminated, truncated, self.buffer_device)

                if self.current_step > self.burn_in_phase:
                    state_batch, next_state_batch, action_batch, \
                        reward_batch, terminated_batch, truncated_batch = self.buffer.sample(self.batch_size)

                    if self.current_step % self.sync_target == 0:
                        # Periodically copy the online weights into the (frozen)
                        # target network. Keeping the bootstrap targets fixed
                        # between syncs stabilises the optimisation.
                        print(f'syncing target network at step {self.current_step}')
                        self.target_dqn.load_state_dict(self.online_dqn.state_dict())

                    loss = self.compute_loss(state_batch, action_batch, reward_batch,
                                             next_state_batch, terminated_batch, truncated_batch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    episode_loss += loss.detach().item()
            else:
                with torch.no_grad():
                    st = convert(state).to(self.device).unsqueeze(0)
                    next_st = convert(next_state).to(self.device).unsqueeze(0)
                    act = action.to(self.device)
                    rew = torch.tensor(reward).to(self.device)
                    trunc = torch.tensor(truncated).to(self.device)
                    term = torch.tensor(terminated).to(self.device)

                    episode_loss += self.compute_loss(st, act, rew, next_st, term, trunc).item()

            state = next_state

            if self.current_step > self.burn_in_phase and self.eps > self.min_eps:
                self.eps *= self.eps_decay

            if terminated or truncated:
                break

        return dict(reward=episode_reward, loss=episode_loss / t)


    def save_checkpoint(self, train_metrics, save_filename):
        save_dict = {'curr_step': self.current_step,
                    'train_metrics': train_metrics,
                    'eps': self.eps,
                    'online_dqn': self.online_dqn.state_dict(),
                    'target_dqn': self.target_dqn.state_dict()}

        torch.save(save_dict, save_filename)'''

cell7 = r'''# Hyperparameters (tuned for a single-GPU / Apple-M2 run; see report)
batch_size = 32
alpha = 1e-4                       # Adam learning rate
gamma = 0.99
eps, eps_decay, min_eps = 1.0, 0.99995, 0.1   # eps decays per (post-burn-in) frame
buffer = ExperienceReplayMemory(10_000)        # kept in CPU RAM (8 GB machine)
burn_in_phase = 5_000              # collect random experience before learning
sync_target = 1_000                # copy online -> target every 1000 frames
max_train_frames = 5_000           # max env steps per episode
max_train_episodes = 1_000
max_test_episodes = 50
run_as_ddqn = False # Set the run_as_ddqn flag to True if you want to run the DDQN algorithm
save_filename = './saved_model.pt'

# Mean-squared-error TD loss and Adam optimiser on the online network parameters.
optimizer = torch.optim.Adam(online_dqn.parameters(), lr=alpha)
criterion = torch.nn.MSELoss()


testing_mode = False # Change to True if you want to load a saved model

if testing_mode:
    load_dict = torch.load(save_filename, map_location='cpu')
    eps = load_dict['eps']
    online_dqn.load_state_dict(load_dict['online_dqn'], strict=True)
    online_dqn.eval()
    target_dqn = copy.deepcopy(online_dqn).requires_grad_(False)
    target_dqn.eval()


agent = AtariAgent(buffer=buffer, eps=eps, eps_decay=eps_decay, min_eps=min_eps, gamma=gamma, batch_size=batch_size,
                   online_dqn=online_dqn, target_dqn=target_dqn, run_as_ddqn=run_as_ddqn,
                   optimizer=optimizer, criterion=criterion, device=device,
                   max_train_frames=max_train_frames, burn_in_phase=burn_in_phase, sync_target=sync_target)

if testing_mode:
    test_metrics = dict(reward=[], loss=[])
    for it in range(max_test_episodes):
        episode_metrics = agent.run_episode(is_training=False)
        update_metrics(test_metrics, episode_metrics)
        print_metrics(it + 1, test_metrics, is_training=False)
else:
    train_metrics = dict(reward=[], loss=[])
    for it in range(max_train_episodes):
        episode_metrics = agent.run_episode(is_training=True)
        update_metrics(train_metrics, episode_metrics)
        if it % 50 == 0:
            print_metrics(it, train_metrics, is_training=True)
            agent.save_checkpoint(train_metrics, save_filename)'''

cell8 = r'''# Plot the training (or testing) reward and TD loss per episode. Because both
# signals are noisy we additionally show a running mean.
import numpy as np
import matplotlib.pyplot as plt


def running_mean(x, w):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    w = max(1, min(w, len(x)))
    return np.convolve(x, np.ones(w) / w, mode='valid')


def plot_metrics(metrics, title_prefix, smooth=20, save_path=None):
    rewards, losses = metrics['reward'], metrics['loss']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(rewards, color='tab:blue', alpha=0.3, label='per episode')
    sm = running_mean(rewards, smooth)
    axes[0].plot(np.arange(len(sm)) + (len(rewards) - len(sm)), sm,
                 color='tab:blue', lw=2, label=f'running mean ({smooth})')
    axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Accumulated reward')
    axes[0].set_title(f'{title_prefix}: reward per episode'); axes[0].legend()

    axes[1].plot(losses, color='tab:red', alpha=0.3, label='per episode')
    sm = running_mean(losses, smooth)
    axes[1].plot(np.arange(len(sm)) + (len(losses) - len(sm)), sm,
                 color='tab:red', lw=2, label=f'running mean ({smooth})')
    axes[1].set_xlabel('Episode'); axes[1].set_ylabel('Mean TD loss')
    axes[1].set_title(f'{title_prefix}: training loss per episode'); axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.show()


if not testing_mode:
    plot_metrics(train_metrics, 'DQN training', smooth=20, save_path='dqn_train_curves.png')
else:
    plot_metrics(test_metrics, 'DQN testing', smooth=5, save_path='dqn_test_curves.png')'''


def to_lines(src: str):
    lines = src.split('\n')
    return [l + '\n' for l in lines[:-1]] + [lines[-1]]


nb = json.load(open(NB))
cells = nb['cells']

# whole-cell replacements
replacements = {2: cell2, 4: cell4, 5: cell5, 7: cell7, 8: cell8}
for idx, src in replacements.items():
    cells[idx]['source'] = to_lines(src)

# targeted edit: device selection in cell 3 (add MPS support)
src3 = ''.join(cells[3]['source'])
old_dev = 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'
new_dev = ('if torch.cuda.is_available():\n'
           '    device = torch.device("cuda")\n'
           'elif torch.backends.mps.is_available():\n'
           '    device = torch.device("mps")  # Apple-Silicon GPU acceleration\n'
           'else:\n'
           '    device = torch.device("cpu")')
assert old_dev in src3, 'device line not found in cell 3'
cells[3]['source'] = to_lines(src3.replace(old_dev, new_dev))

# clear all outputs / execution counts for a clean notebook
for c in cells:
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None

json.dump(nb, open(NB, 'w'), indent=1)
print('Patched notebook written. Cells:', len(cells))
for i, c in enumerate(cells):
    head = (''.join(c['source']).strip().split('\n')[0] if c['source'] else '(empty)')
    print(i, c['cell_type'], '|', head[:70])
