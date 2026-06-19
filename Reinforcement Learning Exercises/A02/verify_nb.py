"""Verify dqn_atari.ipynb runs sequentially without errors, using a tiny config
(few episodes) so it finishes in seconds. Execs cells 1..8; overrides the
training budget in cell 7. Proves the full pipeline incl. plotting works."""
import json, matplotlib
matplotlib.use('Agg')
NB = '/Users/Joachim/Downloads/Reinforcement_Learning_Assignment2/dqn_atari.ipynb'
nb = json.load(open(NB))


def strip_magics(src):
    return '\n'.join(l for l in src.split('\n') if not l.lstrip().startswith(('%', '!')))


g = {'__name__': '__main__'}
for idx in range(1, 9):
    src = strip_magics(''.join(nb['cells'][idx]['source']))
    if idx == 7:
        src = (src.replace('max_train_episodes = 1_000', 'max_train_episodes = 3')
                  .replace('burn_in_phase = 5_000', 'burn_in_phase = 40')
                  .replace('max_train_frames = 5_000', 'max_train_frames = 80')
                  .replace('sync_target = 1_000', 'sync_target = 30')
                  .replace('max_test_episodes = 50', 'max_test_episodes = 2'))
    exec(compile(src, f'<cell{idx}>', 'exec'), g)
    print(f'cell {idx}: OK')
print('ALL CELLS RAN WITHOUT ERRORS')
