
from os import listdir
from os.path import isdir, join

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def model_summary(model_path):
    if not isdir(model_path):
        return
    model = tf.keras.models.load_model(model_path)
    print('======', model_path, '======')
    print(model.summary())

def CartPole(dir, name):
    path = join(dir, name)
    files = listdir(path)
    for fname in files:
        try:
            score = float(fname)
            break
        except ValueError:
            pass
    with open(join(path, fname)) as f:
        results = np.array([line.split(',') for line in f])
    trials = np.array(results[2:,0], dtype=float)
    avg_reward = np.array(results[2:,2], dtype=float)
    plt.plot(trials[::10], avg_reward[::10], label=name[:name.find('-')])
    # model_summary(f'{path}/actor')
    # model_summary(f'{path}/critic')
    # model_summary(f'{path}/critic2')

# CartPole('old-models', 'DDPG-CartPoleContinuousBulletEnv-v0_No5')
# CartPole('old-models', 'TD3-CartPoleContinuousBulletEnv-v0_No8')
# plt.title('CartPoleContinuousBulletEnv')

CartPole('old-models', 'DDPG-CartPoleWobbleContinuousEnv-v0_No9')
CartPole('old-models', 'TD3-CartPoleWobbleContinuousEnv-v0_No22')
plt.title('CartPoleWobbleContinuousEnv')

plt.legend()
plt.show()

# input()
