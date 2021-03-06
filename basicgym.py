# -*- coding: utf-8 -*-
"""
    The following code draws from this blog: https://keras.io/examples/rl/ddpg_pendulum/
"""


import gym
import pybullet_envs
import ECE239AS_Envs

import os
from sys import argv
from getopt import getopt

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

class StepTrigger:
    """
        Activate 'num' times out of 'every' steps
    """
    def __init__(self, every, num=1):
        self.counter = 0
        self.max = every

        if every < num:
            raise Exception("StepTrigger: invalid value (num must be <= every)")

        # 0 should always be part of the active set
        self.active_set = set([0])
        out = every - num
        num -= 1

        # Add 'num' instances to 'active_set'
        for i in range(1, every):
            if num > out:
                self.active_set.add(i)
                num -= 1
            else:
                out -= 1

    def reset(self):
        self.counter = 0

    def step(self):
        self.counter += 1
        if self.counter == self.max:
            self.counter = 0

    def active(self):
        return self.counter in self.active_set

class Bounds:
    def __init__(self, lower, upper):
        if upper < lower:
            lower, upper = upper, lower
        self.lower = lower
        self.upper = upper
    def __call__(self, values):
        return np.clip(values, self.lower, self.upper)

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

def get_actor(num_states, num_actions):
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(ActorNN, activation="relu")(inputs)
    out = layers.Dense(ActorNN, activation="relu")(out)
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model

def get_critic(num_states, num_actions):
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(CriticNN, activation="relu")(concat)
    out = layers.Dense(CriticNN, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

class DDPG:
    def __init__(self, num_states, num_actions, action_bound, actor_lr=0.001, critic_lr=0.002, gamma=0.99, tau=0.005, buffer_size=500_000):
        self.action_bound = action_bound

        # Create set of actor networks
        self.actor = get_actor(num_states, num_actions)
        self.actor.optim = tf.keras.optimizers.Adam(actor_lr)
        self.target_actor = get_actor(num_states, num_actions)
        self.target_actor.set_weights(self.actor.get_weights())

        # Create set of critic networks
        self.critic = get_critic(num_states, num_actions)
        self.critic.optim = tf.keras.optimizers.Adam(critic_lr)
        self.target_critic = get_critic(num_states, num_actions)
        self.target_critic.set_weights(self.critic.get_weights())

        # Training parameters
        self.gamma = gamma
        self.tau = tau

        self.buffer = Buffer(num_states, num_actions, buffer_size, 64)

    def policy(self, state, noise_object):
        sampled_actions = tf.squeeze(self.actor(state))
        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = self.action_bound(sampled_actions)

        return [np.squeeze(legal_action)]

    def record(self, prev_state, action, reward, state, done):
        self.buffer.record((prev_state, action, reward, state, 0.0 if done else 1.0))

    # Get predicted actions that target network would make
    def _get_target_actions(self, states, training):
        return self.target_actor(states, training=training)

    # Evaluate target critic network
    def _get_target_values(self, states, actions):
        return self.target_critic([states, actions], training=True)

    # Update target networks to approach current networks
    def update_targets(self):
        update_target(self.target_actor.variables, self.actor.variables, self.tau)
        update_target(self.target_critic.variables, self.critic.variables, self.tau)

    def save(self, dir, problem, output_csv, avg_reward_list):
        # Construct name of model folder, create it if it does not exist
        model_name = f'{dir}/{type(self).__name__}-{problem}'
        os.path.isdir(model_name) or os.mkdir(model_name)

        # Create unique folder for this trial
        for i in range(1, 100000):
            path = f'{model_name}/No{i}'
            if (os.path.isdir(path) or os.mkdir(path)) is None: break
        print('Saving model to', path)

        # Create results file and write csv values
        with open(f'{path}/{round(np.max(avg_reward_list),2)}', 'w') as f:
            for arr in output_csv:
                print(*arr, sep=', ', file=f)

        # Write models to directory
        self.actor.save(f'{path}/actor')
        self.critic.save(f'{path}/critic')

        # Return path so child classes can use it
        return path

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def learn(self, batch, skip_actor=False):
        # Extract batches from batch dict
        keys = 'states', 'actions', 'rewards', 'next_states', 'dones'
        states, actions, rewards, next_states, dones = [batch[key] for key in keys]

        # Get Q-values of next_states (using target actor actions)
        target_actions = self._get_target_actions(next_states, training=True)
        y = rewards + dones * self.gamma * self._get_target_values(next_states, target_actions)

        # Regress critic_model toward targets
        with tf.GradientTape() as tape:
            critic_value = self.critic([states, actions], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optim.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )


        # Return early (DO NOT update actor)
        if skip_actor: return y

        # TD3-NOTE: Still use critic_model (1) to optimize policy.
        with tf.GradientTape() as tape:
            actor_actions = self.actor(states, training=True)
            critic_value = self.critic([states, actor_actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optim.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )
        return y

    def train(self):
        experiences = self.buffer.get_batch()
        self.learn(experiences)
        self.update_targets()

class TD3(DDPG):
    def __init__(self, num_states, num_actions, action_bound, actor_lr=0.001, critic_lr=0.002, gamma=0.99, tau=0.005):
        super().__init__(num_states, num_actions, action_bound, actor_lr, critic_lr, gamma, tau)

        self.critic2 = get_critic(num_states, num_actions)
        self.critic2.optim = tf.keras.optimizers.Adam(critic_lr)
        self.target_critic2 = get_critic(num_states, num_actions)
        self.target_critic2.set_weights(self.critic2.get_weights())

        self.action_noise = 0.05 #0.01 #0.1
        self.minimize_target_values = True

        self.update_trigger = StepTrigger(every=4, num=2)

    def _get_target_actions(self, states, training=True):
        # Start with the same target actions as DDPG algorithm
        DDPG_target_actions = super()._get_target_actions(states, training)
        shape = DDPG_target_actions.shape
        # Add mean-0 noise
        action_noise =  tf.random.normal(shape, stddev=self.action_noise) #05) #0.01)
        return DDPG_target_actions + action_noise

    def _get_target_values(self, states, actions):
        # Get Q-values from DDPG
        DDPG_values = super()._get_target_values(states, actions)
        if not self.minimize_target_values:
            return DDPG_values
        else:
            # Calculate Q-values according to second critic network
            critic2_values = self.target_critic2([states, actions], training=True)
            # Return minimum (to help prevent Q-value overestimatino)
            return tf.math.minimum(DDPG_values, critic2_values)

    def update_targets(self):
        # Return early if update trigger is not active
        if not self.update_trigger.active(): return
        super().update_targets()
        update_target(self.target_critic2.variables, self.critic2.variables, self.tau)

    def save(self, *args):
        path = super().save(*args)
        self.critic2.save(f'{path}/critic2')

    @tf.function
    def learn(self, batch):
        # Update critic and maybe actor
        y = super().learn(batch, not self.update_trigger.active())

        # Update critic2
        with tf.GradientTape() as tape:
            critic_value = self.critic2([batch['states'], batch['actions']], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, self.critic2.trainable_variables)
        self.critic2.optim.apply_gradients(zip(critic_grad, self.critic2.trainable_variables))

        self.update_trigger.step()

class HIRO:

    def __init__(self, num_states, num_actions, action_bounds, actor_lr=0.001, critic_lr=0.002, gamma=0.99, tau=0.005):
        # Number of low-level actions between high-level actions
        self.period = 20
        self.pretrain = False

        # Instantiate hierarchical algorithms
        self.lo_algo = DDPG(num_states*2, num_actions, action_bounds, actor_lr=0.001, critic_lr=0.001, gamma=0.99, tau=0.002)
        self.hi_algo = DDPG(num_states, num_states, action_bounds, actor_lr=0.001, critic_lr=0.002, gamma=0.99, tau=0.005, buffer_size=2_000)

        self.lo_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.1) * np.ones(1))
        self.hi_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.1) * np.ones(1))

        self.hi_trigger = StepTrigger(every=self.period, num=1)
        # Buffer including sequences of states, goals, actions, rewards, and final state
        self.hi_buffer = [[[], [], [], [], None]]
        self.lo_rewards = []

    def _goal_transition_func(self, state, goal, next_state):
        # state + goal = next_state + next_goal
        state = np.array(state)
        next_goal = goal + (state - next_state)
        return np.reshape(next_goal, (1,-1))

    def _reward(self, state, goal, action, next_state):
        state = np.array(state)
        diff = state + goal - next_state
        # Ignore theta_dot, x_dot, and x_target
        mask = np.array([1,0,1,0,0]).reshape(1,-1)
        return -np.linalg.norm(diff * mask)

    def _squash_hiexp(self, hi_exp, off_policy_correction=False):
        states, goals, actions, rewards, next_state = hi_exp
        return states[0], goals[0], np.sum(rewards), next_state

    def save(self, *args):
        print('Have not implemented saving yet')

    def pretrain(self, env, noise):
        # Pre-train lower level network for 2M steps
        random_goal = lambda : np.random.normal(size=len(prev_state), scale=0.0)
        length = 100_000
        prev_state = env.reset()
        prev_goal = random_goal()
        reward_list = []
        try:
            for i in range(length):
                if i % 1000 == 0: print('Pretrain step', i, '/', length, np.mean(reward_list[-1000:]))
                lo_state = np.append(prev_state, prev_goal)
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(lo_state), 0)
                action = self.lo_algo.policy(tf_prev_state, noise)

                # Interact with environment and record experience
                state, __reward, done, __info = env.step(action)
                reward = self._reward(prev_state, prev_goal, action, state)
                goal = self._goal_transition_func(prev_state, prev_goal, state)
                self.lo_algo.record(lo_state, action, reward, np.append(state, goal), 0.0 if done else 1.0)
                prev_state = state
                prev_goal = goal

                reward_list.append(reward)

                # Offline Experience Replay
                self.lo_algo.train()

                if done:
                    prev_state = env.reset()
                    prev_goal = random_goal()
        except KeyboardInterrupt:
            pass
        with open('output', 'w') as f:
            print(reward_list, sep='\n', file=f)

    def policy(self, state, noise, pretrain=False):
        # Create new goal from hi-network
        if self.hi_trigger.active():
            self.prev_goal = np.reshape(self.hi_algo.policy(state, self.hi_noise), (1,-1))
            self.prev_state = state
            self.pretrain = pretrain
            if pretrain:
                self.prev_goal = np.random.normal(size=state.shape, scale=0.2)
            # self.prev_goal = np.zeros_like(state).reshape(1,-1)
            # print('New Goal: ', self.prev_goal.flatten())
        # print(self.prev_goal, self.prev_state)
        # Transition goal to keep target (state + goal) fixed
        goal = self._goal_transition_func(self.prev_state, self.prev_goal, state)
        # Problem-specific domain knowledge says pay attention only to x
        mask = np.array([1,0,1,0,0]).reshape(1,-1)
        goal = goal * mask
        # print(goal)
        lo_state = tf.concat([state, goal], 1)
        # print(type(lo_state), np.shape(lo_state), lo_state)
        # Prompt lo-network for atomic action (on Env)
        action = self.lo_algo.policy(lo_state, self.lo_noise)
        return action

    def record(self, prev_state, action, reward, state, done):
        self.hi_trigger.step()

        prev_state = np.reshape(prev_state, (1,-1))
        state = np.reshape(state, (1,-1))

        prev_goal = self.prev_goal
        lo_reward = self._reward(prev_state, prev_goal, action, state) if not done else reward
        # if done: lo_reward = -10.0
        # else: lo_reward = reward
        # lo_reward = reward
        next_goal = self._goal_transition_func(prev_state, prev_goal, state)
        self.lo_rewards.append(np.round(lo_reward, 2))

        lo_prev_state = tf.concat([prev_state, prev_goal], 1)
        lo_state = tf.concat([state, next_goal], 1)
        self.lo_algo.record(lo_prev_state, action, lo_reward, lo_state, done)

        # Don't collect experiences while low-level controller is figuring things out
        if not self.pretrain:
            self.hi_buffer[-1][0].append(prev_state)
            self.hi_buffer[-1][1].append(prev_goal)
            self.hi_buffer[-1][2].append(action)
            self.hi_buffer[-1][3].append(reward)

        # If done_val indicates that the trial has terminated
        if done:
            print('    Lo_Reward:', np.round(np.mean(self.lo_rewards),2), end='\t')
            self.lo_rewards = []
            self.hi_trigger.reset()
            self.lo_noise.std_dev = np.maximum(0.005, self.lo_noise.std_dev * 0.98)

            # Not using higher network, leave noise alone
            if not self.pretrain:
                self.hi_noise.std_dev = np.maximum(0.01, self.hi_noise.std_dev * 0.98)

        # Time to update hi_algo
        if self.hi_trigger.active() and not self.pretrain:
            self.hi_buffer[-1][4] = state

            # It's time to package the high-level experiences up.
            # Low-level network is relatively fresh so there's no need to correct for it
            hiexp = self._squash_hiexp(self.hi_buffer[-1], off_policy_correction=False)
            self.hi_algo.record(*hiexp, done)
            # Setup new hi-level list of low-level experiences
            self.hi_buffer.append([[],[],[],[],None])

        self.prev_goal = next_goal
        self.prev_state = state

    def train(self):
        if self.hi_algo.buffer.buffer_counter > 0:
            self.hi_algo.train()
        self.lo_algo.train()


class Buffer:
    def __init__(self, num_states, num_actions, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))

    def reset(self):
        self.buffer_counter = 0

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]

        self.buffer_counter += 1

    # Return batch of examples, use these for algorithm learning
    def get_batch(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        done_batch = tf.convert_to_tensor(self.done_buffer[batch_indices])
        done_batch = tf.cast(done_batch, dtype=tf.float32)

        return {
            'states': state_batch,
            'actions': action_batch,
            'rewards': reward_batch,
            'next_states': next_state_batch,
            'dones': done_batch
        }

envs_pyb = ["InvertedPendulumBulletEnv-v0",
            "CartPoleContinuousBulletEnv-v0",
            "CartPoleWobbleContinuousEnv-v0",
            "ReacherBulletEnv-v0"]
# problem = "Pendulum-v0"
# problem = "MountainCarContinuous-v0"
# problem = "Acrobot-v1"
problem = envs_pyb[2]
env = gym.make(problem)

def get_env_details(env):
    num_states = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))

    upper_bound = +1.0 #env.action_space.high[0]
    lower_bound = -1.0 #env.action_space.low[0]

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))

    return num_states, num_actions, lower_bound, upper_bound
num_states, num_actions, lower_bound, upper_bound = get_env_details(env)
# num_states *= 2

opt, args = getopt(argv[1:], "", ["TD3", "HIRO", "ActorNN=", "CriticNN="])
opt = dict(opt)

AlgoName = "DDPG"
if "--TD3" in opt: AlgoName = "TD3"
if "--HIRO" in opt: AlgoName = "HIRO"
ActorNN = int(opt.get('--ActorNN',32))
CriticNN = int(opt.get('--CriticNN',32))

# Construct noise object
std_dev = 0.5 #1.5
min_std_dev = 0.01
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

# Instantiate Algorithm object
action_bounds = Bounds(lower_bound, upper_bound)
_algo_cls = globals()[AlgoName]
algo = _algo_cls(num_states, num_actions, action_bounds, actor_lr=0.005, critic_lr=0.01, gamma=0.99, tau=0.005)

# if AlgoName == "HIRO":
#     algo.pretrain(env, ou_noise)
# raise SystemExit

# Store reward history of each episode, and averages over last 40
ep_reward_list = []
avg_reward_list = []

total_episodes = 2_000
output_csv = [["ActorNN",ActorNN,"CriticNN",CriticNN]]
output_csv.append(["Ep","Reward","AvgReward40"])
try:
    # Takes about 4 min to train
    for ep in range(total_episodes):
        pretrain = ep < 500

        if problem in envs_pyb: env.render()
        prev_state = env.reset()
        # prev_state = np.append(prev_state, np.zeros_like(prev_state))
        episodic_reward = 0

        moves = []
        for step in range(10000000):
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            if ep % 5 == 0: env.render()
            # env.render()

            # Get move from algorithm
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            # print(type(tf_prev_state), np.shape(tf_prev_state), tf_prev_state)
            action = algo.policy(tf_prev_state, ou_noise, pretrain)

            moves.append(action)

            # Interact with environment and record experience
            state, reward, done, info = env.step(action)
            # state = np.append(state, np.zeros_like(state))
            algo.record(prev_state, action, reward, state, done)
            episodic_reward += reward
            prev_state = state

            # Offline Experience Replay
            algo.train()

            # End this episode when `done` is True
            if done: break

        # Mean of last 40 episodes
        ep_reward_list.append(episodic_reward)
        avg_reward = np.mean(ep_reward_list[-40:])
        print('\n\t',
              "PRETRAIN" if pretrain and AlgoName == "HIRO" else "",
              "{:4}".format(round(episodic_reward,2)),
              ": {}/{}".format(np.round(algo.lo_noise.std_dev, 2), np.round(algo.hi_noise.std_dev, 2)) if AlgoName == "HIRO" else ": {:5}".format(np.round(ou_noise.std_dev,2)),
              "E_{} R_{:5}. Move/{} {:5} with mag {:5}".
                format(ep, np.round(avg_reward,2), len(moves), round(np.mean(moves),2), round(np.mean(np.abs(moves)),2)), end='')
        avg_reward_list.append(avg_reward)
        output_csv.append([ep, round(episodic_reward,2), round(avg_reward,2)])

        # Decrease noise
        ou_noise.std_dev = np.maximum(min_std_dev, ou_noise.std_dev * 0.98)

except KeyboardInterrupt:
    pass

# Save model to models/ directory
os.path.isdir('models') or os.mkdir('models')
algo.save('models', problem, output_csv, avg_reward_list)
