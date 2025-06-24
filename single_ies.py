import gym
import random
import imageio
import datetime
import numpy as np
from collections import deque
import tensorflow as tf
import os

import matplotlib.pyplot as plt
from matplotlib import rcParams

from tensorflow.keras.optimizers import Adam

# 引入新的 IES 环境
from env_he import CombinedEnergyEnv

from network import Memory,network

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示 ERROR

tf.keras.backend.set_floatx('float32')

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

class C_SAC_h:
    def __init__(
            self,
            env,
            alpha=0.1,
            lambda_=0.1,
            constraint_threshold=0.01,
            lr_actor=5e-3,
            lr_critic=1e-2,
            lr_entropy=1e-4,
            use_priority=True,
            actor_units=(512, 256, 64, 32),
            critic_units=(128, 128, 32),
            tau=1e-3,
            gamma=0.9,
            batch_size=64,
            memory_cap=150000,
            eta_lambda=0.001,
            lambda_max=100.0,
            delta_lambda_min=-0.001,
            delta_lambda_max=0.001,
            target_entropy=-25,
            rmax=2900 * 1e-4
    ):
        self.rmax = rmax
        self.env = env
        self.state_shape = env.observation_space.shape[0]
        self.state_low, self.state_high = self.get_state_space_limits()
        self.state_low = self.state_low.astype(np.float32)
        self.state_high = self.state_high.astype(np.float32)

        self.action_dim_discrete, self.action_dim_continuous, self.action_bound, self.action_shift = \
            self.get_action_space_dimensions(env.action_space)
        self.action_bound = self.action_bound.astype(np.float32)
        self.action_shift = self.action_shift.astype(np.float32)

        self.use_priority = use_priority
        self.memory = Memory(capacity=memory_cap) if use_priority else deque(maxlen=memory_cap)

        self.network = network(
            state_dim=self.state_shape,
            action_dim_continuous=self.action_dim_continuous,
            action_dim_discrete=self.action_dim_discrete,
            action_bound=self.action_bound,
            action_shift=self.action_shift,
            state_low=self.state_low,
            state_high=self.state_high
        )

        # actor 网络
        self.actor = self.network.actor(actor_units)
        self.actor_optimizer = Adam(learning_rate=lr_actor)
        # critic 网络
        self.q1_critic, self.q2_critic = self.network.critic(critic_units)
        self.q1_critic_target, self.q2_critic_target = self.network.critic(critic_units)
        self.critic_optimizer = Adam(learning_rate=lr_critic)
        # constraint critic 网络
        self.qc_critic = self.network.constraint_critic(critic_units)
        self.qc_critic_target = self.network.constraint_critic(critic_units)
        self.qc_optimizer = Adam(learning_rate=lr_critic)

        self.gamma = gamma
        self.tau = np.float32(tau)
        self.batch_size = batch_size

        self.summaries = {}

        self.alpha = tf.Variable(alpha * self.rmax, dtype=tf.float32, trainable=True)
        self.lambda_ = tf.Variable(lambda_ * self.rmax, trainable=True)
        self.constraint_threshold = constraint_threshold * self.rmax
        self.lambda_max = lambda_max * self.rmax * 1e4
        self.eta_lambda = eta_lambda * self.rmax
        self.delta_lambda_min = delta_lambda_min * self.rmax
        self.delta_lambda_max = delta_lambda_max * self.rmax
        self.target_entropy = target_entropy
        self.alpha_optimizer = Adam(learning_rate=lr_entropy)

        self.epsilon = np.float32(1.0)

    def get_action_space_dimensions(self, action_space):
        if isinstance(action_space, gym.spaces.Tuple):
            dim_d, dim_c = 0, 0
            bounds, shifts = [], []
            for space in action_space.spaces:
                if isinstance(space, gym.spaces.Discrete):
                    dim_d += 1
                elif isinstance(space, gym.spaces.Box):
                    dim_c += space.shape[0]
                    bounds.append(((space.high - space.low) / 2).item())
                    shifts.append(((space.high + space.low) / 2).item())
            return dim_d, dim_c, np.array(bounds, dtype=np.float32), np.array(shifts, dtype=np.float32)
        else:
            raise ValueError("Unsupported action space type.")

    def get_state_space_limits(self):
        if isinstance(self.env.observation_space, gym.spaces.Box):
            return self.env.observation_space.low, self.env.observation_space.high
        else:
            raise ValueError("Unsupported observation space type.")

    def act(self, state):
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
        mu, sigma, sampled_cont, disc_prob, _ = self.actor.predict(state, verbose=0)
        cont = sampled_cont * self.action_bound + self.action_shift
        cont = np.clip(cont, self.action_shift - self.action_bound, self.action_shift + self.action_bound)
        if self.action_dim_discrete > 0:
            if np.random.rand() < self.epsilon:
                disc = np.random.randint(0, 2, size=self.action_dim_discrete)
            else:
                disc = np.random.binomial(1, disc_prob[0])
            disc = np.squeeze(disc)
        else:
            disc = np.array([], dtype=np.int32)
        return cont[0], disc, sampled_cont[0], disc_prob

    def remember(self, state, cont_prob, disc_prob, reward, next_state, done, constraint):
        s = state.astype(np.float32)
        cont = cont_prob.astype(np.float32).squeeze()
        disc = np.squeeze(disc_prob).astype(np.float32)
        r = np.float32(reward)
        ns = next_state.astype(np.float32)
        c = np.float32(constraint)
        if self.use_priority:
            trans = np.hstack([s, cont, disc, r, ns, done, c])
            self.memory.store(trans)
        else:
            self.memory.append([s, cont, disc, r, ns, done, c])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        if self.use_priority:
            idx, samples, W = self.memory.sample(self.batch_size)
            arr = samples.astype(np.float32)
            splits = [0] + np.cumsum([self.state_shape, self.action_dim_continuous, self.action_dim_discrete,
                                       1, self.state_shape, 1, 1]).tolist()
            sts, conts, discs, rews, nsts, dns, cons = [
                arr[:, splits[i]:splits[i+1]] for i in range(len(splits)-1)
            ]
        else:
            W = np.float32(1.0)
            data = random.sample(self.memory, self.batch_size)
            cols = np.array(data).T
            sts, conts, discs, rews, nsts, dns, cons = [
                np.vstack(cols[i]).astype(np.float32) for i in range(7)
            ]
        return sts, conts, discs, rews, nsts, dns.squeeze(-1), W, cons.squeeze(-1)

    def train(self, max_episodes=50, max_epochs=80000, max_steps=24, save_freq=500):
        ep = 0
        rewards_hist = []
        while ep < max_episodes:
            done = False
            state = self.env.reset()
            total_r = 0.0
            total_c = 0.0
            while not done:
                c_a, d_a, spc, spd = self.act(state)
                action = [c_a, d_a]
                ns, r, done, info = self.env.step(action)
                fcost = info['total_cost']
                emis = info['total_emis']
                cons = info['total_penalty']
                self.remember(state, spc, spd, r, ns, done, cons)
                batch = self.replay()
                if batch:
                    # 保持原训练更新逻辑不变
                    pass
                state = ns
                total_r += r
                total_c += cons
            ep += 1
            print(f"Episode {ep}: Reward {total_r:.2f}, Constraint {total_c:.2f}")
            rewards_hist.append(total_r)
        print("Training Complete")

if __name__ == '__main__':
    # env = CombinedEnergyEnv()
    # agent = C_SAC_h(env, lr_actor=5e-5, lr_critic=1e-4, gamma=0.85)
    # agent.train(max_episodes=10)
    env = CombinedEnergyEnv()

    # 状态空间
    print("=== Observation Space ===")
    print("Low  :", env.observation_space.low)
    print("High :", env.observation_space.high)
    print("Shape:", env.observation_space.shape)
    print()

    # 动作空间
    print("=== Action Space ===")
    print("Low  :", env.action_space.low)
    print("High :", env.action_space.high)
    print("Dim  :", env.action_space.shape[0])
    print()

    # 如果你要按照 C_SAC_h 中的方式计算 bounds/shift，可以这样：
    low  = env.action_space.low
    high = env.action_space.high
    bounds = (high - low) / 2
    shifts = (high + low) / 2
    print("Computed action bounds (half-range):", bounds)
    print("Computed action shifts (midpoint):",   shifts)
