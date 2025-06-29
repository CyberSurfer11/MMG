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
from env import CombinedEnergyEnv

from network import Memory,network

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示 ERROR

tf.keras.backend.set_floatx('float32')

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

class C_SAC_:
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
            rmax=2960 * 1e-4
    ):
        self.rmax = rmax
        self.env = env
        self.state_shape = env.state.shape[0]
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
            action_dim_discrete=0,
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
        # 先处理多分支（离散+连续）Tuple
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

        # 再处理纯连续 Box
        elif isinstance(action_space, gym.spaces.Box):
            dim_d = 0
            dim_c = action_space.shape[0]
            # 半范围：(high - low) / 2
            bound = ((action_space.high - action_space.low) / 2).astype(np.float32)
            # 偏移：(high + low) / 2
            shift = ((action_space.high + action_space.low) / 2).astype(np.float32)
            return dim_d, dim_c, bound, shift

        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")


    def get_state_space_limits(self):
        if isinstance(self.env.observation_space, gym.spaces.Box):
            return self.env.observation_space.low, self.env.observation_space.high
        else:
            raise ValueError("Unsupported observation space type.")

    def act(self, state):
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
        mu, sigma, sampled_cont = self.actor.predict(state, verbose=0)
        # print('sampled_cont',sampled_cont)
        cont = sampled_cont * self.action_bound + self.action_shift
        cont = np.clip(cont, self.action_shift - self.action_bound, self.action_shift + self.action_bound)

        return cont[0],sampled_cont[0]
    
    def compute_target_q(self, rewards, next_states, done_flags, constraint):
        # next action
        if self.action_dim_discrete>0:
            mu, sigma, sc, sd, logits = self.actor(next_states)
        else:
            mu, sigma, sc = self.actor(next_states)
        # next Q
        if self.action_dim_discrete>0:
            nq1 = self.q1_critic_target([next_states, sc, sd])
            nq2 = self.q2_critic_target([next_states, sc, sd])
        else:
            nq1 = self.q1_critic_target([next_states, sc])
            nq2 = self.q2_critic_target([next_states, sc])

        min_nq = tf.minimum(nq1, nq2)

        # log prob cont
        logp_c = -0.5*((sc-mu)/(sigma+1e-8))**2 - tf.math.log(sigma+1e-8) - 0.5*np.log(2*np.pi)
        logp_c = tf.reduce_sum(logp_c, axis=-1, keepdims=True)

        # log prob disc
        if self.action_dim_discrete>0:
            logp_d = tf.nn.log_softmax(logits)
            idx = tf.argmax(sd, axis=-1, output_type=tf.int32)
            logp_d = tf.reduce_sum(logp_d*tf.one_hot(idx, self.action_dim_discrete), axis=-1, keepdims=True)
            logp = logp_c + logp_d
        else:
            logp = logp_c

        entropy = -self.alpha * logp
        target_q = rewards - self.lambda_*constraint + self.gamma*(1-done_flags)*min_nq + entropy
        return target_q
    
        # 修改目标qc
    def compute_target_qc(self, constraint, next_states, done_flags):
        """ 计算 C-SAC 的目标 Q 值（加入熵项） """

        #  **从 Actor 获取下一个动作**
        if self.action_dim_discrete > 0:
            mu, sigma, sampled_cont_action, sampled_disc_action, logits = self.actor(next_states)
        else:
            mu, sigma, sampled_cont_action = self.actor(next_states)

        #  **计算 Critic 预测的目标 Q 值**
        if self.action_dim_discrete > 0:
            next_qc = self.qc_critic_target([next_states, sampled_cont_action, sampled_disc_action])
        else:
            next_qc = self.qc_critic_target([next_states, sampled_cont_action])

        #  **计算连续动作的 log 概率**
        log_probs_cont = -0.5 * ((sampled_cont_action - mu) / (sigma + 1e-8)) ** 2 \
                         - tf.math.log(sigma + 1e-8) - 0.5 * np.log(2 * np.pi)
        log_probs_cont = tf.reduce_sum(log_probs_cont, axis=-1, keepdims=True)

        #  **计算离散动作的 log 概率**
        if self.action_dim_discrete > 0:
            log_probs_disc = tf.nn.log_softmax(logits)
            sampled_idx = tf.argmax(sampled_disc_action, axis=-1, output_type=tf.int32)
            log_probs_disc = tf.reduce_sum(
                log_probs_disc * tf.one_hot(sampled_idx, self.action_dim_discrete),
                axis=-1, keepdims=True
            )
            log_probs = log_probs_cont + log_probs_disc
        else:
            log_probs = log_probs_cont

        #  **计算熵项（但不纳入 Qc 目标）**
        entropy_term = -self.alpha * log_probs

        #  **计算目标 Qc 值（无熵项）**
        target_qc = constraint + self.gamma * (1 - done_flags) * next_qc
        return target_qc

    def compute_critic_loss(self, states, actions, rewards, next_states, done_flags, constraint):
        if self.action_dim_discrete>0:
            cont_act, disc_act = actions
            q1 = self.q1_critic([states, cont_act, disc_act])
            q2 = self.q2_critic([states, cont_act, disc_act])
        else:
            q1 = self.q1_critic([states, actions])
            q2 = self.q2_critic([states, actions])

        target_q = self.compute_target_q(rewards, next_states, done_flags, constraint)
        q1_loss = tf.reduce_mean((q1 - target_q) ** 2)
        q2_loss = tf.reduce_mean((q2 - target_q) ** 2)

        return q1_loss, q2_loss,q1,q2

    # actor动作
    def compute_constraint_critic_loss(self, states, actions, constraints, next_states, done_flags):
        """ 计算 C-SAC 的 约束 Critic 损失（支持离散+连续动作） """
        # Q(s,a) = qc
        if self.action_dim_discrete > 0:
            continuous_actions, discrete_actions = actions  # 拆分动作
            qc = self.qc_critic([states, continuous_actions, discrete_actions])
        else:
            qc = self.qc_critic([states, actions])


        target_qc = self.compute_target_qc(constraints, next_states, done_flags)

        # target_qc = tf.clip_by_value(target_qc, -1e10, 1e10)

        #  限制 target_qc，防止梯度爆炸
        # qc = tf.clip_by_value(qc, -1e10, 1e10)

        #  **计算约束 Critic 损失**
        qc_loss = tf.reduce_mean((qc - target_qc) ** 2)

        #  **检查是否有 NaN**
        assert not tf.reduce_any(tf.math.is_nan(qc_loss)), " qc_loss 出现 NaN"

        return qc_loss

    def compute_actor_loss(self, states):
        if self.action_dim_discrete>0:
            mu, sigma, sc, sd, logits = self.actor(states)
        else:
            mu, sigma, sc = self.actor(states)
        # Q estimates
        if self.action_dim_discrete>0:
            q1 = self.q1_critic([states, sc, sd])
            q2 = self.q2_critic([states, sc, sd])
            qc = self.qc_critic([states, sc, sd])
        else:
            q1 = self.q1_critic([states, sc])
            q2 = self.q2_critic([states, sc])
            qc = self.qc_critic([states, sc])
        min_q = tf.minimum(q1, q2)
        # log probs
        logp_c = -0.5*((sc-mu)/(sigma+1e-8))**2 - tf.math.log(sigma+1e-8) - 0.5*np.log(2*np.pi)
        logp = tf.reduce_sum(logp_c, axis=-1, keepdims=True)

        if self.action_dim_discrete>0:
            logp_d = tf.nn.log_softmax(logits)
            idx = tf.argmax(sd, axis=-1, output_type=tf.int32)
            logp_d = tf.reduce_sum(logp_d*tf.one_hot(idx,self.action_dim_discrete),axis=-1,keepdims=True)
            logp += logp_d

        entropy = -self.alpha*logp
        actor_loss = -tf.reduce_mean(min_q + entropy - self.lambda_*qc)
        alpha_loss = -tf.reduce_mean(self.alpha*(logp + self.target_entropy))

        return actor_loss, alpha_loss

    # actor动作
    def update_lambda(self, lambda_value, qc_expectation):
        """
        更新拉格朗日乘子 λ
        """
        lambda_update = self.eta_lambda * (qc_expectation - self.constraint_threshold)

        # 投影 λ 更新值
        lambda_update = tf.clip_by_value(lambda_update, clip_value_min=self.delta_lambda_min,
                                         clip_value_max=self.delta_lambda_max)
        
        # 计算新的 λ，并投影到 [0, λ_max] 范围内
        new_lambda = tf.clip_by_value(lambda_value + lambda_update, clip_value_min=0.0, clip_value_max=self.lambda_max)

        return new_lambda
    
    def soft_update_all_targets(self):
        self.network.update_target_weights(self.q1_critic, self.q1_critic_target, self.tau)
        self.network.update_target_weights(self.q2_critic, self.q2_critic_target, self.tau)
        self.network.update_target_weights(self.qc_critic, self.qc_critic_target, self.tau)

    # 存的是actor的动作
    def remember(self, state, cont_prob, reward, next_state, done, constraint):
        s = state.astype(np.float32)
        cont = cont_prob.astype(np.float32).squeeze()
        r = np.float32(reward)
        ns = next_state.astype(np.float32)
        c = np.float32(constraint)

        if self.use_priority:
            trans = np.hstack([s, cont, r, ns, done, c])
            self.memory.store(trans)
        else:
            s = np.expand_dims(s,axis=0)
            ns = np.expand_dims(ns,axis=0)
            self.memory.append([s, cont, r, ns, done, c])

    # 取的是actor的动作
    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        if self.use_priority:
            idx, samples, W = self.memory.sample(self.batch_size)
            arr = samples.astype(np.float32)
            splits = [0] + np.cumsum([self.state_shape, self.action_dim_continuous, 
                                       1, self.state_shape, 1, 1]).tolist()
            sts, conts, rews, nsts, dns, cons = [
                arr[:, splits[i]:splits[i+1]] for i in range(len(splits)-1)
            ]
        else:
            W = np.float32(1.0)
            data = random.sample(self.memory, self.batch_size)
            cols = np.array(data).T
            sts, conts, rews, nsts, dns, cons = [
                np.vstack(cols[i]).astype(np.float32) for i in range(6)
            ]

        return sts, conts, rews, nsts, dns.squeeze(-1), W, cons.squeeze(-1)

    def train(self, max_episodes=50, max_steps=24, save_freq=500):
        """
        完整的训练循环，包括 Critic、约束 Critic、Actor、alpha 和 lambda 更新，以及目标网络软更新。
        不保存模型、不绘图。
        """
        episode = 0

        rewards_hist = []
        constraint_hist = []
        actor_losses = []
        critic_losses = []
        constraint_losses = []

        while episode < max_episodes:
            state = self.env.reset()
            done = False
            total_reward = 0.0
            total_penalty = 0.0

            while not done:
                # 1. 选择动作
                cont_action, sampled_cont_action = self.act(state)
                action = cont_action
                # print(action)

                # 2. 与环境交互
                next_state, reward, done, info = self.env.step(action)
                penalty = info['total_penalty']

                # 3. 存储经验：存actor的动作
                self.remember(state, sampled_cont_action, reward, next_state, done, penalty)

                # 4. 采样并更新网络
                batch = self.replay()
                if batch is not None:
                    states, conts, rewards, next_states, dones, ISW, penalties = batch

                    # 4.1 更新 Critic 网络
                    with tf.GradientTape(persistent=True) as tape_q:
                        q1_loss, q2_loss, q1, q2 = self.compute_critic_loss(
                            states, conts, rewards, next_states, dones, penalties
                        )
                    grads_q1 = tape_q.gradient(q1_loss, self.q1_critic.trainable_variables)
                    self.critic_optimizer.apply_gradients(zip(grads_q1, self.q1_critic.trainable_variables))
                    grads_q2 = tape_q.gradient(q2_loss, self.q2_critic.trainable_variables)
                    self.critic_optimizer.apply_gradients(zip(grads_q2, self.q2_critic.trainable_variables))

                    # 4.2 更新约束 Critic 网络
                    with tf.GradientTape() as tape_c:
                        qc_loss = self.compute_constraint_critic_loss(
                            states, conts, penalties, next_states, dones
                        )
                    grads_qc = tape_c.gradient(qc_loss, self.qc_critic.trainable_variables)
                    self.qc_optimizer.apply_gradients(zip(grads_qc, self.qc_critic.trainable_variables))

                    # 4.3 更新 Actor 和 alpha
                    with tf.GradientTape(persistent=True) as tape_a:
                        actor_loss, alpha_loss = self.compute_actor_loss(states)
                    grads_a = tape_a.gradient(actor_loss, self.actor.trainable_variables)
                    self.actor_optimizer.apply_gradients(zip(grads_a, self.actor.trainable_variables))
                    alpha_grad = tape_a.gradient(alpha_loss, [self.alpha])
                    self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.alpha]))
                    # 限制 alpha 范围
                    self.alpha.assign(tf.clip_by_value(self.alpha, 1e-6, np.inf))

                    # 4.4 更新 lambda
                    qc_expect = tf.reduce_mean(self.qc_critic([states, conts]))
                    self.lambda_.assign(self.update_lambda(self.lambda_, qc_expect))

                    # 4.5 软更新目标网络
                    self.soft_update_all_targets()

                    # 4.6 保存损失
                    actor_losses.append(actor_loss.numpy())
                    critic_losses.append((q1_loss.numpy() + q2_loss.numpy()) / 2)
                    constraint_losses.append(qc_loss.numpy())

                # 5. 进入下一个时间步
                state = next_state
                total_reward += reward
                total_penalty += penalty

            episode += 1
            rewards_hist.append(total_reward)
            constraint_hist.append(total_penalty)

            print(f"Episode {episode}: Reward {total_reward:.2f}, Penalty {total_penalty:.2f}")



if __name__ == '__main__':
    env = CombinedEnergyEnv('IES1')
    print(len(env.elec_load))
    # agent = C_SAC_(env, lr_actor=5e-5, lr_critic=1e-4, gamma=0.85)
    # agent.train(max_episodes=10)

    # env = CombinedEnergyEnv()

    # # 状态空间
    # print("=== Observation Space ===")
    # print("Low  :", env.observation_space.low)
    # print("High :", env.observation_space.high)
    # print("Shape:", env.observation_space.shape)
    # print()

    # # 动作空间
    # print("=== Action Space ===")
    # print("Low  :", env.action_space.low)
    # print("High :", env.action_space.high)
    # print("Dim  :", env.action_space.shape[0])
    # print()

    # # 如果你要按照 C_SAC_h 中的方式计算 bounds/shift，可以这样：
    # low  = env.action_space.low
    # high = env.action_space.high
    # bounds = (high - low) / 2
    # shifts = (high + low) / 2
    # print("Computed action bounds (half-range):", bounds)
    # print("Computed action shifts (midpoint):",   shifts)

    # # 测试动作空间
    # dim_d, dim_c, bounds, shifts = agent.get_action_space_dimensions(env.action_space)
    # print("离散动作维度 dim_d    :", dim_d)
    # print("连续动作维度 dim_c    :", dim_c)
    # print("动作半范围 bounds     :", bounds)
    # print("动作中点 shifts      :", shifts)

    # # 测试状态空间
    # state_low, state_high = agent.get_state_space_limits()
    # print("状态下限 state_low   :", state_low)
    # print("状态上限 state_high  :", state_high)
    # print("状态维度 state_shape :", agent.state_shape)
