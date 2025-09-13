import gym
import random
import datetime
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import os

import matplotlib.pyplot as plt
from matplotlib import rcParams

# 引入新的 IES 环境
from env import CombinedEnergyEnv

from network2.network_normalized_q_p_gpu import NetworkGPU
from network2.Prioritized_Replay_gpu import Memory

import warnings
warnings.filterwarnings("ignore")

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

class C_SAC_GPU:
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
            target_entropy=-26,
            rmax=3020 * 1e-4,
            device=None
            
    ):
        # 设备设置
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Using device: {self.device}")
        
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

        self.network = NetworkGPU(
            state_dim=self.state_shape,
            action_dim_continuous=self.action_dim_continuous,
            action_dim_discrete=0,
            action_bound=self.action_bound,
            action_shift=self.action_shift,
            state_low=self.state_low,
            state_high=self.state_high
        )

        # actor 网络
        self.actor = self.network.actor(actor_units).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # critic 网络 (双Q网络)
        self.critic = self.network.critic(critic_units).to(self.device)
        self.critic_target = self.network.critic(critic_units).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # constraint critic 网络
        self.qc_critic = self.network.constraint_critic(critic_units).to(self.device)
        self.qc_critic_target = self.network.constraint_critic(critic_units).to(self.device)
        self.qc_optimizer = optim.Adam(self.qc_critic.parameters(), lr=lr_critic)

        # 初始化目标网络权重
        self.network.update_target_weights(self.critic,     self.critic_target,     tau=1.0)  
        self.network.update_target_weights(self.qc_critic,  self.qc_critic_target,  tau=1.0)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.summaries = {}

        # PyTorch parameters
        self.alpha = nn.Parameter(torch.tensor(alpha * self.rmax, dtype=torch.float32, device=self.device))
        self.lambda_ = nn.Parameter(torch.tensor(lambda_ * self.rmax, dtype=torch.float32, device=self.device))
        self.constraint_threshold = constraint_threshold * self.rmax
        self.lambda_max = lambda_max * self.rmax * 1e4
        self.eta_lambda = eta_lambda * self.rmax
        self.delta_lambda_min = delta_lambda_min * self.rmax
        self.delta_lambda_max = delta_lambda_max * self.rmax
        self.target_entropy = target_entropy
        self.alpha_optimizer = optim.Adam([self.alpha], lr=lr_entropy)

        self.epsilon = 1.0
        
    
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
        """动作选择方法"""
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
        
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            self.actor.eval()
            mu, sigma, sampled_cont = self.actor(state_tensor)
            self.actor.train()
        
        # 转换回numpy并应用动作缩放
        sampled_cont_np = sampled_cont.cpu().numpy()
        cont = sampled_cont_np * self.action_bound + self.action_shift
        cont = np.clip(cont, self.action_shift - self.action_bound, self.action_shift + self.action_bound)

        return cont[0], sampled_cont_np[0]
    
    def compute_target_q(self, rewards, next_states, done_flags, constraint):
        """计算目标Q值"""
        with torch.no_grad():
            # 1) 通过 actor 得到下一个动作
            mu, sigma, sc = self.actor(next_states)

            # 2) 通过 target critic 得到两个 Q
            nq1, nq2 = self.critic_target(next_states, sc)
            min_nq = torch.min(nq1, nq2)

            # 3) 连续动作 log-prob 和 entropy
            logp_c = -0.5 * ((sc - mu) / (sigma + 1e-8))**2 \
                    - torch.log(sigma + 1e-8) - 0.5 * np.log(2 * np.pi)
            logp_c = torch.sum(logp_c, dim=-1, keepdim=True)
            entropy = -self.alpha * logp_c

            target_q = rewards - self.lambda_ * constraint + self.gamma * (1 - done_flags) * min_nq + entropy

        return target_q
    
    def compute_target_qc(self, constraint, next_states, done_flags):
        """计算约束Critic的目标值"""
        with torch.no_grad():
            # 从 Actor 获取下一个动作
            if self.action_dim_discrete > 0:
                mu, sigma, sampled_cont_action, sampled_disc_action, logits = self.actor(next_states)
            else:
                mu, sigma, sampled_cont_action = self.actor(next_states)

            # 计算 Critic 预测的目标 Q 值
            if self.action_dim_discrete > 0:
                next_qc = self.qc_critic_target(next_states, sampled_cont_action, sampled_disc_action)
            else:
                next_qc = self.qc_critic_target(next_states, sampled_cont_action)

            # 计算连续动作的 log 概率
            log_probs_cont = -0.5 * ((sampled_cont_action - mu) / (sigma + 1e-8)) ** 2 \
                            - torch.log(sigma + 1e-8) - 0.5 * np.log(2 * np.pi)
            log_probs_cont = torch.sum(log_probs_cont, dim=-1, keepdim=True)

            # 计算离散动作的 log 概率
            if self.action_dim_discrete > 0:
                log_probs_disc = torch.log_softmax(logits, dim=-1)
                sampled_idx = torch.argmax(sampled_disc_action, dim=-1)
                log_probs_disc = torch.sum(
                    log_probs_disc * torch.nn.functional.one_hot(sampled_idx, self.action_dim_discrete).float(),
                    dim=-1, keepdim=True
                )
                log_probs = log_probs_cont + log_probs_disc
            else:
                log_probs = log_probs_cont

            # 计算熵项（但不纳入 Qc 目标）
            entropy_term = -self.alpha * log_probs

            # 计算目标 Qc 值（无熵项）
            target_qc = constraint + self.gamma * (1 - done_flags) * next_qc
            
        return target_qc

    def compute_critic_loss(self, states, actions, rewards, next_states, done_flags, constraint):
        """计算Critic损失"""
        if self.action_dim_discrete > 0:
            cont_act, disc_act = actions
            q1, q2 = self.critic(states, cont_act, disc_act)
        else:
            q1, q2 = self.critic(states, actions)

        target_q = self.compute_target_q(rewards, next_states, done_flags, constraint)
        q1_loss = torch.mean((q1 - target_q) ** 2)
        q2_loss = torch.mean((q2 - target_q) ** 2)

        return q1_loss, q2_loss, q1, q2

    def compute_constraint_critic_loss(self, states, actions, constraints, next_states, done_flags):
        """计算约束Critic损失"""
        # Q(s,a) = qc
        if self.action_dim_discrete > 0:
            continuous_actions, discrete_actions = actions
            qc = self.qc_critic(states, continuous_actions, discrete_actions)
        else:
            qc = self.qc_critic(states, actions)

        target_qc = self.compute_target_qc(constraints, next_states, done_flags)

        # 计算约束 Critic 损失
        qc_loss = torch.mean((qc - target_qc) ** 2)

        # 检查是否有 NaN
        if torch.isnan(qc_loss).any():
            raise ValueError("qc_loss contains NaN")

        return qc_loss

    def compute_actor_loss(self, states):
        """计算Actor损失（只更新Actor与alpha，不更新Critic/Qc；alpha对logp断图）"""
        # 1) Actor前向（需要梯度）
        if self.action_dim_discrete > 0:
            mu, sigma, sc, sd, logits = self.actor(states)
        else:
            mu, sigma, sc = self.actor(states)

        # 2) 临时冻结 Critic / Qc，并设为 eval()（降低策略梯度方差）
        was_train_c  = self.critic.training
        was_train_qc = self.qc_critic.training
        for p in self.critic.parameters():     p.requires_grad_(False)
        for p in self.qc_critic.parameters():  p.requires_grad_(False)
        self.critic.eval(); self.qc_critic.eval()

        try:
            # 3) 用冻结的 Critic / Qc 打分（梯度只会通过sc回Actor，不会进到Critic参数）
            if self.action_dim_discrete > 0:
                q1, q2 = self.critic(states, sc, sd)
                qc     = self.qc_critic(states, sc, sd)
            else:
                q1, q2 = self.critic(states, sc)
                qc     = self.qc_critic(states, sc)

            min_q = torch.min(q1, q2)

            # 4) log π(a|s)（未做tanh雅可比修正的简化版；如需更准可后续补）
            logp_c = -0.5 * ((sc - mu) / (sigma + 1e-8))**2 - torch.log(sigma + 1e-8) - 0.5 * np.log(2*np.pi)
            logp   = torch.sum(logp_c, dim=-1, keepdim=True)

            if self.action_dim_discrete > 0:
                logp_d = torch.log_softmax(logits, dim=-1)
                idx    = torch.argmax(sd, dim=-1)
                logp  += torch.sum(logp_d * torch.nn.functional.one_hot(idx, self.action_dim_discrete).float(),
                                dim=-1, keepdim=True)

            entropy    = -self.alpha.detach() * logp
            actor_loss = -(min_q + entropy - (self.lambda_.detach()) * qc).mean()   

            # 5) α只更新α本身；对logp断图，避免对Actor二次反传
            alpha_loss = -(self.alpha * (logp.detach() + self.target_entropy)).mean()

        finally:
            # 6) 恢复 Critic / Qc 的梯度与模式
            for p in self.critic.parameters():     p.requires_grad_(True)
            for p in self.qc_critic.parameters():  p.requires_grad_(True)
            if was_train_c:  self.critic.train()
            if was_train_qc: self.qc_critic.train()

        return actor_loss, alpha_loss


    def update_lambda(self, lambda_value, qc_expectation):
        """更新拉格朗日乘子 λ"""
        lambda_update = self.eta_lambda * (qc_expectation - self.constraint_threshold)

        # 投影 λ 更新值
        lambda_update = torch.clamp(lambda_update, min=self.delta_lambda_min, max=self.delta_lambda_max)
        
        # 计算新的 λ，并投影到 [0, λ_max] 范围内
        new_lambda = torch.clamp(lambda_value + lambda_update, min=0.0, max=self.lambda_max)

        return new_lambda

    def soft_update_all_targets(self):
        """软更新目标网络"""
        self.network.update_target_weights(self.critic, self.critic_target, self.tau)
        self.network.update_target_weights(self.qc_critic, self.qc_critic_target, self.tau)

    def remember(self, state, cont_prob, reward, next_state, done, constraint):
        """存储经验"""
        s = state.astype(np.float32)
        cont = cont_prob.astype(np.float32).squeeze()
        r = np.float32(reward)
        ns = next_state.astype(np.float32)
        c = np.float32(constraint)

        if self.use_priority:
            trans = np.hstack([s, cont, r, ns, done, c])
            self.memory.store(trans)
        else:
            s = np.expand_dims(s, axis=0)
            ns = np.expand_dims(ns, axis=0)
            self.memory.append([s, cont, r, ns, done, c])

    def replay(self):
        """经验回放采样"""
        if len(self.memory) < self.batch_size:
            return None
            
        if self.use_priority:
            idx, samples, W = self.memory.sample(self.batch_size, device=self.device)
            arr = samples.astype(np.float32)
            splits = [0] + np.cumsum([self.state_shape, self.action_dim_continuous, 
                                     1, self.state_shape, 1, 1]).tolist()
            sts, conts, rews, nsts, dns, cons = [
                arr[:, splits[i]:splits[i+1]] for i in range(len(splits)-1)
            ]
        else:
            W = torch.tensor(1.0, dtype=torch.float32, device=self.device)
            data = random.sample(self.memory, self.batch_size)
            cols = np.array(data).T
            sts, conts, rews, nsts, dns, cons = [
                np.vstack(cols[i]).astype(np.float32) for i in range(6)
            ]

        return sts, conts, rews, nsts, dns.squeeze(-1), W, cons.squeeze(-1)

    def _update_from_batch(self, batch):
        """用一批经验更新所有网络"""
        states, conts, rews, next_states, dones, ISW, cons = batch

        # 转换为PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        conts = torch.tensor(conts, dtype=torch.float32, device=self.device)
        rews = torch.tensor(rews, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        cons = torch.tensor(cons, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # 1) Q1/Q2 Critic 更新
        self.critic_optimizer.zero_grad()
        q1_loss, q2_loss, _, _ = self.compute_critic_loss(
            states, conts, rews, next_states, dones, cons
        )
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2) Constraint Critic 更新
        self.qc_optimizer.zero_grad()
        qc_loss = self.compute_constraint_critic_loss(
            states, conts, cons, next_states, dones
        )
        qc_loss.backward()
        self.qc_optimizer.step()

        # 3) Actor & alpha 更新
        self.actor_optimizer.zero_grad()
        self.alpha_optimizer.zero_grad()
        actor_loss, alpha_loss = self.compute_actor_loss(states)
        actor_loss.backward()
        alpha_loss.backward()
        self.actor_optimizer.step()
        self.alpha_optimizer.step()
        
        # 保证 alpha 正数
        with torch.no_grad():
            self.alpha.data = torch.clamp(self.alpha.data, min=1e-6)

        # 4) λ 更新
        with torch.no_grad():
            qc_expect = torch.mean(self.qc_critic(states, conts))
            self.lambda_.data = self.update_lambda(self.lambda_.data, qc_expect)

        # 5) 软更新目标网络
        # self.soft_update_all_targets()
        
        return q1_loss.item(), q2_loss.item(), qc_loss.item(), actor_loss.item()

    def save_models(self, filepath):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'qc_critic_state_dict': self.qc_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'qc_optimizer_state_dict': self.qc_optimizer.state_dict(),
            'alpha': self.alpha.data,
            'lambda_': self.lambda_.data,
        }, filepath)

    def load_models(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.qc_critic.load_state_dict(checkpoint['qc_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.qc_optimizer.load_state_dict(checkpoint['qc_optimizer_state_dict'])
        self.alpha.data = checkpoint['alpha']
        self.lambda_.data = checkpoint['lambda_']


# 保持原来的类名以便兼容
C_SAC_ = C_SAC_GPU

if __name__ == '__main__':
    env = CombinedEnergyEnv('IES1')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = C_SAC_GPU(env, lr_actor=5e-5, lr_critic=1e-4, gamma=0.85, device=device)

    # 测试动作空间
    dim_d, dim_c, bounds, shifts = agent.get_action_space_dimensions(env.action_space)
    print("离散动作维度 dim_d    :", dim_d)
    print("连续动作维度 dim_c    :", dim_c)
    print("动作半范围 bounds     :", bounds)
    print("动作中点 shifts      :", shifts)

    # 测试状态空间
    state_low, state_high = agent.get_state_space_limits()
    print("状态下限 state_low   :", state_low)
    print("状态上限 state_high  :", state_high)
    print("状态维度 state_shape :", agent.state_shape)

    print("PyTorch C-SAC agent initialized successfully!")