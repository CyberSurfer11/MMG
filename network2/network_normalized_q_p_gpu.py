import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 输入的状态是原始的，进行归一化状态。得到的动作是归一化的
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim_continuous, action_dim_discrete,
                 action_bound, action_shift, state_low, state_high, units=(512, 256, 64, 32), tau=0.5):
        super(ActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim_continuous = action_dim_continuous
        self.action_dim_discrete = 0  # 如需启用离散动作，请把这行改回：action_dim_discrete
        self.register_buffer('action_bound', torch.tensor(action_bound, dtype=torch.float32))
        self.register_buffer('action_shift', torch.tensor(action_shift, dtype=torch.float32))

        self.tau = tau
        
        # 状态的上下限
        self.register_buffer('state_low', torch.tensor(state_low, dtype=torch.float32))
        self.register_buffer('state_high', torch.tensor(state_high, dtype=torch.float32))
        self.register_buffer('state_range', self.state_high - self.state_low)
        
        # 划分浅层/深层的分界索引
        total_layers = len(units)
        self.shallow_count = max(1, total_layers // 2)
        
        # 构建浅层网络 (Actor_Shared_*)
        self.shared_layers = nn.ModuleList()
        input_dim = state_dim
        
        for i in range(self.shallow_count):
            self.shared_layers.append(nn.Linear(input_dim, units[i]))
            nn.init.kaiming_uniform_(self.shared_layers[-1].weight)
            input_dim = units[i]
        
        # 构建深层网络 (Actor_Deep_*)
        self.deep_layers = nn.ModuleList()
        for i in range(self.shallow_count, total_layers):
            self.deep_layers.append(nn.Linear(input_dim, units[i]))
            nn.init.kaiming_uniform_(self.deep_layers[-1].weight)
            input_dim = units[i]
        
        # 连续动作输出层
        self.mu_head = nn.Linear(input_dim, action_dim_continuous)
        self.sigma_head = nn.Linear(input_dim, action_dim_continuous)
        nn.init.kaiming_uniform_(self.mu_head.weight)
        nn.init.kaiming_uniform_(self.sigma_head.weight)
        
        # 离散动作输出层（如果需要）
        if self.action_dim_discrete > 0:
            self.logits_head = nn.Linear(input_dim, action_dim_discrete)
            nn.init.kaiming_uniform_(self.logits_head.weight)
    
    def forward(self, state):
        # 状态归一化
        normalized_state = (state - self.state_low) / self.state_range
        
        # 浅层前向传播
        x = normalized_state
        for layer in self.shared_layers:
            x = F.leaky_relu(layer(x))
        
        # 深层前向传播
        for layer in self.deep_layers:
            x = F.leaky_relu(layer(x))
        
        # 连续动作输出
        mu_output = torch.tanh(self.mu_head(x))
        sigma_output = F.softplus(self.sigma_head(x))
        sigma_output = torch.clamp(sigma_output, min=0.1)
        
        # 重参数化采样
        epsilon = torch.randn_like(sigma_output)
        sampled_action = mu_output + sigma_output * epsilon
        
        # 将连续动作归一化至 [-1, 1] 后映射回原空间
        sampled_cont_action = torch.tanh(sampled_action)
        
        # 离散动作（如果需要）
        if self.action_dim_discrete > 0:
            logits = self.logits_head(x)
            disc_action_prob = F.softmax(logits, dim=-1)
            disc_action = torch.argmax(disc_action_prob, dim=-1)
            return mu_output, sigma_output, sampled_cont_action, disc_action, logits
        
        return mu_output, sigma_output, sampled_cont_action


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim_continuous, action_dim_discrete,
                 state_low, state_high, units=(128, 128, 32)):
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim_continuous = action_dim_continuous
        self.action_dim_discrete = 0  # 与Actor保持一致
        
        # 状态的上下限
        self.register_buffer('state_low', torch.tensor(state_low, dtype=torch.float32))
        self.register_buffer('state_high', torch.tensor(state_high, dtype=torch.float32))
        self.register_buffer('state_range', self.state_high - self.state_low)
        
        # 输入维度：状态 + 连续动作 + (离散动作)
        input_dim = state_dim + action_dim_continuous
        if self.action_dim_discrete > 0:
            input_dim += action_dim_discrete
        
        # 划分浅层/深层
        total_layers = len(units)
        self.shallow_count = max(1, total_layers // 2)
        
        # 浅层网络 (Critic_Shared_*)
        self.shared_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for i in range(self.shallow_count):
            self.shared_layers.append(nn.Linear(prev_dim, units[i]))
            nn.init.kaiming_uniform_(self.shared_layers[-1].weight)
            prev_dim = units[i]
        
        # 深层网络 (Critic_Deep_*)
        self.deep_layers = nn.ModuleList()
        for i in range(self.shallow_count, total_layers):
            self.deep_layers.append(nn.Linear(prev_dim, units[i]))
            nn.init.kaiming_uniform_(self.deep_layers[-1].weight)
            prev_dim = units[i]
        
        # 只在第一层后使用 LayerNorm（维度=第一层输出）
        self.ln_first = nn.LayerNorm(units[0])
        
        # Dropout 层
        self.dropout = nn.Dropout(0.1)
        
        # 双Q输出头
        self.q1_head = nn.Linear(prev_dim, 1)
        self.q2_head = nn.Linear(prev_dim, 1)
    
    def forward(self, state, cont_action, disc_action=None):
        # 状态归一化
        normalized_state = (state - self.state_low) / self.state_range
        
        # 拼接状态和动作
        if self.action_dim_discrete > 0 and disc_action is not None:
            x = torch.cat([normalized_state, cont_action, disc_action], dim=-1)
        else:
            x = torch.cat([normalized_state, cont_action], dim=-1)
        
        # 浅层前向传播
        layer_idx = 0
        for i, layer in enumerate(self.shared_layers):
            x = layer(x)
            x = F.leaky_relu(x)
            if i == 0:  # 第一层使用 LayerNorm
                x = self.ln_first(x)
            layer_idx += 1
        
        # 深层前向传播
        for i, layer in enumerate(self.deep_layers):
            x = layer(x)
            x = F.leaky_relu(x)
            x = self.dropout(x)
            layer_idx += 1
        
        # 双Q输出
        q1 = self.q1_head(x)
        q2 = self.q2_head(x)
        
        return q1, q2


class ConstraintCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim_continuous, action_dim_discrete,
                 state_low, state_high, units=(128, 128, 32)):
        super(ConstraintCriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim_continuous = action_dim_continuous
        self.action_dim_discrete = 0
        
        # 状态的上下限
        self.register_buffer('state_low', torch.tensor(state_low, dtype=torch.float32))
        self.register_buffer('state_high', torch.tensor(state_high, dtype=torch.float32))
        self.register_buffer('state_range', self.state_high - self.state_low)
        
        # 输入维度
        input_dim = state_dim + action_dim_continuous
        if self.action_dim_discrete > 0:
            input_dim += action_dim_discrete
        
        # 构建网络层
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for i, unit in enumerate(units):
            self.layers.append(nn.Linear(prev_dim, unit))
            nn.init.kaiming_uniform_(self.layers[-1].weight)
            prev_dim = unit
        
        # 只在第一层后使用 LayerNorm（维度=第一层输出）
        self.ln_first = nn.LayerNorm(units[0])
        self.dropout = nn.Dropout(0.1)
        
        # 输出层
        self.qc_head = nn.Linear(prev_dim, 1)
    
    def forward(self, state, cont_action, disc_action=None):
        # 状态归一化
        normalized_state = (state - self.state_low) / self.state_range
        
        # 拼接状态和动作
        if self.action_dim_discrete > 0 and disc_action is not None:
            x = torch.cat([normalized_state, cont_action, disc_action], dim=-1)
        else:
            x = torch.cat([normalized_state, cont_action], dim=-1)
        
        # 前向传播
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = F.leaky_relu(x)
            if i == 0:  # 第一层使用 LayerNorm
                x = self.ln_first(x)
            if i > 0:  # 其他层使用Dropout
                x = self.dropout(x)
        
        # 输出约束Q值
        qc = self.qc_head(x)
        return qc


@torch.no_grad()
def hard_update(target: nn.Module, source: nn.Module, copy_buffers: bool = True) -> None:
    """
    把 source 的参数硬拷贝到 target（用于初始化 target 或需要立即对齐时）。
    copy_buffers=True 时，也会同步所有 buffer（如归一化常量、BN running stats）。
    """
    for t, s in zip(target.parameters(), source.parameters()):
        t.copy_(s)
    if copy_buffers:
        for tb, sb in zip(target.buffers(), source.buffers()):
            tb.copy_(sb)

@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float, copy_buffers: bool = True) -> None:
    """
    Polyak 软更新：target ← (1-τ)*target + τ*source
    """
    one_minus_tau = 1.0 - tau
    for t, s in zip(target.parameters(), source.parameters()):
        t.mul_(one_minus_tau).add_(s, alpha=tau)  # 原地操作
    if copy_buffers:
        for tb, sb in zip(target.buffers(), source.buffers()):
            tb.copy_(sb)

class NetworkGPU:
    """PyTorch 版网络工厂，保持与原 TF 版接口一致"""
    def __init__(self, state_dim, action_dim_continuous, action_dim_discrete,
                 action_bound, action_shift, state_low, state_high):
        self.state_dim = state_dim
        self.action_dim_continuous = action_dim_continuous
        self.action_dim_discrete = 0  # 保持与原版一致
        self.action_bound = action_bound
        self.action_shift = action_shift
        self.state_low = np.array(state_low, dtype=np.float32)
        self.state_high = np.array(state_high, dtype=np.float32)
        self.state_range = self.state_high - self.state_low
    
    def actor(self, units=(512, 256, 64, 32), tau_=0.5):
        """创建Actor网络"""
        return ActorNetwork(
            self.state_dim, self.action_dim_continuous, self.action_dim_discrete,
            self.action_bound, self.action_shift, self.state_low, self.state_high,
            units, tau_
        )
    
    def critic(self, units=(128, 128, 32)):
        """创建Critic网络"""
        return CriticNetwork(
            self.state_dim, self.action_dim_continuous, self.action_dim_discrete,
            self.state_low, self.state_high, units
        )
    
    def constraint_critic(self, units=(128, 128, 32)):
        """创建约束Critic网络"""
        return ConstraintCriticNetwork(
            self.state_dim, self.action_dim_continuous, self.action_dim_discrete,
            self.state_low, self.state_high, units
        )

    def update_target_weights(self, online: nn.Module, target: nn.Module, tau: float, hard: bool = False) -> None:
        """
        - online: 在线网络（source）
        - target: 目标网络（target）
        - tau   : 软更新系数；当 hard=True 或 tau>=1.0 时执行硬拷贝
        """
        if hard or tau >= 1.0:
            hard_update(target, online, copy_buffers=True)
        else:
            soft_update(target, online, tau, copy_buffers=True)


if __name__ == "__main__":
    # 用随便的参数创建网络
    net = NetworkGPU(state_dim=10,
                    action_dim_continuous=2,
                    action_dim_discrete=0,
                    action_bound=[1.0,1.0],
                    action_shift=[0.0,0.0],
                    state_low=[0.0]*10,
                    state_high=[1.0]*10)

    actor = net.actor()
    critic = net.critic()
    qc     = net.constraint_critic()

    print("=== Actor 参数前缀检查 ===")
    for name, _ in actor.named_parameters():
        tag = "SHARED" if name.startswith("shared_layers") else "PERSONAL"
        print(name, "->", tag)

    print("\n=== Critic 参数前缀检查 ===")
    for name, _ in critic.named_parameters():
        tag = "SHARED" if name.startswith("shared_layers") else "PERSONAL"
        print(name, "->", tag)

    print("\n=== Constraint Critic 参数前缀检查 ===")
    for name, _ in qc.named_parameters():
        # 约束Critic没有拆分浅/深，全算PERSONAL
        print(name, "-> PERSONAL")