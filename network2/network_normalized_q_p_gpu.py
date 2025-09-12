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
        self.action_bound = torch.tensor(action_bound, dtype=torch.float32)
        self.action_shift = torch.tensor(action_shift, dtype=torch.float32)
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
        sampled_cont_action = torch.tanh(mu_output + sigma_output * epsilon)
        
        if self.action_dim_discrete > 0:
            logits = self.logits_head(x)
            sampled_disc_action = self.gumbel_softmax_sample(logits, self.tau)
            return mu_output, sigma_output, sampled_cont_action, sampled_disc_action, logits
        
        return mu_output, sigma_output, sampled_cont_action
    
    def gumbel_softmax_sample(self, logits, tau=0.5):
        """Gumbel-Softmax 采样函数"""
        uniform_noise = torch.rand_like(logits).clamp(min=1e-8, max=1.0)
        gumbel_noise = -torch.log(-torch.log(uniform_noise))
        return F.softmax((logits + gumbel_noise) / tau, dim=-1)


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
        
        # LayerNorm 层
        self.layer_norms = nn.ModuleList()
        for _ in range(total_layers):
            self.layer_norms.append(nn.LayerNorm(prev_dim))
        
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
            if i == 0:  # 第一层使用LayerNorm
                x = self.layer_norms[layer_idx](x)
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
        
        # LayerNorm 和 Dropout
        self.layer_norms = nn.ModuleList()
        for _ in range(len(units)):
            self.layer_norms.append(nn.LayerNorm(prev_dim))
        
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
            if i == 0:  # 第一层使用LayerNorm
                x = self.layer_norms[i](x)
            if i > 0:  # 其他层使用Dropout
                x = self.dropout(x)
        
        # 输出约束Q值
        qc = self.qc_head(x)
        return qc


class NetworkGPU:
    """PyTorch版本的网络工厂类，保持与原始TensorFlow版本的接口兼容"""
    
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
    
    @staticmethod
    def update_target_weights(model, target_model, tau=0.01):
        """软更新目标网络权重"""
        with torch.no_grad():
            for target_param, param in zip(target_model.parameters(), model.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


# 为了向后兼容，保持原来的类名
network = NetworkGPU

if __name__ == "__main__":
    print("PyTorch network imports worked!")
    
    # 简单测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 测试网络创建
    state_dim = 20
    action_dim_continuous = 8
    action_dim_discrete = 0
    action_bound = np.ones(action_dim_continuous)
    action_shift = np.zeros(action_dim_continuous)
    state_low = np.zeros(state_dim)
    state_high = np.ones(state_dim)
    
    net_factory = NetworkGPU(state_dim, action_dim_continuous, action_dim_discrete,
                           action_bound, action_shift, state_low, state_high)
    
    actor = net_factory.actor()
    critic = net_factory.critic()
    constraint_critic = net_factory.constraint_critic()
    
    print(f"Actor parameters: {sum(p.numel() for p in actor.parameters())}")
    print(f"Critic parameters: {sum(p.numel() for p in critic.parameters())}")
    print(f"Constraint Critic parameters: {sum(p.numel() for p in constraint_critic.parameters())}")