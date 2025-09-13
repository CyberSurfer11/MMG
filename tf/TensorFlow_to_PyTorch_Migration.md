# TensorFlow 到 PyTorch 迁移分析

## 改动幅度评估

**总体改动幅度：中等到大幅改动 (70-80%)**

### 改动复杂度分析：
- **网络架构改动**：需要完全重写，但逻辑结构可保持 (复杂度：高)
- **训练流程改动**：梯度计算和优化器需要重写 (复杂度：中等)
- **算法逻辑改动**：Fed-CSAC算法逻辑基本保持不变 (复杂度：低)
- **数据处理改动**：numpy兼容，改动较小 (复杂度：低)

---

## 核心改动文件列表

### 1. 网络定义文件 (HIGH PRIORITY - 完全重写)

#### `network2/network_normalized_shared_q.py`
**当前TensorFlow实现：**
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate, LeakyReLU

class network:
    def actor(self, units=(512, 256, 64, 32)):
        state_input = Input(shape=(self.state_dim,))
        x = Dense(units[0])(normalized_state)
        # ...
        return Model(inputs=state_input, outputs=[mu_output, sigma_output, sampled_cont_action])
```

**需要改为PyTorch：**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, units=(512, 256, 64, 32)):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 构建网络层
        layers = []
        prev_dim = state_dim
        for unit in units:
            layers.extend([
                nn.Linear(prev_dim, unit),
                nn.LeakyReLU()
            ])
            prev_dim = unit
        
        self.shared_layers = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev_dim, action_dim)
        self.sigma_head = nn.Linear(prev_dim, action_dim)
    
    def forward(self, state):
        # 状态归一化
        normalized_state = (state - self.state_low) / self.state_range
        x = self.shared_layers(normalized_state)
        
        mu = torch.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        sigma = torch.clamp(sigma, min=0.1)
        
        # 重参数化技巧
        epsilon = torch.randn_like(sigma)
        sampled_action = torch.tanh(mu + sigma * epsilon)
        
        return mu, sigma, sampled_action
```

#### `network2/network_normalized_q_p.py`
**类似改动，需要改写为PyTorch格式**

#### `network/network_normalized.py`
**类似改动，需要改写为PyTorch格式**

### 2. 主要智能体文件 (HIGH PRIORITY - 大幅修改)

#### `single_ies_shared_q.py`
**当前TensorFlow实现关键部分：**
```python
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# 变量定义
self.alpha = tf.Variable(alpha * self.rmax, dtype=tf.float32, trainable=True)
self.lambda_ = tf.Variable(lambda_ * self.rmax, trainable=True)

# 梯度计算
with tf.GradientTape() as tape_q:
    q1_loss, q2_loss, _, _ = self.compute_critic_loss(...)
    critic_loss = q1_loss + q2_loss

grads = tape_q.gradient(critic_loss, self.critic.trainable_variables)
self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
```

**需要改为PyTorch：**
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 变量定义
self.alpha = nn.Parameter(torch.tensor(alpha * self.rmax, dtype=torch.float32))
self.lambda_ = nn.Parameter(torch.tensor(lambda_ * self.rmax, dtype=torch.float32))

# 梯度计算
self.critic_optimizer.zero_grad()
q1_loss, q2_loss = self.compute_critic_loss(...)
critic_loss = q1_loss + q2_loss
critic_loss.backward()
self.critic_optimizer.step()
```

**具体需要改动的方法：**
- `__init__()`: 初始化网络、优化器、参数
- `compute_critic_loss()`: 损失计算逻辑
- `compute_constraint_critic_loss()`: 约束损失计算
- `compute_actor_loss()`: Actor损失计算
- `_update_from_batch()`: 整个训练更新流程
- `act()`: 动作选择方法

### 3. 联邦学习文件 (MEDIUM PRIORITY - 中等修改)

#### `Fed_CSAC_shared_q.py`
**当前TensorFlow实现：**
```python
# 模型权重聚合
def build_equal_avg_weights(models, prefixes):
    for m in models:
        for l in m.layers:
            if any(l.name.startswith(p) for p in prefixes):
                per_agent_ws.append(layer.get_weights())
    # 权重平均
    avg_weights = [np.mean([w[i] for w in valid_weights], axis=0) for i in range(len(template))]
```

**需要改为PyTorch：**
```python
# 模型权重聚合
def federated_average_state_dicts(models, layer_prefixes):
    avg_state_dict = {}
    for key in models[0].state_dict().keys():
        if any(key.startswith(prefix) for prefix in layer_prefixes):
            # 计算平均权重
            avg_state_dict[key] = torch.mean(
                torch.stack([model.state_dict()[key] for model in models]), 
                dim=0
            )
    return avg_state_dict
```

#### `Fed_train.py`
**类似的联邦学习逻辑需要适配PyTorch**

### 4. 经验回放缓冲区 (LOW PRIORITY - 轻微修改)

#### `network2/Prioritized_Replay.py` & `network/Prioritized_Replay.py`
**当前实现主要使用numpy，改动较小：**
- 将TensorFlow相关的类型转换改为PyTorch
- `tf.cast()` → `torch.tensor().to(dtype)`
- 其他逻辑基本保持不变

---

## 详细改动清单

### A. 导入语句改动
**所有相关文件需要修改导入：**
```python
# 删除
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate, LeakyReLU, BatchNormalization, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam

# 添加
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
```

### B. 网络定义改动

**1. 从Keras Model继承改为nn.Module继承**
```python
# TensorFlow
class network:
    def actor(self):
        # 返回 Model 实例
        return Model(inputs=..., outputs=...)

# PyTorch  
class ActorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义层
    
    def forward(self, x):
        # 前向传播
        return outputs
```

**2. 层定义语法改动**
```python
# TensorFlow
x = Dense(128, kernel_initializer=tf.keras.initializers.he_uniform())(x)
x = LeakyReLU()(x)
x = LayerNormalization()(x)

# PyTorch
self.dense = nn.Linear(input_dim, 128)
nn.init.kaiming_uniform_(self.dense.weight)
x = F.leaky_relu(self.dense(x))
x = F.layer_norm(x, x.size()[1:])
```

**3. Lambda层改动**
```python
# TensorFlow
normalized_state = Lambda(lambda x: (x - self.state_low) / self.state_range)(state_input)

# PyTorch (在forward中)
normalized_state = (state - self.state_low) / self.state_range
```

### C. 训练流程改动

**1. 梯度计算**
```python
# TensorFlow
with tf.GradientTape() as tape:
    loss = compute_loss(...)
grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))

# PyTorch
optimizer.zero_grad()
loss = compute_loss(...)
loss.backward()
optimizer.step()
```

**2. 优化器定义**
```python
# TensorFlow
self.actor_optimizer = Adam(learning_rate=lr_actor)

# PyTorch
self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
```

**3. 损失函数**
```python
# TensorFlow
loss = tf.reduce_mean((q1 - target_q) ** 2)

# PyTorch  
loss = torch.mean((q1 - target_q) ** 2)
```

### D. 特殊函数改动

**1. 数学运算**
```python
# TensorFlow → PyTorch
tf.tanh() → torch.tanh()
tf.clip_by_value() → torch.clamp()
tf.reduce_mean() → torch.mean()
tf.stop_gradient() → tensor.detach()
tf.random.normal() → torch.randn()
tf.nn.log_softmax() → F.log_softmax()
tf.nn.softmax() → F.softmax()
```

**2. 重参数化技巧**
```python
# TensorFlow
epsilon = tf.random.normal(shape=tf.shape(sigma))
sampled_action = mu + sigma * epsilon

# PyTorch
epsilon = torch.randn_like(sigma)
sampled_action = mu + sigma * epsilon
```

**3. Gumbel-Softmax实现**
```python
# TensorFlow
uniform_noise = tf.random.uniform(tf.shape(logits), minval=1e-8, maxval=1)
gumbel_noise = -tf.math.log(-tf.math.log(uniform_noise))

# PyTorch
uniform_noise = torch.rand_like(logits).clamp(min=1e-8, max=1.0)
gumbel_noise = -torch.log(-torch.log(uniform_noise))
```

---

## 迁移建议与注意事项

### 1. 迁移顺序建议
1. **先迁移网络定义** (`network2/`, `network/`)
2. **再迁移智能体核心** (`single_ies_shared_q.py`)  
3. **最后迁移联邦学习** (`Fed_CSAC_shared_q.py`, `Fed_train.py`)
4. **测试和调试**

### 2. 关键注意事项

**A. 设备管理**
PyTorch需要显式管理GPU/CPU：
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)
```

**B. 模型保存/加载**
```python
# 保存
torch.save(model.state_dict(), 'model.pth')
# 加载  
model.load_state_dict(torch.load('model.pth'))
```

**C. 评估模式**
```python
# 训练模式
model.train()
# 评估模式 (关闭dropout, batch_norm等)
model.eval()
```

**D. 联邦学习权重聚合**
PyTorch的state_dict结构与TensorFlow的get_weights()不同，需要适配

### 3. 验证要点
- 确保网络输出维度和范围一致
- 验证损失函数计算结果
- 检查梯度更新是否正常
- 对比联邦平均结果
- 确保训练收敛行为一致

---

## 预估工作量

- **网络定义重写**: 2-3天
- **训练逻辑改写**: 2-3天  
- **联邦学习适配**: 1-2天
- **测试调试**: 2-3天
- **总计**: 约1-1.5周

**建议先做小规模验证，确保单个组件工作正常后再整合。**