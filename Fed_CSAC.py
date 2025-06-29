import numpy as np  # 用于数值计算，包括参数平均等
import datetime       # 用于生成时间戳日志目录
import os             # 文件和目录操作
from env import CombinedEnergyEnv  # 引入自定义环境工厂函数
from single_ies import C_SAC_  # 引入单系统智能体类

# ---------- Federated Averaging Helper ----------
def federated_average(agents):
    """
    对所有 agents 的网络参数进行联邦平均，并下发更新。
    参数:
      agents: 智能体实例列表
    """
    # 1. 收集所有客户端 Actor 网络的权重列表
    actor_weights = [agent.actor.get_weights() for agent in agents]
    # 2. 按层对权重求平均
    avg_actor = [np.mean(w_list, axis=0) for w_list in zip(*actor_weights)]

    # 3. 同理，对 q1、q2、qc 三个 Critic 网络求平均
    q1_weights = [agent.q1_critic.get_weights() for agent in agents]
    q2_weights = [agent.q2_critic.get_weights() for agent in agents]
    qc_weights = [agent.qc_critic.get_weights() for agent in agents]
    avg_q1 = [np.mean(w_list, axis=0) for w_list in zip(*q1_weights)]
    avg_q2 = [np.mean(w_list, axis=0) for w_list in zip(*q2_weights)]
    avg_qc = [np.mean(w_list, axis=0) for w_list in zip(*qc_weights)]

    # 4. 将平均后的权重下发给每个客户端，并同步它们的目标网络
    for agent in agents:
        agent.actor.set_weights(avg_actor)
        agent.q1_critic.set_weights(avg_q1)
        agent.q2_critic.set_weights(avg_q2)
        agent.qc_critic.set_weights(avg_qc)
        # 同步对应的 target 网络
        agent.q1_critic_target.set_weights(avg_q1)
        agent.q2_critic_target.set_weights(avg_q2)
        agent.qc_critic_target.set_weights(avg_qc)

# ---------- 联邦训练函数 ----------
def train_federated(env_ids, max_rounds=100, local_episodes=1, **agent_kwargs):
    """
    联邦训练主循环：
      env_ids: 列表，IES 环境标识符，如 ['IES1','IES2','IES3']
      max_rounds: 联邦轮数
      local_episodes: 每轮本地训练的 episode 数
      agent_kwargs: 初始化 C_SAC_ 的参数，如学习率、网络结构等
    """
    # 1. 初始化所有客户端（每个 IES 对应一个 C_SAC_ 实例）
    agents = []
    for eid in env_ids:
        env = CombinedEnergyEnv(eid)              # 根据标识创建环境
        agent = C_SAC_(env, **agent_kwargs)       # 用相同超参初始化智能体
        agents.append(agent)

    # 2. 创建日志目录
    os.makedirs('logs/federated', exist_ok=True)
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # 3. 联邦训练循环
    for rnd in range(1, max_rounds+1):
        print(f"===== Federated Round {rnd}/{max_rounds} =====")
        # 3.1 每个客户端进行本地训练
        for idx, agent in enumerate(agents, start=1):
            print(f"-- Client {idx} local training {local_episodes} episodes --")
            agent.train(max_episodes=local_episodes)  # 本地多轮训练

        # 3.2 本轮本地训练完成后，执行参数聚合
        print("-- Aggregating parameters via FedAvg --")
        federated_average(agents)

    # 4. 联邦训练结束后，保存每个客户端的最终模型
    for idx, agent in enumerate(agents, start=1):
        agent.save_model(
            f"federated_agent{idx}_actor.h5",
            f"federated_agent{idx}_q1.h5",
            f"federated_agent{idx}_q2.h5",
            f"federated_agent{idx}_qc.h5"
        )
    print("Federated training complete. Models saved.")

# ---------- 主入口 ----------
if __name__ == '__main__':
    # 定义三个 IES 环境标识符
    env_ids = ['IES1', 'IES2', 'IES3']
    # 调用联邦训练函数，传入环境列表及智能体超参
    train_federated(
        env_ids,
        max_rounds=50,       # 总共 50 轮联邦聚合
        local_episodes=5,    # 每轮每个客户端本地训练 5 个 episode
        alpha=0.1,
        lambda_=0.1,
        constraint_threshold=0.01,
        lr_actor=5e-5,
        lr_critic=1e-4,
        lr_entropy=1e-4,
        use_priority=True,
        actor_units=(512, 256, 64, 32),
        critic_units=(128, 128, 32),
        tau=1e-3,
        gamma=0.85,
        batch_size=64,
        memory_cap=150000,
        eta_lambda=0.001,
        lambda_max=100.0,
        delta_lambda_min=-0.001,
        delta_lambda_max=0.001,
        target_entropy=-25,
        rmax=2900 * 1e-4
    )  # 运行后将依次打印训练进度并保存模型
