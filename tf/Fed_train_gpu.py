import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from env import CombinedEnergyEnv
from env import da_market_clearing, get_market_prices_car
from single_ies_shared_q_gpu import C_SAC_GPU
from env.carbon import calculate_carbon_quota_split

# === NEW: 记录与导出 ===
import pandas as pd
import json
import os

# 获取时序定价
tou_buy, fit_sell, car_buy, car_sell, grid_co2 = get_market_prices_car()

def federated_weighted_average_gpu(agents, personal_steps=1):
    """
    PyTorch版本的等权联邦聚合 + 个性化更新（带记录功能）
    - 等权聚合共享浅层（Actor_Shared_*, Critic_Shared_*）
    - 在本函数内对个性化深层做本地更新：临时冻结共享层，调用现有 ag.replay() / ag._update_from_batch()
    """
    def set_trainable_by_prefix(model, prefixes, trainable):
        if isinstance(prefixes, str):
            prefixes = (prefixes,)
        for name, param in model.named_parameters():
            if any(name.startswith(p.replace("_", ".")) for p in prefixes):
                param.requires_grad = trainable

    def build_equal_avg_weights_gpu(models, prefixes):
        if isinstance(prefixes, str):
            prefixes = (prefixes,)
        
        # 收集所有匹配的参数名
        param_names = set()
        for model in models:
            for name in model.state_dict().keys():
                formatted_name = name.replace(".", "_")
                if any(formatted_name.startswith(p) for p in prefixes):
                    param_names.add(name)
        
        # 逐参数做等权平均
        name_to_avg = {}
        for param_name in param_names:
            param_tensors = []
            for model in models:
                if param_name in model.state_dict():
                    param_tensors.append(model.state_dict()[param_name])
            
            if param_tensors:
                avg_param = torch.mean(torch.stack(param_tensors), dim=0)
                name_to_avg[param_name] = avg_param
                
        return name_to_avg

    ACTOR_SHARED_PREFIX = "shared_layers"
    CRITIC_SHARED_PREFIXES = ("shared_layers",)

    for ag in agents:
        set_trainable_by_prefix(ag.actor, ACTOR_SHARED_PREFIX, False)
        set_trainable_by_prefix(ag.critic, CRITIC_SHARED_PREFIXES, False)
        for _ in range(max(1, personal_steps)):
            batch = ag.replay()
            if batch is not None:
                ag._update_from_batch(batch)
        set_trainable_by_prefix(ag.actor, ACTOR_SHARED_PREFIX, True)
        set_trainable_by_prefix(ag.critic, CRITIC_SHARED_PREFIXES, True)

    actor_models = [ag.actor for ag in agents]
    actor_avg = build_equal_avg_weights_gpu(actor_models, prefixes=ACTOR_SHARED_PREFIX)

    critic_models = [ag.critic for ag in agents]
    critic_avg = build_equal_avg_weights_gpu(critic_models, prefixes=CRITIC_SHARED_PREFIXES)

    def apply_avg_gpu(model, avg_dict):
        with torch.no_grad():
            state_dict = model.state_dict()
            for param_name, avg_weight in avg_dict.items():
                if param_name in state_dict:
                    state_dict[param_name].copy_(avg_weight)

    for ag in agents:
        apply_avg_gpu(ag.actor, actor_avg)
        apply_avg_gpu(ag.critic, critic_avg)
        ag.soft_update_all_targets()


def compute_c_n(p_grid_trade, car_emis, t, MG_car=8800):
    if p_grid_trade > 0:
        C_n = grid_co2[t] * p_grid_trade + car_emis - MG_car
    else:
        C_n = car_emis - MG_car
    return C_n


def train_and_export_gpu(
    scenarios,
    max_rounds=5000,
    max_steps=24,
    agg_interval=1,
    gamma=0.99,
    tau=0.005,
    device=None,
    export_dir="training_results"
):
    """
    PyTorch版本的训练+记录导出功能：
    - 训练多智能体联邦学习
    - 记录详细训练数据
    - 导出结果到Excel和JSON
    """
    # 设备设置
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 创建导出目录
    os.makedirs(export_dir, exist_ok=True)

    # 初始化 agents
    agents = []
    for s in scenarios:
        env = CombinedEnergyEnv(s)
        ag = C_SAC_GPU(env, gamma=gamma, tau=tau, device=device)
        agents.append(ag)
    
    n_agents = len(agents)
    
    # === NEW: 详细记录结构 ===
    training_log = {
        'scenarios': scenarios,
        'hyperparameters': {
            'max_rounds': max_rounds,
            'max_steps': max_steps,
            'agg_interval': agg_interval,
            'gamma': gamma,
            'tau': tau,
            'device': str(device)
        },
        'agents_data': {}
    }
    
    # 为每个智能体创建记录结构
    for idx in range(n_agents):
        training_log['agents_data'][f'agent_{idx}'] = {
            'scenario': scenarios[idx],
            'episode_rewards': [],
            'episode_penalties': [],
            'critic_losses': [],
            'actor_losses': [],
            'qc_losses': [],
            'step_losses': {
                'critic': [],
                'actor': [],
                'qc': []
            },
            'network_parameters': {
                'alpha_values': [],
                'lambda_values': []
            }
        }

    critic_loss_hist = [[] for _ in range(n_agents)]
    actor_loss_hist = [[] for _ in range(n_agents)]
    qc_loss_hist = [[] for _ in range(n_agents)]
    reward_hist = [[] for _ in range(n_agents)]
    constraint_hist = [[] for _ in range(n_agents)]

    # 全局训练轮
    for rnd in range(1, max_rounds + 1):
        print(f"== 轮次 {rnd}/{max_rounds} ==")
        
        # 重置 envs
        states = [ag.env.reset() for ag in agents]
        dones = [False] * len(agents)
        step = 0

        # 每个 agent 本轮的累积指标
        total_reward_agent = [0.0] * n_agents
        total_penalty_agent = [0.0] * n_agents

        # 单个 episode
        while not all(dones) and step < max_steps:
            # 并行选动作
            actions = []
            samples = []
            for ag, s in zip(agents, states):
                a, sampled = ag.act(s)
                actions.append(a)
                samples.append(sampled)

            # 并行执行 env.step，并收集信息
            local_emis = [0.0] * len(agents)
            local_cost_operate = []
            infos = []
            P_buys, P_sells, C_buys, C_sells = [], [], [], []
            P_n_record = [0.0] * len(agents)
            C_n_record = [0.0] * len(agents)
            P_load_record = [0.0] * len(agents)

            pb_rec = [0.0] * n_agents; ps_rec = [0.0] * n_agents
            cb_rec = [0.0] * n_agents; cs_rec = [0.0] * n_agents

            for idx, (ag, a) in enumerate(zip(agents, actions)):
                _, done, info = ag.env.step(a)
                local_cost_operate.append(info['operate_cost'])
                infos.append(info)
                dones[idx] = done

                # 收集挂单信息
                P_n = info['P_n']
                P_n_record[idx] = P_n
                local_emis[idx] = info['carbon_emis']
                P_load_record[idx] = info['P_load']

                pb_rec[idx] = states[idx][8]; ps_rec[idx] = states[idx][9]
                cb_rec[idx] = states[idx][10]; cs_rec[idx] = states[idx][11]
                
                if P_n > 0: P_buys.append((idx, pb_rec[idx], P_n))
                else: P_sells.append((idx, ps_rec[idx], -P_n))

            # 市场撮合 - 电
            lambda_e_buy = tou_buy[step]
            ele_res, e_p_m_buy, e_p_m_sell, e_summary = da_market_clearing(
                P_buys, P_sells,
                lambda_buy=lambda_e_buy, lambda_sell=fit_sell)
            
            for i, v in ele_res.items():
                p_grid_trade = v['grid_qty']
                car_emis = local_emis[i]
                MG_car = calculate_carbon_quota_split(P_load_record[i]).get('quota_total', 37000)
                C_n = compute_c_n(p_grid_trade, car_emis, step, MG_car=MG_car)
                C_n_record[i] = C_n

                if C_n > 0: C_buys.append((i, cb_rec[i], C_n))
                else: C_sells.append((i, cs_rec[i], -C_n))

            # 市场撮合 - 碳
            car_res, car_p_m_buy, car_p_m_sell, car_summary = da_market_clearing(
                C_buys, C_sells,
                lambda_buy=car_buy, lambda_sell=car_sell)

            # 执行交易成本计算及训练
            next_states = []
            for idx, ag in enumerate(agents):
                info = infos[idx]
                trade_cost, full_state = ag.env.compute_trade_cost(
                    P_n_record[idx], C_n_record[idx],
                    elec_price_buy=e_p_m_buy,
                    elec_price_sell=e_p_m_sell,
                    carbon_price_buy=car_p_m_buy,
                    carbon_price_sell=car_p_m_sell
                )
                
                new_r = -(local_cost_operate[idx] + trade_cost)
                total_reward_agent[idx] += new_r * 1e-6
                total_penalty_agent[idx] += info['penalty'] * 1e-8

                ag.remember(
                    states[idx], samples[idx],
                    new_r, full_state, dones[idx], info['penalty']
                )
                
                # 更新网络并记录损失
                batch = ag.replay()
                if batch is not None:
                    q1_l, q2_l, qc_l, a_l = ag._update_from_batch(batch)
                    critic_loss = (q1_l + q2_l) / 2
                    
                    critic_loss_hist[idx].append(critic_loss)
                    qc_loss_hist[idx].append(qc_l)
                    actor_loss_hist[idx].append(a_l)
                    
                    # === NEW: 记录到详细日志 ===
                    training_log['agents_data'][f'agent_{idx}']['step_losses']['critic'].append({
                        'round': rnd,
                        'step': step,
                        'value': critic_loss
                    })
                    training_log['agents_data'][f'agent_{idx}']['step_losses']['actor'].append({
                        'round': rnd,
                        'step': step,
                        'value': a_l
                    })
                    training_log['agents_data'][f'agent_{idx}']['step_losses']['qc'].append({
                        'round': rnd,
                        'step': step,
                        'value': qc_l
                    })

                next_states.append(full_state)

            states = next_states
            step += 1

        # Episode 结束记录
        for ies in range(n_agents):
            print(f" Agent{ies} | Reward: {total_reward_agent[ies]:.3f} | Penalty: {total_penalty_agent[ies]:.3f}")
            
            reward_hist[ies].append(total_reward_agent[ies])
            constraint_hist[ies].append(total_penalty_agent[ies])
            
            # === NEW: 记录到详细日志 ===
            agent_log = training_log['agents_data'][f'agent_{ies}']
            agent_log['episode_rewards'].append({
                'round': rnd,
                'value': total_reward_agent[ies]
            })
            agent_log['episode_penalties'].append({
                'round': rnd,
                'value': total_penalty_agent[ies]
            })
            
            # 记录网络参数
            agent_log['network_parameters']['alpha_values'].append({
                'round': rnd,
                'value': float(agents[ies].alpha.detach().cpu().item())
            })
            agent_log['network_parameters']['lambda_values'].append({
                'round': rnd,
                'value': float(agents[ies].lambda_.detach().cpu().item())
            })

        # 周期联邦平均
        if rnd % agg_interval == 0:
            print(f"-- 执行联邦聚合 (轮次 {rnd}) --")
            federated_weighted_average_gpu(agents)

    # === NEW: 导出训练结果 ===
    print("正在导出训练结果...")
    
    # 1. 导出JSON格式的完整日志
    json_path = os.path.join(export_dir, "training_log_gpu.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)
    print(f"详细训练日志已导出到: {json_path}")
    
    # 2. 导出Excel格式的汇总数据
    excel_path = os.path.join(export_dir, "training_summary_gpu.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        # 每个智能体的episode数据
        for idx in range(n_agents):
            df_data = {
                'Round': list(range(1, max_rounds + 1)),
                'Reward': reward_hist[idx],
                'Penalty': constraint_hist[idx]
            }
            
            # 添加损失数据（取每episode最后一个值）
            episode_critic_loss = []
            episode_actor_loss = []
            episode_qc_loss = []
            
            for ep in range(max_rounds):
                ep_critic_losses = [l for l in critic_loss_hist[idx] 
                                  if len(episode_critic_loss) * max_steps <= len(critic_loss_hist[idx])]
                ep_actor_losses = [l for l in actor_loss_hist[idx] 
                                 if len(episode_actor_loss) * max_steps <= len(actor_loss_hist[idx])]
                ep_qc_losses = [l for l in qc_loss_hist[idx] 
                              if len(episode_qc_loss) * max_steps <= len(qc_loss_hist[idx])]
                
                episode_critic_loss.append(ep_critic_losses[-1] if ep_critic_losses else 0)
                episode_actor_loss.append(ep_actor_losses[-1] if ep_actor_losses else 0)
                episode_qc_loss.append(ep_qc_losses[-1] if ep_qc_losses else 0)
            
            df_data['Critic_Loss'] = episode_critic_loss[:len(reward_hist[idx])]
            df_data['Actor_Loss'] = episode_actor_loss[:len(reward_hist[idx])]
            df_data['QC_Loss'] = episode_qc_loss[:len(reward_hist[idx])]
            
            df = pd.DataFrame(df_data)
            df.to_excel(writer, sheet_name=f'Agent_{idx}_{scenarios[idx]}', index=False)
        
        # 汇总统计
        summary_data = {
            'Agent': [f'Agent_{i}' for i in range(n_agents)],
            'Scenario': scenarios,
            'Final_Reward': [reward_hist[i][-1] if reward_hist[i] else 0 for i in range(n_agents)],
            'Avg_Reward': [np.mean(reward_hist[i]) if reward_hist[i] else 0 for i in range(n_agents)],
            'Final_Penalty': [constraint_hist[i][-1] if constraint_hist[i] else 0 for i in range(n_agents)],
            'Avg_Penalty': [np.mean(constraint_hist[i]) if constraint_hist[i] else 0 for i in range(n_agents)]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"训练摘要已导出到Excel: {excel_path}")

    # 3. 生成图表并保存
    outdir = os.path.join(export_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    
    def draw_and_save(y, title, ylabel, fname, label=None, xlabel=None):
        if not y:
            return
        plt.figure(figsize=(10, 6))
        if label is None:
            plt.plot(y)
        else:
            plt.plot(y, label=label)
            plt.legend()
        if xlabel: plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(outdir, fname)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()  # 不显示图窗口，只保存

    # 生成所有图表
    for idx in range(n_agents):
        draw_and_save(
            reward_hist[idx],
            title=f"Agent {idx} ({scenarios[idx]}) - Reward per Episode (PyTorch)",
            ylabel="Reward",
            fname=f"agent{idx}_reward_gpu.png",
            label="Reward",
            xlabel="Episode"
        )
        draw_and_save(
            constraint_hist[idx],
            title=f"Agent {idx} ({scenarios[idx]}) - Penalty per Episode (PyTorch)",
            ylabel="Penalty",
            fname=f"agent{idx}_penalty_gpu.png",
            label="Penalty",
            xlabel="Episode"
        )

    print(f"所有图表已保存到: {outdir}")

    # 4. 保存模型
    model_path = os.path.join(export_dir, "federated_models_gpu.pth")
    save_agents_gpu(agents, model_path)

    return agents, training_log


def save_agents_gpu(agents, filepath):
    """保存所有智能体的模型 - PyTorch版本"""
    saved_data = {}
    for i, agent in enumerate(agents):
        agent_data = {
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'qc_critic_state_dict': agent.qc_critic.state_dict(),
            'alpha': agent.alpha.data,
            'lambda_': agent.lambda_.data,
        }
        saved_data[f'agent_{i}'] = agent_data
    
    torch.save(saved_data, filepath)
    print(f"模型已保存到: {filepath}")


def load_agents_gpu(agents, filepath):
    """加载所有智能体的模型 - PyTorch版本"""
    saved_data = torch.load(filepath, map_location='cpu')
    
    for i, agent in enumerate(agents):
        agent_key = f'agent_{i}'
        if agent_key in saved_data:
            agent_data = saved_data[agent_key]
            agent.actor.load_state_dict(agent_data['actor_state_dict'])
            agent.critic.load_state_dict(agent_data['critic_state_dict'])
            agent.qc_critic.load_state_dict(agent_data['qc_critic_state_dict'])
            agent.alpha.data = agent_data['alpha']
            agent.lambda_.data = agent_data['lambda_']
    
    print(f"模型已从 {filepath} 加载")


if __name__ == '__main__':
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    scenarios = ['IES1', 'IES2', 'IES3']
    # scenarios = ['IES1']  # 单智能体测试
    
    # 运行训练并导出
    agents, training_log = train_and_export_gpu(
        scenarios, 
        max_rounds=5000,
        device=device,
        export_dir="training_results_gpu"
    )
    
    print("训练和导出完成！")
    print("结果文件：")
    print("- training_results_gpu/training_log_gpu.json (详细日志)")
    print("- training_results_gpu/training_summary_gpu.xlsx (Excel汇总)")
    print("- training_results_gpu/plots/ (图表)")
    print("- training_results_gpu/federated_models_gpu.pth (模型)")