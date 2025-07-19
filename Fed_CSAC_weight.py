import numpy as np
import matplotlib.pyplot as plt
from env import CombinedEnergyEnv
from env import da_market_clearing, get_market_prices_car
from single_ies import C_SAC_
from env.carbon import calculate_carbon_quota_split

# 获取时序定价
tou_buy, fit_sell, car_buy, car_sell, grid_co2 = get_market_prices_car()

def federated_weighted_average(agents):
    """
    FedWAvg-电负荷版：只用 avg_P（平均电负荷）作为权重。
    要求每个 ag.env 事先有 avg_P 属性。
    """
    # 1) 计算每个 agent 的权重系数 w_i，仅基于 avg_P
    load_sums = np.array([ag.env.avg_P for ag in agents], dtype=np.float64)  # 只取电负荷
    total = load_sums.sum()
    weights = load_sums / total    # 归一化为和为1的权重数组

    # 辅助函数：对一层权重列表做加权平均
    def weighted_average_layer(layer_weights_list):
        return sum(w * lw for w, lw in zip(weights, layer_weights_list))

    # 2) 聚合 actor
    actor_weights = [ag.actor.get_weights() for ag in agents]
    avg_actor = [
        weighted_average_layer(layer_layers)
        for layer_layers in zip(*actor_weights)
    ]

    # 3) 聚合 Q1 critic
    q1_weights = [ag.q1_critic.get_weights() for ag in agents]
    avg_q1 = [
        weighted_average_layer(layer_layers)
        for layer_layers in zip(*q1_weights)
    ]

    # 4) 聚合 Q2 critic
    q2_weights = [ag.q2_critic.get_weights() for ag in agents]
    avg_q2 = [
        weighted_average_layer(layer_layers)
        for layer_layers in zip(*q2_weights)
    ]

    # 5) 下发加权参数并软更新
    for ag in agents:
        ag.actor.set_weights(avg_actor)
        ag.q1_critic.set_weights(avg_q1)
        ag.q2_critic.set_weights(avg_q2)
        ag.soft_update_all_targets()


def compute_c_n(p_grid_trade,car_emis,t,MG_car=8800):
    if p_grid_trade > 0:
        C_n = grid_co2[t]*p_grid_trade + car_emis - MG_car # 8800占位，不同MG8800位置不一样
    else:
        C_n = car_emis - 8800
        
    return C_n

def train_multi_agents(
    scenarios,
    max_rounds=10,
    max_steps=24,
    agg_interval=1,
    gamma=0.99,
    tau=0.005
):
    """
    多 MG 串并联训练：
    每轮:
      每步: 所有 MG act-> step -> 集中撮合 -> compute_trade_cost -> remember -> update
      每 agg_interval 轮后 FedAvg
    """


    # 初始化 agents
    agents = []

    for s in scenarios:
        env = CombinedEnergyEnv(s)
        ag = C_SAC_(env, gamma=gamma, tau=tau)
        agents.append(ag)
    
    n_agents = len(agents)
    critic_loss_hist   = [ [] for _ in range(n_agents) ]
    actor_loss_hist    = [ [] for _ in range(n_agents) ]
    qc_loss_hist       = [ [] for _ in range(n_agents) ]
    reward_hist        = [ [] for _ in range(n_agents) ]
    constraint_hist    = [ [] for _ in range(n_agents) ]

    # 全局训练轮
    for rnd in range(1, max_rounds+1):
        print(f"== 轮次 {rnd}/{max_rounds} ==")
        # 重置 envs
        states = [ag.env.reset() for ag in agents]
        dones  = [False]*len(agents)
        step   = 0

        # 每个 agent 本轮的累积指标
        total_reward_agent  = [0.0]*n_agents
        total_penalty_agent = [0.0]*n_agents

        # 单个 episode
        while not all(dones) and step < max_steps:
            # 并行选动作
            actions = []
            for ag, s in zip(agents, states):
                a, sampled = ag.act(s)
                ag._last_sample = sampled
                actions.append(a)

            # 并行执行 env.step，并收集 P_n/C_n 和本地 reward/emis
            local_emis = [0.0]*len(agents)
            local_cost_operate = []
            infos      = []
            P_buys, P_sells, C_buys, C_sells = [], [], [], []
            P_n_record = [0.0]*len(agents)
            C_n_record = [0.0]*len(agents)
            P_load_record = [0.0]*len(agents)

            for idx, (ag, a) in enumerate(zip(agents, actions)):
                # 只更新部分状态，获得 P_n 等
                _, done, info = ag.env.step(a)
                local_cost_operate.append(info['operate_cost'])
                infos.append(info)
                dones[idx] = done

                # 收集挂单信息
                P_n = info['P_n']
                P_n_record[idx] = P_n
                local_emis[idx] = info['carbon_emis']
                P_load_record[idx] = info['P_load']

                # 占位价格存在旧 state 的 8-11
                pb, ps, cb, cs = states[idx][8], states[idx][9],states[idx][10], states[idx][11]
                # 电碳盈余有正负
                if P_n > 0:  P_buys.append((idx, pb,  P_n))
                else:        P_sells.append((idx, ps, -P_n))

# ============================== 注意碳和电的索引是否对应 =====================================
            # 市场撮合
            # 电
            lambda_e_buy  = tou_buy[step]
            ele_res, e_p_m_buy, e_p_m_sell, e_summary = da_market_clearing(
                P_buys, P_sells,
                lambda_buy=lambda_e_buy, lambda_sell=fit_sell)
            
            for idx,v in ele_res.items():
                p_grid_trade = v['grid_qty']
                car_emis = local_emis[idx]
                MG_car = calculate_carbon_quota_split(P_load_record[idx]).get('quota_total',8800)
                C_n = compute_c_n(p_grid_trade,car_emis,step,MG_car=MG_car)
                C_n_record[idx] = C_n

                if C_n > 0: C_buys.append((idx, cb,  C_n))
                else:       C_buys.append((idx, cs,  -C_n))

            # 碳
            car_res, car_p_m_buy, car_p_m_sell, car_summary = da_market_clearing(
                C_buys, C_sells,
                lambda_buy=car_buy,lambda_sell=car_sell)
            

# ====================================================================================================================
            # 执行交易成本计算及完整状态更新，并存储/训练
            next_states = []
            for idx, ag in enumerate(agents):
                info = infos[idx]
                # compute_trade_cost 返回 (trade_cost, full_state)
                trade_cost, full_state = ag.env.compute_trade_cost(
                    P_n_record[idx], C_n_record[idx],
                    elec_price_buy  = e_p_m_buy,
                    elec_price_sell = e_p_m_sell,
                    carbon_price_buy  = car_p_m_buy,
                    carbon_price_sell = car_p_m_sell
                )
                # 组合总 reward
                new_r = -(
                    local_cost_operate[idx] + trade_cost
                )

                total_reward_agent[idx]  += new_r*1e-6
                total_penalty_agent[idx] += info['penalty']*1e-8

                # 记忆使用完整状态
                ag.remember(
                    states[idx], ag._last_sample,
                    new_r, full_state, dones[idx], info['penalty']
                )
                # 更新网络
                batch = ag.replay()
                if batch is not None:
                    q1_l, q2_l, qc_l, a_l = ag._update_from_batch(batch)
                    critic_loss_hist[idx].append( (q1_l + q2_l) / 2 )
                    qc_loss_hist[idx].append( qc_l )
                    actor_loss_hist[idx].append( a_l )

                next_states.append(full_state)

            # 迭代
            states = next_states
            step  += 1

        # Episode 结束：打印并记录每个 agent 的指标
        for ies in range(n_agents):
            print(f" Agent{ies} | Reward: {total_reward_agent[idx]:.3f} | Penalty: {total_penalty_agent[idx]:.3f}")
            reward_hist[idx].append(total_reward_agent[idx])
            constraint_hist[idx].append(total_penalty_agent[idx])

        # 周期联邦平均
        if rnd % agg_interval == 0:
            print(f"-- 执行联邦聚合 (轮次 {rnd}) --")
            federated_weighted_average(agents)
    
    # 训练结束后画图
    for idx in range(n_agents):
        plt.figure()
        plt.plot(reward_hist[idx],        label='Reward')
        plt.plot(constraint_hist[idx],    label='Penalty')
        plt.title(f'Agent {idx} Episode Metrics')
        plt.xlabel('Episode'); plt.ylabel('Value')
        plt.legend(); plt.show()

    for idx in range(n_agents):
        plt.figure()
        plt.plot(critic_loss_hist[idx], label='Critic Loss')
        plt.plot(actor_loss_hist[idx],  label='Actor Loss')
        plt.plot(qc_loss_hist[idx],     label='Constraint Loss')
        plt.title(f'Agent {idx} Loss Curves')
        plt.xlabel('Training Steps'); plt.ylabel('Loss')
        plt.legend(); plt.show()

    return agents


if __name__=='__main__':
    scenarios=['IES1','IES2','IES3']
    # scenarios=['IES1']
    agents = train_multi_agents(scenarios)
