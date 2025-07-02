import numpy as np
from env import CombinedEnergyEnv
from env import da_market_clearing, get_market_prices_car
from single_ies import C_SAC_

# 获取时序定价
tou_buy, fit_sell, car_buy, car_sell, grid_co2 = get_market_prices_car()

def federated_average(agents):
    """
    联邦平均：对所有客户端 agent 的 actor 与 critic 网络参数做平均后下发。
    """
    n = len(agents)
    # actor 聚合
    avg_actor = None
    for ag in agents:
        state = ag.actor.state_dict()
        if avg_actor is None:
            avg_actor = {k: v.clone() for k, v in state.items()}
        else:
            for k, v in state.items():
                avg_actor[k] += v
    for k in avg_actor:
        avg_actor[k] /= n
    # critic 聚合
    avg_critic = None
    for ag in agents:
        state = ag.critic.state_dict()
        if avg_critic is None:
            avg_critic = {k: v.clone() for k, v in state.items()}
        else:
            for k, v in state.items():
                avg_critic[k] += v
    for k in avg_critic:
        avg_critic[k] /= n
    # 下发平均参数
    for ag in agents:
        ag.actor.load_state_dict(avg_actor)
        ag.critic.load_state_dict(avg_critic)

def compute_c_n(p_grid_trade,car_emis,t,MG_car=8800):
    if p_grid_trade > 0:
        C_n = grid_co2[t]*p_grid_trade + car_emis - MG_car # 8800占位，不同MG8800位置不一样
    else:
        C_n = car_emis - 8800
        
    return C_n

def train_multi_agents(
    scenarios,
    max_rounds=50,
    max_steps=24,
    agg_interval=5,
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



    # 全局训练轮
    for rnd in range(1, max_rounds+1):
        print(f"== 轮次 {rnd}/{max_rounds} ==")
        # 重置 envs
        states = [ag.env.reset() for ag in agents]
        dones  = [False]*len(agents)
        step   = 0

        # 单个 episode
        while not all(dones) and step < max_steps:
            # 并行选动作
            actions = []
            for ag, s in zip(agents, states):
                a, sampled = ag.act(s)
                ag._last_sample = sampled
                actions.append(a)

            # 并行执行 env.step，并收集 P_n/C_n 和本地 reward/emis
            local_r_trade = []
            local_emis = []
            local_cost_operate = []
            infos      = []
            P_buys, P_sells, C_buys, C_sells = [], [], [], []

            for idx, (ag, a) in enumerate(zip(agents, actions)):
                # 只更新部分状态，获得 P_n 等
                _, done, info = ag.env.step(a)
                local_cost_operate.append(info['operate_cost'])
                infos.append(info)
                dones[idx] = done

                # 收集挂单信息
                P_n = info['P_n']
                local_emis.append({'idx':info['carbon_emis']})

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
                C_n = compute_c_n(p_grid_trade,car_emis,step,MG_car=8800)

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
                    info['P_n'], info['C_n'],
                    elec_price_buy  = p_plus,
                    elec_price_sell = p_minus,
                    carbon_price_buy  = c_plus,
                    carbon_price_sell = c_minus
                )
                # 组合总 reward
                new_r = (
                    local_rs[idx]
                    - trade_cost * 1e-3
                    - local_emis[idx] * 0.01 * 1e-3
                )
                # 记忆使用完整状态
                ag.remember(
                    states[idx], ag._last_sample,
                    new_r, full_state, dones[idx], info['total_penalty']
                )
                # 更新网络
                batch = ag.replay()
                if batch is not None:
                    ag._update_from_batch(batch)

                next_states.append(full_state)

            # 迭代
            states = next_states
            step  += 1

        # 周期联邦平均
        if rnd % agg_interval == 0:
            print(f"-- 执行联邦聚合 (轮次 {rnd}) --")
            federated_average(agents)

    return agents


if __name__=='__main__':
    scenarios=['IES1','IES2','IES3','IES4']
    agents = train_multi_agents(scenarios)
