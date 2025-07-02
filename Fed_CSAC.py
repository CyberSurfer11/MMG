import numpy as np
from env import CombinedEnergyEnv
from env import da_market_clearing, get_market_prices
from single_ies import C_SAC_


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

    # 获取时序定价
    tou_buy, fit_sell, car_buy, car_sell = get_market_prices()

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
            local_rs   = []
            local_emis = []
            local_cost_operate = []
            infos      = []
            P_buys, P_sells, C_buys, C_sells = [], [], [], []

            for idx, (ag, a) in enumerate(zip(agents, actions)):
                # 只更新部分状态，获得 P_n/C_n 等
                _, r_loc, done, info = ag.env.step(a)
                local_rs.append(r_loc)
                local_emis.append(info['total_emis'])
                local_cost_operate.append(info['total_cost'])
                infos.append(info)
                dones[idx] = done

                # 收集挂单信息
                P_n, C_n = info['P_n'], info['C_n']
                # 占位价格存在旧 state 的 8-11
                pb, ps, cb, cs = states[idx][8], states[idx][9], states[idx][10], states[idx][11]
                if P_n > 0:  P_buys.append((idx, pb,  P_n))
                else:        P_sells.append((idx, ps, -P_n))
                if C_n > 0:  C_buys.append((idx, cb,  C_n))
                else:        C_sells.append((idx, cs, -C_n))

            # 市场撮合
            lambda_buy  = tou_buy[step]
            ele_res, p_plus, p_minus = da_market_clearing(
                P_buys, P_sells,
                lambda_buy=lambda_buy, lambda_sell=fit_sell)
            car_res, c_plus, c_minus = da_market_clearing(
                C_buys, C_sells,
                lambda_buy=car_buy,    lambda_sell=car_sell)

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
