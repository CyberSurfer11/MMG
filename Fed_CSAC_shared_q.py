import numpy as np
import matplotlib.pyplot as plt
from env import CombinedEnergyEnv
from env import da_market_clearing, get_market_prices_car
from single_ies_shared_q import C_SAC_
from env.carbon import calculate_carbon_quota_split

# 获取时序定价
tou_buy, fit_sell, car_buy, car_sell, grid_co2 = get_market_prices_car()

def federated_weighted_average(agents, personal_steps=1):
    """
    等权联邦聚合 + 个性化更新（最小改动版）
    - 等权聚合共享浅层（Actor_Shared_*, Critic_Shared_*；兼容旧前缀 Shared_*）。
    - 在本函数内对个性化深层做本地更新：临时冻结共享层，调用现有 ag.replay() / ag._update_from_batch()。
    """

    # --------- 小工具：按前缀冻结/解冻 ----------
    def set_trainable_by_prefix(model, prefixes, trainable):
        if isinstance(prefixes, str):
            prefixes = (prefixes,)
        for l in model.layers:
            if any(l.name.startswith(p) for p in prefixes):
                l.trainable = trainable

    # --------- 小工具：构建“等权聚合”的层名->权重字典（仅匹配指定前缀） ----------
    def build_equal_avg_weights(models, prefixes):
        if isinstance(prefixes, str):
            prefixes = (prefixes,)
        # 收集所有匹配的层名
        name_set = set()
        for m in models:
            for l in m.layers:
                if any(l.name.startswith(p) for p in prefixes):
                    name_set.add(l.name)
        # 逐层做等权平均（只对有权重的层）
        name_to_avg = {}
        for lname in name_set:
            per_agent_ws = []
            for m in models:
                layer = next((x for x in m.layers if x.name == lname), None)
                if layer is None:
                    per_agent_ws.append(None)
                else:
                    per_agent_ws.append(layer.get_weights())
            if all((w is None or len(w) == 0) for w in per_agent_ws):
                continue
            template = next((w for w in per_agent_ws if (w is not None and len(w) > 0)), None)
            if template is None:
                continue
            n = sum(1 for w in per_agent_ws if (w is not None and len(w) > 0))
            avg_ws = []
            for arr_idx in range(len(template)):
                acc = None
                for wlist in per_agent_ws:
                    if wlist is None or len(wlist) == 0:
                        continue
                    arr = wlist[arr_idx]
                    acc = arr if acc is None else (acc + arr)
                avg_ws.append(acc / float(n))
            name_to_avg[lname] = avg_ws
        return name_to_avg

    # --------- 1) 先做“个性化深层”的本地更新（冻结共享层，只训深层） ----------
    # 共享层前缀（新命名 & 兼容老命名）
    ACTOR_SHARED_PREFIX = "Actor_Shared_"
    CRITIC_SHARED_PREFIXES = ("Critic_Shared_", "Shared_")

    for ag in agents:
        # 冻结共享层
        set_trainable_by_prefix(ag.actor, ACTOR_SHARED_PREFIX, False)
        # === 修改点 #1：critic 是单个模型 ===
        set_trainable_by_prefix(ag.critic, CRITIC_SHARED_PREFIXES, False)

        # 个性化更新若干步（只会更新未被冻结的“深层”）
        for _ in range(max(1, personal_steps)):
            batch = ag.replay()
            if batch is not None:
                ag._update_from_batch(batch)

        # 恢复共享层可训练标志（不改变外部训练行为）
        set_trainable_by_prefix(ag.actor, ACTOR_SHARED_PREFIX, True)
        # === 修改点 #2：critic 是单个模型 ===
        set_trainable_by_prefix(ag.critic, CRITIC_SHARED_PREFIXES, True)

    # --------- 2) 对“共享浅层”做等权聚合并下发 ----------
    # Actor 共享浅层
    actor_models = [ag.actor for ag in agents]
    actor_avg = build_equal_avg_weights(actor_models, prefixes=ACTOR_SHARED_PREFIX)

    # Critic 共享浅层（单模型双头：只聚合主干浅层）
    # === 修改点 #3：critic 平台化为单个模型列表 ===
    critic_models = [ag.critic for ag in agents]
    critic_avg = build_equal_avg_weights(critic_models, prefixes=CRITIC_SHARED_PREFIXES)

    # 应用平均权重到所有 agent 的共享层（个性化层保持本地参数）
    def apply_avg(model, avg_dict):
        name_to_layer = {l.name: l for l in model.layers}
        for lname, avg_ws in avg_dict.items():
            layer = name_to_layer.get(lname)
            if layer is not None and avg_ws is not None and len(avg_ws) > 0:
                try:
                    layer.set_weights(avg_ws)
                except Exception:
                    pass

    for ag in agents:
        apply_avg(ag.actor, actor_avg)
        # === 修改点 #4：下发到单个 critic 模型 ===
        apply_avg(ag.critic, critic_avg)
        # 目标网络软更新保持不变
        ag.soft_update_all_targets()


def compute_c_n(p_grid_trade,car_emis,t,MG_car=8800):
    if p_grid_trade > 0:
        C_n = grid_co2[t]*p_grid_trade + car_emis - MG_car # 8800占位，不同MG8800位置不一样
    else:
        C_n = car_emis - MG_car
    return C_n

def train_multi_agents(
    scenarios,
    max_rounds=5000,
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

    critic_loss_ep_hist = [[] for _ in range(n_agents)]
    actor_loss_ep_hist  = [[] for _ in range(n_agents)]
    qc_loss_ep_hist     = [[] for _ in range(n_agents)]


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
            samples = []  
            for ag, s in zip(agents, states):
                a, sampled = ag.act(s)
                actions.append(a)
                samples.append(sampled)

            # 并行执行 env.step，并收集 P_n/C_n 和本地 reward/emis
            local_emis = [0.0]*len(agents)
            local_cost_operate = []
            infos      = []
            P_buys, P_sells, C_buys, C_sells = [], [], [], []
            P_n_record = [0.0]*len(agents)
            C_n_record = [0.0]*len(agents)
            P_load_record = [0.0]*len(agents)

            pb_rec = [0.0]*n_agents; ps_rec = [0.0]*n_agents
            cb_rec = [0.0]*n_agents; cs_rec = [0.0]*n_agents

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
                pb_rec[idx] = states[idx][8];  ps_rec[idx] = states[idx][9]
                cb_rec[idx] = states[idx][10]; cs_rec[idx] = states[idx][11]
                # 电碳盈余有正负
                if P_n > 0:  P_buys.append((idx, pb_rec[idx],  P_n))
                else:        P_sells.append((idx, ps_rec[idx], -P_n))

            # ============================== 注意碳和电的索引是否对应 =====================================
            # 市场撮合
            # 电
            lambda_e_buy  = tou_buy[step]
            ele_res, e_p_m_buy, e_p_m_sell, e_summary = da_market_clearing(
                P_buys, P_sells,
                lambda_buy=lambda_e_buy, lambda_sell=fit_sell)
            
            for i,v in ele_res.items():
                p_grid_trade = v['grid_qty']
                car_emis = local_emis[i]
                MG_car = calculate_carbon_quota_split(P_load_record[i]).get('quota_total',37000)
                C_n = compute_c_n(p_grid_trade,car_emis,step,MG_car=MG_car)
                C_n_record[i] = C_n

                if C_n > 0: C_buys.append((i, cb_rec[i],  C_n))
                else:       C_sells.append((i, cs_rec[i],  -C_n))

            # 碳
            car_res, car_p_m_buy, car_p_m_sell, car_summary = da_market_clearing(
                C_buys, C_sells,
                lambda_buy=car_buy,lambda_sell=car_sell)
            

            # ===================================================================================================================
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
                    states[idx], samples[idx],
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
            print(f" Agent{ies} | Reward: {total_reward_agent[ies]:.3f} | Penalty: {total_penalty_agent[ies]:.3f}")
            reward_hist[ies].append(total_reward_agent[ies])
            constraint_hist[ies].append(total_penalty_agent[ies])

            if critic_loss_hist[ies]:
                critic_loss_ep_hist[ies].append(critic_loss_hist[ies][-1])
            if actor_loss_hist[ies]:
                actor_loss_ep_hist[ies].append(actor_loss_hist[ies][-1])
            if qc_loss_hist[ies]:
                qc_loss_ep_hist[ies].append(qc_loss_hist[ies][-1])


        # 周期联邦平均
        if rnd % agg_interval == 0:
            print(f"-- 执行联邦聚合 (轮次 {rnd}) --")
            federated_weighted_average(agents)
    
    # # 训练结束后画图
    # for idx in range(n_agents):
    #     plt.figure()
    #     plt.plot(reward_hist[idx],        label='Reward')
    #     plt.plot(constraint_hist[idx],    label='Penalty')
    #     plt.title(f'Agent {idx} Episode Metrics')
    #     plt.xlabel('Episode'); plt.ylabel('Value')
    #     plt.legend(); plt.show()

    # for idx in range(n_agents):
    #     plt.figure()
    #     plt.plot(critic_loss_hist[idx], label='Critic Loss')
    #     plt.plot(actor_loss_hist[idx],  label='Actor Loss')
    #     plt.plot(qc_loss_hist[idx],     label='Constraint Loss')
    #     plt.title(f'Agent {idx} Loss Curves')
    #     plt.xlabel('Training Steps'); plt.ylabel('Loss')
    #     plt.legend(); plt.show()

    # 训练结束后画图 —— 每个变量单独一张图
    import os

    # 确保输出目录存在
    outdir = "image_result"
    os.makedirs(outdir, exist_ok=True)

    def draw_and_save(y, title, ylabel, fname, label=None, xlabel=None):
        if not y:  # 空数据直接跳过
            return
        plt.figure()
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
        # 保存 PNG
        save_path = os.path.join(outdir, fname)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()   # 需要同时显示就保留；若不显示可改为 plt.close()
        # plt.close()  # 如果不想弹图窗口，把上面的 plt.show() 注释掉，并打开这一行

    # 训练结束后画图 —— 每个变量单独一张图，并保存
    for idx in range(n_agents):
        draw_and_save(
            reward_hist[idx],
            title=f"Agent {idx} - Reward per Episode",
            ylabel="Reward",
            fname=f"agent{idx}_reward.png",
            label="Reward",
            xlabel="Episode"
        )
        draw_and_save(
            constraint_hist[idx],
            title=f"Agent {idx} - Penalty per Episode",
            ylabel="Penalty",
            fname=f"agent{idx}_penalty.png",
            label="Penalty",
            xlabel="Episode"
        )

    for idx in range(n_agents):
        draw_and_save(
            critic_loss_ep_hist[idx],
            title=f"Agent {idx} - Critic Loss",
            ylabel="Loss",
            fname=f"agent{idx}_critic_loss.png",
            label="Critic Loss",
            xlabel="Episode"
        )
        draw_and_save(
            actor_loss_ep_hist[idx],
            title=f"Agent {idx} - Actor Loss",
            ylabel="Loss",
            fname=f"agent{idx}_actor_loss.png",
            label="Actor Loss",
            xlabel="Episode"
        )
        draw_and_save(
            qc_loss_ep_hist[idx],
            title=f"Agent {idx} - Constraint(Qc) Loss",
            ylabel="Loss",
            fname=f"agent{idx}_qc_loss.png",
            label="Constraint Loss",
            xlabel="Episode"
        )

    return agents


if __name__=='__main__':
    scenarios=['IES1','IES2','IES3']
    # scenarios=['IES1']
    agents = train_multi_agents(scenarios,max_rounds=5000)
