
"""
Demo: 3-MG (RG / CG / IG) Day-Ahead P2P electricity market clearing

· 连续双向竞价 + 中点价撮合
· 剩余电量走主网（ToU 购电价 / FiT 上网电价）
· 画出每小时 P2P 买卖结算价与电网标杆价
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict


# ----------------------------------------------------------------------
# 1. 市场出清：连续双向竞价 + 中点价
# ----------------------------------------------------------------------
def da_market_clearing(
    buys:  List[Tuple[str, float, float]],   # (buyer_id,  bid_price,  quantity>0)
    sells: List[Tuple[str, float, float]],   # (seller_id, ask_price, quantity>0)
    lambda_buy:  float,                      # ToU——从电网买电价
    lambda_sell: float                       # FiT——向电网卖电价
) -> Dict[str, Dict[str, float]]:
    """返回各主体在 P2P 与主电网的撮合结果。"""
    buy_book  = sorted(buys,  key=lambda x: -x[1])  # 价格高→低
    sell_book = sorted(sells, key=lambda x:  x[1])  # 价格低→高

    res = defaultdict(lambda: {
        'p2p_qty':  0.0, 'p2p_cost':  0.0,
        'grid_qty': 0.0, 'grid_cost': 0.0
    })

    b_idx = s_idx = 0
    # —— 1) P2P 撮合 ————————————————————————————
    while b_idx < len(buy_book) and s_idx < len(sell_book):
        b_id, b_price, b_qty = buy_book[b_idx]
        s_id, s_price, s_qty = sell_book[s_idx]

        if b_price < s_price:      # 无重叠，退出
            break

        qty   = min(b_qty, s_qty)
        price = 0.5 * (b_price + s_price)

        # 记录买家
        res[b_id]['p2p_qty']  += +qty
        res[b_id]['p2p_cost'] += +price * qty
        # 记录卖家
        res[s_id]['p2p_qty']  += -qty
        res[s_id]['p2p_cost'] += -price * qty

        # 更新剩余量
        buy_book[b_idx]  = (b_id, b_price, b_qty - qty)
        sell_book[s_idx] = (s_id, s_price, s_qty - qty)
        if buy_book[b_idx][2] == 0:
            b_idx += 1
        if sell_book[s_idx][2] == 0:
            s_idx += 1

    # —— 2) 剩余量走主网 ————————————————————————
    for idx in range(b_idx, len(buy_book)):
        agent, _, qty = buy_book[idx]          # buy_book 中剩余量必定 > 0
        res[agent]['grid_qty']  +=  qty
        res[agent]['grid_cost'] +=  qty * lambda_buy

    for idx in range(s_idx, len(sell_book)):
        agent, _, qty = sell_book[idx]         # sell_book 中剩余量必定 > 0
        res[agent]['grid_qty']  += -qty
        res[agent]['grid_cost'] += -qty * lambda_sell

    return res


# ----------------------------------------------------------------------
# 2. ToU / FiT 价格曲线
# ----------------------------------------------------------------------
def get_market_prices():
    elec_price_buy = (
        [5.8]*7 + [10.2] + [16.4]*3 + [10.2]*2 +
        [16.4]*3 + [10.2]*2 + [16.4]*4 + [5.8]*2
    )
    elec_price_sell = 4.8
    return elec_price_buy, elec_price_sell


# ----------------------------------------------------------------------
# 3. 主程序：生成三微网净功率 → 撮合 → 画图
# ----------------------------------------------------------------------
def main():
    np.random.seed(1)                 # 结果可复现
    hours = np.arange(24)

    # —— 3.1 光伏发电（kWh）  --------------------------------------------------
    pv_profile = np.maximum(0, np.sin((hours - 6) / 24 * 2 * np.pi))  # 日出≈6h
    gen = {
        'RG': 1.2 * pv_profile,       # Rooftop-rich
        'CG': 0.6 * pv_profile,       # Common
        'IG': 0.2 * pv_profile        # Industrial
    }

    # —— 3.2 负荷曲线（kWh）  ---------------------------------------------------
    load = {
        'RG': 0.8 + 0.3 * np.sin(hours / 24 * 2 * np.pi + np.pi/3) + 0.05 * np.random.randn(24),
        'CG': 1.0 + 0.4 * np.sin(hours / 24 * 2 * np.pi - np.pi/4) + 0.05 * np.random.randn(24),
        'IG': 1.2 + 0.2 * np.sin(hours / 24 * 2 * np.pi)           + 0.05 * np.random.randn(24)
    }

    elec_price_buy, elec_price_sell = get_market_prices()
    p2p_buy_prices, p2p_sell_prices = [], []

    # —— 3.3 逐时撮合  ---------------------------------------------------------
    for h in hours:
        buys, sells = [], []
        for sys in ['RG', 'CG', 'IG']:
            net = load[sys][h] - gen[sys][h]            # 正：缺电；负：盈余
            alpha = np.clip(np.random.normal(0.5, 0.15), 0, 1)  # 报价偏移因子
            price = elec_price_sell + alpha * (elec_price_buy[h] - elec_price_sell)

            if net > 1e-3:
                buys.append((sys, price, round(net, 3)))
            elif net < -1e-3:
                sells.append((sys, price, round(-net, 3)))

        result = da_market_clearing(buys, sells,
                                    elec_price_buy[h], elec_price_sell)

        # 汇总该小时加权平均 P2P 成交价
        bq = bc = sq = sc = 0.0
        for rec in result.values():
            if rec['p2p_qty'] > 0:
                bq += rec['p2p_qty'];  bc += rec['p2p_cost']
            elif rec['p2p_qty'] < 0:
                sq += -rec['p2p_qty']; sc += -rec['p2p_cost']
        p2p_buy_prices.append( bc / bq if bq else np.nan )
        p2p_sell_prices.append( sc / sq if sq else np.nan )

    # —— 3.4 画图  -------------------------------------------------------------
    plt.figure(figsize=(11, 5))
    plt.step(hours, p2p_buy_prices,  where='mid', marker='o', label='P2P buy price')
    plt.step(hours, p2p_sell_prices, where='mid', marker='s', label='P2P sell price')
    plt.step(hours, elec_price_buy,  where='mid', marker='D', label='Time-of-Use (grid buy)')
    plt.step(hours, [elec_price_sell] * 24, where='mid', marker='^',
             label='Feed-in Tariff (grid sell)')
    plt.xlabel('Hour of day')
    plt.ylabel('Electricity price (¢/kWh)')
    plt.title('24-h clearing prices with three MGs')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
