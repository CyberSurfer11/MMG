
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from typing import List, Tuple, Dict

def da_market_clearing(
    buys:  List[Tuple[str, float, float]],   # (buyer_id,  bid_price,  quantity>0)
    sells: List[Tuple[str, float, float]],   # (seller_id, ask_price, quantity>0)
    lambda_buy:  float,                      # ToU—从电网买电价  λtᵇ
    lambda_sell: float                       # FiT—向电网卖电价  λtˢ
) -> Dict[str, Dict[str, float]]:
    """
    执行 1 h DA P2P 市场出清。
    返回：
        {agent_id: {
            'p2p_qty' : 已在 P2P 市场成交的电量  (买为 +, 卖为 −),
            'p2p_cost': P2P 总支出(+)/收入(−)，已含撮合中点价，
            'grid_qty': 与上级电网成交的电量   (买为 +, 卖为 −),
            'grid_cost': 向电网支付(+)/收入(−)
        }}
    """
    # ---- 1. 订单簿按价格排序 ----
    buy_book  = sorted(buys,  key=lambda x: -x[1])  # 价格高→低
    sell_book = sorted(sells, key=lambda x:  x[1])  # 价格低→高

    # ---- 2. 初始化结果容器 ----
    res = defaultdict(lambda: {'p2p_qty': 0.0, 'p2p_cost': 0.0,
                               'grid_qty': 0.0, 'grid_cost': 0.0})

    b_idx = s_idx = 0  # 订单簿游标

    # ---- 3. 循环撮合 ----
    while b_idx < len(buy_book) and s_idx < len(sell_book):
        b_id, b_price, b_qty = buy_book[b_idx]
        s_id, s_price, s_qty = sell_book[s_idx]

        if b_price < s_price:      # 无法继续成交
            break

        match_qty = min(b_qty, s_qty)
        clr_price = 0.5 * (b_price + s_price)   # 中点价

        # 记录买家
        res[b_id]['p2p_qty']  += +match_qty
        res[b_id]['p2p_cost'] += +clr_price * match_qty

        # 记录卖家
        res[s_id]['p2p_qty']  += -match_qty
        res[s_id]['p2p_cost'] += -clr_price * match_qty

        # 更新剩余量
        buy_book[b_idx]  = (b_id, b_price, b_qty - match_qty)
        sell_book[s_idx] = (s_id, s_price, s_qty - match_qty)

        if buy_book[b_idx][2] == 0:   # 该买单撮合完
            b_idx += 1
        if sell_book[s_idx][2] == 0:  # 该卖单撮合完
            s_idx += 1

    # ---- 4. 剩余量走电网 ----
    for idx in range(b_idx, len(buy_book)):
        agent, _, qty = buy_book[idx]
        if qty > 0:
            res[agent]['grid_qty']  += +qty
            res[agent]['grid_cost'] += +lambda_buy * qty

    for idx in range(s_idx, len(sell_book)):
        agent, _, qty = sell_book[idx]
        if qty > 0:
            res[agent]['grid_qty']  += -qty
            res[agent]['grid_cost'] += -lambda_sell * qty

    return res


def get_market_prices():
    elec_price_buy = [
        5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8,
        10.2,
        16.4, 16.4, 16.4,
        10.2, 10.2,
        16.4, 16.4, 16.4,
        10.2, 10.2,
        16.4, 16.4, 16.4, 16.4,
        5.8, 5.8
    ]
    elec_price_sell = 4.8
    carbon_price_buy = [0.0]*24
    carbon_price_sell = [0.0]*24
    return elec_price_buy, elec_price_sell, carbon_price_buy, carbon_price_sell

def main():
    np.random.seed(0)
    hours = list(range(24))
    elec_price_buy, elec_price_sell, _, _ = get_market_prices()

    systems = ['RG', 'CG', 'IG']
    load = {
        'RG': 0.6 + 0.2 * np.sin(np.linspace(0, 2*np.pi, 24)) + 0.05 * np.random.randn(24),
        'CG': 1.0 + 0.3 * np.sin(np.linspace(-np.pi/3, 5*np.pi/3, 24)) + 0.05 * np.random.randn(24),
        'IG': 1.4 + 0.1 * np.sin(np.linspace(np.pi/2, 2.5*np.pi, 24)) + 0.05 * np.random.randn(24)
    }
    pv = np.maximum(0, np.sin(np.linspace(-np.pi/2, 3*np.pi/2, 24)))
    gen = {
        'RG': 0.5 * pv,
        'CG': 0.3 * pv,
        'IG': 0.1 * pv + 0.2 * np.random.rand(24)
    }

    p2p_buy_prices = []
    p2p_sell_prices = []

    for t in hours:
        buys, sells = [], []
        for sys in systems:
            l = load[sys][t]
            g = gen[sys][t]
            net = l - g
            ap = np.clip(np.random.normal(0.5, 0.15), 0, 1)
            p = elec_price_sell + ap * (elec_price_buy[t] - elec_price_sell)
            if net > 1e-3:
                buys.append((sys, p, round(net, 3)))
            elif net < -1e-3:
                sells.append((sys, p, round(-net, 3)))
        result = da_market_clearing(buys, sells, elec_price_buy[t], elec_price_sell)

        bq = bc = sq = sc = 0
        for r in result.values():
            if r['p2p_qty'] > 0:
                bq += r['p2p_qty']
                bc += r['p2p_cost']
            elif r['p2p_qty'] < 0:
                sq += -r['p2p_qty']
                sc += -r['p2p_cost']
        p2p_buy_prices.append(bc / bq if bq else np.nan)
        p2p_sell_prices.append(sc / sq if sq else np.nan)

    plt.figure(figsize=(10, 5))
    plt.step(hours, p2p_buy_prices,  where='mid', marker='o', label='P2P electricity purchase price')
    plt.step(hours, p2p_sell_prices, where='mid', marker='s', label='P2P electricity sale price')
    plt.step(hours, elec_price_buy,  where='mid', marker='D', label='Time-of-Use')
    plt.step(hours, [elec_price_sell]*24, where='mid', marker='^', label='Feed-in Tariff')
    plt.xlabel('Hour')
    plt.ylabel('Electricity price (¢/kWh)')
    plt.title('24-hour P2P electricity price (Test with 3 systems)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()