import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------
# 1. CDA + 统一价：把式 (7)–(8) 算在函数内部
# ----------------------------------------------------------------------
def da_market_clearing(
    buys:  List[Tuple[str, float, float]],   # (buyer_id, bid_price,   qty>0)
    sells: List[Tuple[str, float, float]],   # (seller_id, ask_price,  qty>0)
    lambda_buy:  float,                      # ToU 价 p_t^+
    lambda_sell: float                       # FiT 价 p_t^-
) -> Tuple[Dict[str, Dict[str, float]], float, float]:
    """
    返回 (各 MG 结果字典, 统一买价 p_m_plus, 统一卖价 p_m_minus)
    """
    # —— 0) 先用订单量直接算式 (7)–(8) 的统一价 ——————————
    D_plus  = sum(q for _, _, q in buys)      # D_t^{+}
    D_minus = sum(q for _, _, q in sells)     # |D_t^{-}|
    D_t     = D_plus - D_minus                # D_t  (正=缺电, 负=盈余)

    p_m_plus  = np.nan
    p_m_minus = np.nan
    if D_plus  > 1e-6:
        p_m_plus  = (lambda_buy * D_t + lambda_sell * D_minus) / D_plus
    if D_minus > 1e-6:
        p_m_minus = (lambda_sell * D_minus + lambda_buy * D_t) / D_minus

    # —— 1) 按报价高低进行 CDA 撮合 (量不变) ————————————
    buy_book  = sorted(buys,  key=lambda x: -x[1])
    sell_book = sorted(sells, key=lambda x:  x[1])
    res = defaultdict(lambda: {'p2p_qty': 0.0, 'p2p_cost': 0.0,
                               'grid_qty': 0.0, 'grid_cost': 0.0})

    b_idx = s_idx = 0
    while b_idx < len(buy_book) and s_idx < len(sell_book):
        b_id, b_price, b_qty = buy_book[b_idx]
        s_id, s_price, s_qty = sell_book[s_idx]
        if b_price < s_price:
            break
        q = min(b_qty, s_qty)
        p = 0.5 * (b_price + s_price)         # 中点价仅做记账
        res[b_id]['p2p_qty']  +=  q;  res[b_id]['p2p_cost'] +=  p * q
        res[s_id]['p2p_qty']  -=  q;  res[s_id]['p2p_cost'] -=  p * q
        buy_book[b_idx]  = (b_id, b_price, b_qty - q)
        sell_book[s_idx] = (s_id, s_price, s_qty - q)
        if buy_book[b_idx][2]  == 0: b_idx += 1
        if sell_book[s_idx][2] == 0: s_idx += 1

    # —— 2) 剩余量走电网 ————————————————————————————
    for idx in range(b_idx, len(buy_book)):
        agent, _, qty = buy_book[idx]
        res[agent]['grid_qty']  +=  qty
        res[agent]['grid_cost'] +=  qty * lambda_buy
    for idx in range(s_idx, len(sell_book)):
        agent, _, qty = sell_book[idx]
        res[agent]['grid_qty']  -=  qty
        res[agent]['grid_cost'] -=  qty * lambda_sell

    return res, p_m_plus, p_m_minus


# ----------------------------------------------------------------------
def get_market_prices():
    tou_buy = (
        [5.8]*7 + [10.2] + [16.4]*3 + [10.2]*2 +
        [16.4]*3 + [10.2]*2 + [16.4]*4 + [5.8]*2
    )
    fit_sell = 4.8
    return tou_buy, fit_sell, [0.0]*24, [0.0]*24


# ----------------------------------------------------------------------
def main():
    np.random.seed(1)
    hours = np.arange(24)
    pv = np.maximum(0, np.sin((hours - 6) / 24 * 2 * np.pi))
    gen = {'RG': 1.2*pv, 'CG': 0.6*pv, 'IG': 0.2*pv}
    load = {
        'RG': 0.8 + 0.3*np.sin(hours/24*2*np.pi + np.pi/3) + 0.05*np.random.randn(24),
        'CG': 1.0 + 0.4*np.sin(hours/24*2*np.pi - np.pi/4) + 0.05*np.random.randn(24),
        'IG': 1.2 + 0.2*np.sin(hours/24*2*np.pi)           + 0.05*np.random.randn(24)
    }

    tou_buy, fit_sell, _, _ = get_market_prices()
    p_plus_series, p_minus_series = [], []

    for h in hours:
        buys = []; sells = []
        for sys in ['RG', 'CG', 'IG']:
            net = load[sys][h] - gen[sys][h]
            alpha = np.clip(np.random.normal(0.5, 0.15), 0, 1)
            bid   = fit_sell + alpha * (tou_buy[h] - fit_sell)
            if net > 1e-3:
                buys.append((sys, bid, round(net, 3)))
            elif net < -1e-3:
                sells.append((sys, bid, round(-net, 3)))

        _, p_plus, p_minus = da_market_clearing(
            buys, sells, tou_buy[h], fit_sell)
        p_plus_series.append(p_plus); p_minus_series.append(p_minus)

    # ----------------------------- Plot ------------------------------
    plt.figure(figsize=(11, 5))
    plt.step(hours, p_plus_series,  where='mid', marker='o', label='Unified buy price $p_{m,t}^+$')
    plt.step(hours, p_minus_series, where='mid', marker='s', label='Unified sell price $p_{m,t}^-$')
    plt.step(hours, tou_buy,        where='mid', marker='D', label='ToU (grid buy)')
    plt.step(hours, [fit_sell]*24,  where='mid', marker='^', label='FiT (grid sell)')
    plt.xlabel('Hour of day'); plt.ylabel('Electricity price (¢/kWh)')
    plt.title('24-h unified P2P prices (3 MGs, AE-2025 scheme)')
    plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout(); plt.show()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
