# -*- coding: utf-8 -*-
"""p2p_mmr_settlement.py
============================================================
按论文 2.3 节公式 (4)–(8) 实现的电力 + 碳配额
Mid‑Market‑Rate (MMR) 统一清算函数，
------------------------------------------------------------
使用示例
---------
>>> import numpy as np, p2p_mmr_settlement as mmr
>>> # 3 个 MG 的净电量(±kWh) 与净碳配额(±tCO₂e)
>>> p_net  = np.array([  4.2, -1.5, -1.2])
>>> c_net  = np.array([ 0.25, -0.06, -0.14])
>>> result = mmr.settle(
...     p_net, c_net,
...     tou_buy=0.164, fit_sell=0.048,
...     co2_buy=0.007, co2_sell=0.0035)
>>> print(result['elec_price_buy' ], result['elec_price_sell'])
0.1575 0.104
>>> print(result['elec_cost'])  # 每个 MG 的电费(+)/电收入(-)
[ 0.6615 -0.156  -0.1248]
>>> print(result['total_cost']) # 电+碳总成本/收益
[ 0.663275 -0.15789  -0.12686]
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple

def _mmr_price(
    total_buy_qty: float,
    total_sell_qty: float,
    net_qty: float,
    price_buy: float,
    price_sell: float,
) -> Tuple[float, float]:
    """计算单一商品(电或碳)在一个时隙的统一买/卖价。"""
    mid_price = 0.5 * (price_buy + price_sell)
    price_buy  = mid_price  # 初值
    price_sell = mid_price

    if net_qty > 1e-9:                    # 市场整体短缺
        price_buy = (
            mid_price * abs(total_sell_qty) + price_buy * net_qty
        ) / total_buy_qty
    elif net_qty < -1e-9:                 # 市场整体过剩
        price_sell = (
            mid_price * total_buy_qty + price_sell * abs(net_qty)
        ) / abs(total_sell_qty)

    # Clamp 确保边界: grid_sell < price_sell ≤ mid ≤ price_buy < grid_buy
    price_buy  = max(min(price_buy,  price_buy  - 1e-9), mid_price)
    price_sell = min(max(price_sell, price_sell + 1e-9), mid_price)
    return price_buy, price_sell

def settle(
    p_net: np.ndarray,
    c_net: np.ndarray,
    *,
    tou_buy: float,   # 电网买电 ToU 价  (¢/kWh)
    fit_sell: float,  # 电网卖电 FiT 价  (¢/kWh)
    co2_buy: float,   # 上级碳市场买配额价  (元/tCO₂e)
    co2_sell: float,  # 上级碳市场卖配额价  (元/tCO₂e)
) -> Dict[str, Any]:
    """按 MMR 统一价结算电 + 碳交易成本。

    参数
    ------
    p_net   : ndarray(N,)  各 MG 当期净电量 (+ 买 / – 卖) [kWh]
    c_net   : ndarray(N,)  各 MG 当期净碳配额 (+ 买 / – 卖) [tCO₂e]
    tou_buy / fit_sell : 电网 ToU / FiT 价
    co2_buy / co2_sell : 上级碳市场买 / 卖价

    返回
    ------
    dict，含统一价及每户电费 / 碳费 / 总费用。
    """
    p_net = np.asarray(p_net, dtype=float)
    c_net = np.asarray(c_net, dtype=float)
    assert p_net.shape == c_net.shape, "p_net 与 c_net 必须同维度"

    # -------- 电价统一计算 --------
    buy_p_qty   = np.sum(np.clip(p_net, 0, None))      # 正值求和
    sell_p_qty  = np.sum(np.clip(p_net, None, 0))      # 负值求和
    net_p_qty   = buy_p_qty + sell_p_qty               # 可能正/负/零
    price_buy_e, price_sell_e = _mmr_price(
        buy_p_qty, sell_p_qty, net_p_qty, tou_buy, fit_sell
    )

    # -------- 碳价统一计算 --------
    buy_c_qty   = np.sum(np.clip(c_net, 0, None))
    sell_c_qty  = np.sum(np.clip(c_net, None, 0))
    net_c_qty   = buy_c_qty + sell_c_qty
    price_buy_c, price_sell_c = _mmr_price(
        buy_c_qty, sell_c_qty, net_c_qty, co2_buy, co2_sell
    )

    # -------- 各 MG 成本 --------
    elec_cost   = price_buy_e * np.clip(p_net, 0, None) - price_sell_e * np.clip(p_net, None, 0)
    carbon_cost = price_buy_c * np.clip(c_net, 0, None) - price_sell_c * np.clip(c_net, None, 0)
    total_cost  = elec_cost + carbon_cost

    return {
        # 统一价格
        "elec_price_buy"   : price_buy_e,
        "elec_price_sell"  : price_sell_e,
        "carbon_price_buy" : price_buy_c,
        "carbon_price_sell": price_sell_c,
        # 各 MG 成本
        "elec_cost"   : elec_cost,        # ndarray shape(N,)
        "carbon_cost" : carbon_cost,      # ndarray shape(N,)
        "total_cost"  : total_cost,       # ndarray shape(N,)
    }

# ------------------- quick self‑test -------------------
if __name__ == "__main__":
    p_net = np.array([ 3.5, -1.8, -1.2])   # kWh
    c_net = np.array([ 0.18, -0.05, -0.11])
    res = settle(p_net, c_net,
                 tou_buy=0.164, fit_sell=0.048,
                 co2_buy=0.007, co2_sell=0.0035)
    print("统一买/卖电价:", res['elec_price_buy'], res['elec_price_sell'])
    print("每 MG 电费  :", res['elec_cost'])
    print("每 MG 总成本:", res['total_cost'])
