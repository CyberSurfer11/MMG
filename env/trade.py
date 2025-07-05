# -*- coding: utf-8 -*-
"""
DA_market_algo1_mmr.py
--------------------------------------------------------
· 第 1 部分 —— Algorithm-1: CDA 逐单撮合（mid-price）
· 第 2 部分 —— 未撮合量与上级电网 ToU / FiT 结算
· 第 3 部分 —— 公式 (7)(8) 计算统一买/卖价  p_m^{+/-}
【重点】统一价只用于后续 reward / 结算，不参与撮合成交过程！
"""

from __future__ import annotations
from collections import defaultdict
from typing import List, Tuple, Dict
import numpy as np

# 电价 碳价
# ----------------------------------------------------------------------
def get_market_prices_car():
    tou_buy = (
        [0.058]*7 + [0.102] + [0.164]*3 + [0.102]*2 +
        [0.164]*3 + [0.102]*2 + [0.164]*4 + [0.058]*2
    )
    fit_sell = 0.048
    car_buy = 0.007
    car_sell = 0.0035

    grid_co2 = [0.60, 0.50, 0.60, 0.60, 0.70, 0.80, 0.80, 0.80,
                0.80, 0.80, 0.80, 0.90, 0.90, 0.90, 0.90, 1.10,
                1.10, 1.10, 0.90, 0.90, 0.80, 0.80, 0.80, 0.7]
                    
    return tou_buy, fit_sell, car_buy, car_sell,grid_co2



Bid = Tuple[str, float, float]    # (mg_id, bid_price,  qty>0)
Ask = Tuple[str, float, float]    # (mg_id, ask_price,  qty>0)

# ---------------------------------------------------
# 主函数
# ---------------------------------------------------
def da_market_clearing(
    bids : List[Bid],
    asks : List[Ask],
    lambda_buy : float,    # ToU 价  p_t^+
    lambda_sell: float     # FiT 价  p_t^-
) -> Tuple[
        Dict[str, Dict[str, float]],   # 各 MG 成交结果
        float, float,                  # 统一买 / 卖价  p_m^{+/-}
        Dict[str, float]               # 汇总信息
    ]:
    """
    Returns
    -------
    mg_res : {mg_id: {cda_qty, cda_cost, grid_qty, grid_cost}}
    p_m_buy / p_m_sell : 统一买价、统一卖价   —— 仅供奖励函数使用
    summary : 各类校验量  {D_plus, D_minus, D, platform_balance}
    """
    # ------------------------------------------------------------------
    # ❶ CDA 撮合（按价排序，成交价 = 中点）
    # ------------------------------------------------------------------
    buy_book  = sorted(bids,  key=lambda x: -x[1])   # 价高在前
    sell_book = sorted(asks, key=lambda x:  x[1])    # 价低在前

    res = defaultdict(lambda: dict(cda_qty  = 0.0, cda_cost = 0.0,
                                   grid_qty = 0.0, grid_cost= 0.0))

    b = s = 0
    while b < len(buy_book) and s < len(sell_book):
        bid_id,  p_bid, q_bid = buy_book [b]
        ask_id,  p_ask, q_ask = sell_book[s]

        if p_bid < p_ask:                 # 价格条件不满足
            break

        q = min(q_bid, q_ask)             # 成交量
        p_mid = 0.5 * (p_bid + p_ask)     # 中点价

        # 买家
        res[bid_id]['cda_qty']  +=  q
        res[bid_id]['cda_cost'] +=  p_mid * q
        # 卖家
        res[ask_id]['cda_qty']  -=  q
        res[ask_id]['cda_cost'] -=  p_mid * q

        # 更新订单剩余量
        buy_book [b] = (bid_id, p_bid, q_bid - q)
        sell_book[s] = (ask_id, p_ask, q_ask - q)
        if buy_book [b][2] == 0: b += 1
        if sell_book[s][2] == 0: s += 1

    # ------------------------------------------------------------------
    # ❷ 未撮合量直接与上级电网买/卖
    # ------------------------------------------------------------------
    # 剩余买单 → 向电网买
    for mg, _, qty in buy_book[b:]:
        res[mg]['grid_qty']  +=  qty
        res[mg]['grid_cost'] +=  qty * lambda_buy

    # 剩余卖单 → 向电网卖
    for mg, _, qty in sell_book[s:]:
        res[mg]['grid_qty']  -=  qty
        res[mg]['grid_cost'] -=  qty * lambda_sell

    # ------------------------------------------------------------------
    # ❸ 公式 (7)(8) 计算统一买/卖价——只用于奖励 / 成本结算
    # ------------------------------------------------------------------
    D_plus  = sum(q for _,_,q in bids )       # Σ buy qty
    D_minus = sum(q for _,_,q in asks )       # Σ sell qty
    D       = D_plus - D_minus
    p_mid   = 0.5 * (lambda_buy + lambda_sell)

    p_m_buy  = p_mid
    p_m_sell = p_mid
    if D >  1e-9:            # 整社区缺电：抬高买价
        p_m_buy  = (p_mid*D_minus + lambda_buy*D) / D_plus
    elif D < -1e-9:          # 过剩：压低卖价
        p_m_sell = (p_mid*D_plus  + lambda_sell*D) / D_minus

    # 平台收支校验：应该是0
    platform_balance = 0.0
    for v in res.values():
        platform_balance += v['cda_cost'] + v['grid_cost']
    summary = dict(D_plus=D_plus, D_minus=D_minus, D=D,
                   platform_balance=platform_balance)

    return res, p_m_buy, p_m_sell, summary


# ------------------------------------------------------------------
# demo 用例
# ------------------------------------------------------------------
if __name__ == "__main__":
    bids  = [(1, 0.11, 5.0),
             (2, 0.10, 3.0)]
    asks  = [(3, 0.19, 4.0),
             (4, 0.08, 2.0)]
    λ_buy  = 0.16
    λ_sell = 0.05

    mg_res, p_buy, p_sell, info = da_market_clearing(bids, asks, λ_buy, λ_sell)

    print(f"统一买价 p_m+ = {p_buy:.4f}  统一卖价 p_m- = {p_sell:.4f}")
    print("各 MG 成交结果：")
    for k,v in mg_res.items():
        print(f"  {k}: {v}")
    print("平台收支（应≈0）：", info['platform_balance'])

    '''
    统一买价 p_m+ = 0.1188  统一卖价 p_m- = 0.1050
    各 MG 成交结果：
    MG1: {'cda_qty': 5.0, 'cda_cost': 0.49000000000000005, 'grid_qty': 0.0, 'grid_cost': 0.0}
    MG4: {'cda_qty': -2.0, 'cda_cost': -0.19, 'grid_qty': 0.0, 'grid_cost': 0.0}
    MG3: {'cda_qty': -4.0, 'cda_cost': -0.395, 'grid_qty': 0.0, 'grid_cost': 0.0}
    MG2: {'cda_qty': 1.0, 'cda_cost': 0.095, 'grid_qty': 2.0, 'grid_cost': 0.32}
    平台收支（应≈0）： 0.32000000000000006
    '''

