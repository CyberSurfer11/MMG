# -*- coding: utf-8 -*-
"""
CombinedEnergyEnv 环境定义，参数从 config.Config 动态加载
"""
import gym
from gym import spaces
import numpy as np
import pandas as pd
from env import Config,da_market_clearing

class CombinedEnergyEnv(gym.Env):
    """
    综合电-热能源强化学习环境，使用 config.Config 管理参数。
    """
    def __init__(self, scenario=None):
        super().__init__()
        # ------------------ 共享参数 ------------------
        self.Hss = Config.get_shared('Hss')
        self.Hhs = Config.get_shared('Hhs')
        self.Hms = Config.get_shared('Hms')
        self.Hls = Config.get_shared('Hls')
        self.Hsc = Config.get_shared('Hsc')
        self.Hbfw = Config.get_shared('Hbfw')
        self.hs_cost = Config.get_shared('hs_cost')
        self.ms_cost = Config.get_shared('ms_cost')
        self.ls_cost = Config.get_shared('ls_cost')
        self.grid_cost = Config.get_shared('grid_cost')
        self.grid_co2 = Config.get_shared('grid_co2')

        # 电力系统参数
        self.Hgt = Config.get_shared('Hgt')

        self.ng_co2 = Config.get_shared('ng_co2')
        self.ng_cost = Config.get_shared('ng_cost')
        self.Vcin = Config.get_shared('Vcin')
        self.Vrat = Config.get_shared('Vrat')
        self.Vcout = Config.get_shared('Vcout')
        self.Grat = Config.get_shared('Grat')
        self.xwt = Config.get_shared('xwt')

        # 场景专属参数
        self.Gst_user = Config.get_scenario(scenario, 'Gst_user')  # 在 config
        self.ngt = Config.get_scenario(scenario,'ngt')
        self.clv = Config.get_scenario(scenario,'clv')

        # 数据序列
        self.wind_speed_day = Config.load_data('wind_speed')      # 在 config

        # 热力系统参数
        self.LHV = Config.get_shared('LHV')
        self.Fbmax = Config.get_scenario(scenario, 'Fbmax')      # 在 config
        self.nb = Config.get_scenario(scenario, 'nb')  
        self.Fr = Config.get_shared('Fr')
        self.effSHC = Config.get_shared('effSHC')
        self.Tfw = Config.get_shared('Tfw')
        self.Tsat = Config.get_shared('Tsat')
        self.Tls = Config.get_shared('Tls')
        self.cpw = Config.get_shared('cpw')
        self.cpsat = Config.get_shared('cpsat')
        self.rw = Config.get_shared('rw')
        self.solar_area = Config.get_scenario(scenario, 'solar_area')  # 在 config
        self.fuel_cost = Config.get_shared('fuel_cost')

        # 蒸汽碳排因子
        self.ghs = Config.get_shared('ghs')
        self.gms = Config.get_shared('gms')
        self.gls = Config.get_shared('gls')

        # 蒸汽需求
        self.Mwhrs_ss = Config.get_scenario(scenario,'Mwhrs_ss')
        self.base_r_ss = Config.get_scenario(scenario,'base_r_ss')
        self.base_r_hs = Config.get_scenario(scenario,'base_r_hs')
        self.base_r_ms = Config.get_scenario(scenario,'base_r_ms')
        self.base_r_ls = Config.get_scenario(scenario,'base_r_ls')

        # 辐射序列
        self.rad_day = Config.load_data('solar_radiation')  # 在 config

        # 电需求
        self.elec_load = Config.load_elec_load(scenario)

        # 时序与状态
        self.max_step = Config.get_shared('max_step')
        self.time_step = 0

        # 动作向量27维：边界从 config
        ab = Config.get_scenario(scenario, 'action_bounds')
        ele_low, ele_high = ab['ele_low'], ab['ele_high']
        th_cont_low, th_cont_high = ab['th_cont_low'], ab['th_cont_high']
        low = np.concatenate([ele_low, th_cont_low])
        high = np.concatenate([ele_high, th_cont_high])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 观测空间
        ob = Config.get_scenario(scenario, 'obs_bounds')
        obs_low, obs_high = ob['low'], ob['high']
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)

    def reset(self):
        self.state.fill(0)
        self.time_step = 0
        return self.state.copy()

    def step(self, action):
        # print('action',action)
        # 解析动作
        p_gas, G_imp = action[0:2]
        th_cont = action[2:19]
        th_disc = np.array([1]*8)

        # ---- 热力系统计算 ----
        # 锅炉燃料消耗
        bf = th_cont[0]
        fuel_cons = self._calc_boiler_fuel(bf)
        # 太阳能产生低压蒸汽
        rad = self.rad_day[self.time_step]
        solar_out = self._calc_solar_output(rad)
        sol_ls = self._calc_solar_ls(solar_out)
        # 蒸汽涡轮机功率与蒸汽外送
        turb_pows, steam_ext = self._calc_turbine_power(th_cont, th_disc)
        # 更新热状态
        self.state[1] = fuel_cons
        self.state[2:8] = turb_pows[:6]
        self.state[9] = sol_ls

        # ---- 电力系统计算 ----
        wind_pow = self._calc_wind_power(self.wind_speed_day[self.time_step])
        gas_cons = self._calc_gas_consum(p_gas)
        # 用电需求由无蒸汽外送的电机驱动
        P_ele = np.where(np.array(steam_ext)==0, self.Gst_user, 0) #?
        self.state[0] = gas_cons
        self.state[8] = wind_pow

        # 电盈余
        P_n = self.elec_load[self.time_step] - (wind_pow + gas_cons)
        # 碳盈余
        C_n = self.grid_co2*G_imp + p_gas*self.ng_co2 +fuel_cons*self.grid_co2 + th_cont[1]*self.ghs + th_cont[2] * self.gms + th_cont[2]*self.gls - 8800


        # ---- 奖励与约束 ----
        total_cost = (
            self.grid_cost*G_imp + gas_cons*self.ng_cost +
            fuel_cons*self.fuel_cost + th_cont[1]*self.hs_cost +
            th_cont[2]*self.ms_cost + th_cont[3]*self.ls_cost
        )
        total_emis = (
            self.grid_co2*G_imp + p_gas*self.ng_co2 +
            fuel_cons*self.grid_co2 + th_cont[1]*self.ghs + th_cont[2] * self.gms + th_cont[2]*self.gls
        )

        # pen_e = self._constraint_e(wind_pow, G_imp, p_gas, P_ele)
        pen_h = self._constraint_h(th_cont, turb_pows, steam_ext, sol_ls)

        total_pen = pen_h
        reward = -(total_cost + total_emis*0.01)*1e-3



        info = {'total_cost': total_cost,
                'total_emis': total_emis,
                'total_penalty': total_pen,
                'P_n':P_n,
                'C_n':C_n
                }

        # 步进
        self.time_step += 1
        done = self.time_step >= self.max_step
        return self.state.copy(), reward, done, info

    # 计算交易之后的reward
    def compute_trade_cost(self,
                            p_n: float,
                            c_n: float,
                            elec_price_buy: float,
                            elec_price_sell: float,
                            carbon_price_buy: float,
                            carbon_price_sell: float
                        ) -> float:
        """
        计算 MG 在 P2P 市场的电＋碳交易净成本：
        cost = p^+·[P_n]^+ − p^−·[P_n]^−
            + p^{c,+}·[C_n]^+ − p^{c,−}·[C_n]^−
        其中 [x]^+ = max(x,0), [x]^− = −min(x,0)
        返回一个正数表示净支出，负数表示净收益。
        """

        self.state[8]  = elec_price_buy
        self.state[9]  = elec_price_sell
        self.state[10] = carbon_price_buy
        self.state[11] = carbon_price_sell

        # 电力交易成本
        cost_elec = (elec_price_buy  * max(p_n, 0) - elec_price_sell * min(p_n, 0))

        # 碳交易成本
        cost_carbon = (carbon_price_buy  * max(c_n, 0) - carbon_price_sell * min(c_n, 0))

        return cost_elec + cost_carbon,self.state.copy()


    # ------------------ 内部计算方法 ------------------
    def _calc_boiler_fuel(self, m):
        """计算锅炉燃料消耗 (kg/h)"""
        # nb = np.float32(0.97)
        return m * (self.Hss - self.Hbfw) / (self.LHV * self.nb)


    def _calc_wind_power(self, v):
        if v < self.Vcin:
            return 0.0
        elif self.Vcin <= v < self.Vrat:
            p = self.Grat * (v**3 - self.Vcin**3) / (self.Vrat**3 - self.Vcin**3)
        elif self.Vrat <= v < self.Vcout:
            p = self.Grat
        else:
            p = 0.0
        return p * self.xwt

    def _calc_gas_consum(self, p):
        return p / (self.ngt * self.Hgt)

    def _calc_solar_output(self, rad):
        return abs(self.solar_area * self.Fr * (self.effSHC * rad) * 3.6)

    def _calc_solar_ls(self, solar_out):
        return solar_out / (
            self.cpw*(self.Tsat-self.Tfw) +
            self.cpsat*(self.Tls-self.Tsat) + self.rw
        )

    def _calc_turbine_power(self, cont, disc):
        # 提取连续变量
        ext = cont[4:9]      # ST01-05 抽汽
        cond = cont[9:14]    # ST01-04, ST06 凝汽
        p = np.zeros(7, dtype=np.float32)
        # ST01-ST06
        p[0] = ((ext[0]+cond[0])*self.Hss - ext[0]*self.Hhs - cond[0]*self.Hsc) / 3600
        p[1] = ((ext[1]+cond[1])*self.Hss - ext[1]*self.Hms - cond[1]*self.Hsc) / 3600
        p[2] = ((ext[2]+cond[2])*self.Hss - ext[2]*self.Hhs - cond[2]*self.Hsc) / 3600
        p[3] = ((ext[3]+cond[3])*self.Hhs - ext[3]*self.Hms - cond[3]*self.Hsc) / 3600
        p[4] =  ext[4]*(self.Hhs - self.Hms) / 3600
        p[5] =  cond[4]*(self.Hhs - self.Hms) / 3600
        # ST07-ST14 固定蒸汽量
        fixed = [13294,8456,3731,8550,8220,1360,1084,2067]
        steam_ext = [fixed[i] if disc[i]==1 else 0 for i in range(8)]
        return np.round(p), steam_ext

    def _constraint_e(self, wind, G_imp, p_gas, P_ele):
        supply = round(wind) + round(G_imp) + round(p_gas)
        demand = np.sum(P_ele)
        gap = round(supply - demand)
        return abs(gap) if gap < 1 else 0

    def _constraint_h(self, cont, turb_pows, steam_ext, sol_ls=0):
        """
        四级蒸汽平衡约束，包含 3 路泄压阀：
          cont[0]   = M_bf_ss      锅炉高压蒸汽
          cont[1]   = M_HS_imp     HS 进口蒸汽
          cont[2]   = M_MS_imp     MS 进口蒸汽
          cont[3]   = M_LS_imp     LS 进口蒸汽
          cont[4:9] = M_ext_ST01-05  = 蒸汽机抽汽量 ST01–ST05 
          cont[9:14]= M_out_ST01-04,ST06 = 蒸汽机凝汽量 ST01–ST04,ST06
          cont[14:17]= M_lv01-03    = 三路泄压阀进气量
        steam_ext: 长度 8 的列表，ST07–ST14 抽汽量
        sol_ls:   太阳能产生的 LS 蒸汽（本函数不直接使用，但保留签名）
        """
        # 拆解 cont
        M_bf_ss, M_HS_imp, M_MS_imp, M_LS_imp = cont[0:4]
        M_ext      = cont[4:9]
        M_out      = cont[9:14]
        M_lv       = cont[14:17]

        # —— SS 级平衡 ——
        l_ss = self.Mwhrs_ss + M_bf_ss
        # print(M_bf_ss)
        r_ss = (
            (M_ext[0] + M_out[0]) +
            (M_ext[1] + M_out[1]) +
            (M_ext[2] + M_out[2]) +
            M_lv[0] +
            self.base_r_ss
        )
        gap_ss = round(l_ss - r_ss)
        pen_ss = abs(gap_ss) if abs(gap_ss) > 1 else 0

        # —— HS 级平衡 ——
        l_hs = (
            M_HS_imp +
            M_ext[0] +
            M_ext[2] +
            M_lv[0] * self.clv[0]
        )
        r_hs = (
            (M_ext[3] + M_out[3]) +
            M_ext[4] + M_out[4] +
            sum(steam_ext[0:4]) +
            M_lv[1] + 
            self.base_r_hs
        )
        gap_hs = round(l_hs - r_hs)
        pen_hs = abs(gap_hs) if abs(gap_hs) > 1 else 0

        # —— MS 级平衡 ——
        l_ms = (
            M_MS_imp +
            M_ext[1] +
            M_ext[3] +
            M_ext[4] +
            M_lv[1] * self.clv[1]
        )
        r_ms = (
            sum(steam_ext[4:]) +
            self.base_r_ms +
            M_lv[2]
        )
        gap_ms = round(l_ms - r_ms)
        pen_ms = abs(gap_ms) if abs(gap_ms) > 10 else 0

        # —— LS 级平衡 ——
        l_ls = sol_ls + M_LS_imp + sum(steam_ext) + M_lv[2] * self.clv[2]
        r_ls = self.base_r_ls
        gap_ls = round(l_ls - r_ls)
        pen_ls = abs(gap_ls) if abs(gap_ls) > 10 else 0

        # 返回各级罚项之和
        return pen_ss + pen_hs + pen_ms + pen_ls