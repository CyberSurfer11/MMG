import gym
from gym import spaces
import numpy as np
import pandas as pd

class CombinedEnergyEnv(gym.Env):
    """
    综合电-热能源强化学习环境，统一状态、动作及奖励约束计算。
    """
    def __init__(self):
        super().__init__()
        # ------------------ 共享参数 ------------------
        self.Hss = np.float32(3407.34); 
        self.Hhs = np.float32(3164.02)
        self.Hms = np.float32(2877.53); 
        self.Hls = np.float32(2742.5)
        self.Hsc = np.float32(2400.0); 
        self.Hbfw = np.float32(642.12)
        self.hs_cost = np.float32(0.0164); 
        self.ms_cost = np.float32(0.01495)
        self.ls_cost = np.float32(0.01161); 
        self.grid_cost = np.float32(0.0821)
        self.grid_co2 = np.float32(0.4019)

        # 电力系统参数
        self.Hgt = np.float32(10.0); 
        self.ngt = np.float32(0.65)
        self.ng_co2 = np.float32(0.48); 
        self.ng_cost = np.float32(0.484)
        self.Vcin, self.Vrat, self.Vcout, self.Grat = 3.0,12.0,25.0,1000.0
        self.xwt = 94
        self.Gst_user = np.array([1557,990,437,1001,308,51,41,78], dtype=np.float32)
        wind_df = pd.read_excel('data/env_data2.xlsx', engine='openpyxl')
        self.wind_speed_day = wind_df['wind'].values.astype(np.float32)

        # 热力系统参数
        self.LHV = np.float32(45200.0); 
        self.Fbmax = np.float32(150000.0)
        self.Fr = np.float32(0.5573); 
        self.effSHC = np.float32(0.84)
        self.Tfw, self.Tsat, self.Tls = np.float32(25.0), np.float32(143.61), np.float32(145.52)
        self.cpw, self.cpsat, self.rw = np.float32(4.1819), np.float32(2.3175), np.float32(2260.0)
        self.solar_area = np.float32(50000.0)
        self.fuel_cost = np.float32(0.21085)

        self.ghs = np.float32(0.1991)
        self.gms = np.float32(0.1811)
        self.gls = np.float32(0.1726)

        rad_df = pd.read_excel('data/env_data.xlsx', engine='openpyxl')
        self.rad_day = rad_df['salor'].values.astype(np.float32)

        # 时序与状态
        self.max_step = 24
        # 状态向量16维：
        # [0] M_gas_consumption, [1] Boiler fuel_consumption,
        # [2-7] Turbine power ST01-06,
        # [8] Wind turbine power, [9] Solar thermal output,
        # [10-15] 保留占位
        self.state = np.zeros(16, dtype=np.float32)
        self.time_step = 0

        # 动作向量27维：
        ele_low = np.zeros(2); ele_high = np.array([2000,2000], dtype=np.float32) # 燃气消耗量，上级电网交易
        th_cont_low = np.zeros(17, dtype=np.float32)
        th_cont_high = np.concatenate([
            [2e6,7e5,6e6,2e5],       # BF, HS_imp, MS_imp, LS_imp
            [2e6,6e5,2e6,1e6,6e5],    # ext ST01-05
            [5e5,5e5,6e5,6e5,1e6],    # cond ST01-04, ST06
            [1e5,5e4,5e4]            # LP release 1-3
        ]).astype(np.float32)
        th_disc_low = np.zeros(8); th_disc_high = np.ones(8)
        low = np.concatenate([ele_low, th_cont_low, th_disc_low])
        high = np.concatenate([ele_high, th_cont_high, th_disc_high])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 观测空间
        obs_low = np.zeros(16, dtype=np.float32)
        obs_high = np.concatenate([
            np.array([2000], dtype=np.float32),       # gas_cons upper
            np.array([1e7] + [1e8]*6, dtype=np.float32),# fuel and power upper
            np.zeros(6, dtype=np.float32)              # 占位
        ])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def reset(self):
        self.state.fill(0)
        self.time_step = 0
        return self.state.copy()

    def step(self, action):
        # 解析动作
        p_gas, G_imp = action[0:2]
        th_cont = action[2:19]
        th_disc = (action[19:27] > 0.5).astype(int)

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
        pen_e = self._constraint_e(wind_pow, G_imp, p_gas, P_ele)
        pen_h = self._constraint_h(th_cont, turb_pows, sol_ls)
        total_pen = self.penalty_weight_e*pen_e + self.penalty_weight_h*pen_h
        reward = -(total_cost + total_emis*0.01)*1e-3 - total_pen
        info = {'total_cost': total_cost,
                'total_emis': total_emis,
                'total_penalty': total_pen}

        # 步进
        self.time_step += 1
        done = self.time_step >= self.max_step
        return self.state.copy(), reward, done, info

    # ------------------ 内部计算方法 ------------------
    def _calc_boiler_fuel(self, m):
        """计算锅炉燃料消耗 (kg/h)"""
        nb = np.float32(0.97)
        return m * (self.Hss - self.Hbfw) / (self.LHV * nb)


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

    def _constraint_h(self, cont, turb_pows, steam_ext, sol_ls):
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



# 测试
if __name__ == '__main__':
    env = CombinedEnergyEnv()
    obs = env.reset()
    print('obs shape:', obs.shape)
    action = env.action_space.sample()
    obs2, r, d, info = env.step(action)
    print('reward:', r, 'info:', info)