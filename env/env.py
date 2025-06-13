# 导入必要的库
import gym
import numpy as np

# IES强化学习环境
class MicrogridEnv(gym.Env):
    def __init__(self, config, forecast_data, episode_length=24):
        super().__init__()
        self.config = config  # 每个MG的个性化配置，例如设备容量、效率等
        self.forecast_data = forecast_data  # 包含PV、风电、电负荷、热负荷等预测信息
        self.episode_length = episode_length

        self.t = 0
        self.SoC_elec = 0.5  # 初始电储能状态
        self.SoC_heat = 0.5  # 初始热储能状态

        # 当前时刻市场价格（由step中动态更新）
        self.elec_price_buy = 0.0
        self.elec_price_sell = 0.0
        self.carbon_price_buy = 0.0
        self.carbon_price_sell = 0.0

        # 动作空间：[a_EES, a_TES, a_CHP, a_GB]
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, 0, 0, ]),
            high=np.array([1, 1, 1, 1, ]),
            dtype=np.float32
        )

        # 状态空间：[pe+, pe-, pc+, pc-, P_L, Q_L, P_PV, P_WG, SoC_elec, SoC_heat]
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.float32
        )

    def reset(self):
        # 重置环境状态
        self.t = 0
        self.SoC_elec = 0.5
        self.SoC_heat = 0.5
        self.elec_price_buy = 0.0
        self.elec_price_sell = 0.0
        self.carbon_price_buy = 0.0
        self.carbon_price_sell = 0.0
        return self._get_state()

    def _get_state(self):
        # 获取当前状态
        return np.array([
            self.elec_price_buy,     # pe+
            self.elec_price_sell,    # pe-
            self.carbon_price_buy,   # pc+
            self.carbon_price_sell,  # pc-
            self.forecast_data['Load_elec'][self.t],      # P_L
            self.forecast_data['Load_heat'][self.t],      # Q_L
            self.forecast_data['PV'][self.t],             # P_PV
            self.forecast_data['WT'][self.t],             # P_WG
            self.SoC_elec,
            self.SoC_heat
        ], dtype=np.float32)

    def step(self, action, market_info=None, trade_result=None):
        """
        执行动作并更新环境状态

        参数:
        - action: 动作数组
        - market_info: 当前时刻市场价格，包括：
            "elec_price_buy", "elec_price_sell", "carbon_price_buy", "carbon_price_sell"
        - trade_result: 可选，电力和碳配额交易结果
        """

        # 动作解包，使用新配置格式及对应额定功率
        EES = action[0] * self.config['EES']['P_rated']
        TES = action[1] * self.config['TES']['P_rated']
        CHP = action[2] * self.config['CHP']['P_rated']
        GB  = action[3] * self.config['GB']['P_rated']

        Load_elec = self.forecast_data['Load_elec'][self.t]
        Load_heat = self.forecast_data['Load_heat'][self.t]
        PV = self.forecast_data['PV'][self.t]
        WT = self.forecast_data['WT'][self.t]

        # 存储前一个时刻的储能状态
        prev_soc_elec = self.SoC_elec
        prev_soc_heat = self.SoC_heat

        # 储能状态更新
        rho_elec = self.config['EES']['rho']
        eta_c_e = self.config['EES']['eta_c']
        eta_d_e = self.config['EES']['eta_d']
        delta_t = 1.0
        if EES > 0:
            self.SoC_elec = (1 - rho_elec) * self.SoC_elec - (EES * delta_t) / eta_d_e
        else:
            self.SoC_elec = (1 - rho_elec) * self.SoC_elec - eta_c_e * EES * delta_t
        self.SoC_elec = np.clip(self.SoC_elec, 0, self.config['EES']['SoC_max']) # 需要进一步修改

        rho_heat = self.config['TES']['rho']
        eta_c_h = self.config['TES']['eta_c']
        eta_d_h = self.config['TES']['eta_d']
        if TES > 0:
            self.SoC_heat = (1 - rho_heat) * self.SoC_heat - (TES * delta_t) / eta_d_h
        else:
            self.SoC_heat = (1 - rho_heat) * self.SoC_heat - eta_c_h * TES * delta_t
        self.SoC_heat = np.clip(self.SoC_heat, 0, self.config['TES']['SoC_max']) # 需要进一步修改

        # r_op
        CHP_cost = CHP * self.config['CHP']['cost']
        GB_cost  = GB * self.config['GB']['cost']
        EES_cost = abs((1-self.config['EES']['rho'])*prev_soc_elec-self.SoC_elec) * self.config['EES']['cost']
        TES_cost = abs((1-self.config['TES']['rho'])*prev_soc_heat-self.SoC_heat ) * self.config['TES']['cost']
        r_op = CHP_cost + GB_cost + EES_cost + TES_cost

        # 设备单位出力对应的气耗（单位 m³/kWh 或 m³/h）
        gas_CHP = CHP / self.config['CHP']['eta_elec']   # CHP 燃气消耗量
        gas_GB  = GB  / self.config['GB']['eta_heat']    # GB 燃气消耗量
        total_gas = gas_CHP + gas_GB
        gas_cost = self.config['as_price'] * total_gas

        # 热平衡
        Q_CHP = CHP * self.config['CHP']['eta_heat']
        heat_supply = Q_CHP + GB + TES
        heat_demand = Load_heat
        heat_imbalance = abs(heat_supply - heat_demand)

        # 电盈余
        P_n = Load_elec - EES - CHP -  PV

        # 碳盈余计算（碳配额买卖）
        # 获取碳排放因子
        ce_t = self.config['carbon_emission_factor_grid'][self.t]  # 电力碳排放因子
        cg_t = self.config['carbon_emission_factor_gas']  # gas碳排放因子

        # 计算碳盈余
        C_n = ce_t * P_t + cg_t * (Q_CHP + GB) - self.config['c_MG']

        # 新方式：调用市场定价函数（占位）
        self.elec_price_buy, self.elec_price_sell, self.carbon_price_buy, self.carbon_price_sell = get_market_prices()


        # 计算交易cost
        trade_cost = self.elec_price_buy*max(0,P_n) + self.carbon_price_buy*max(0,C_n) - self.elec_price_sell*min(0,P_n) - self.carbon_price_sell*min(0,C_n)

        reward = -(trade_cost + gas_cost) - heat_imbalance - r_op

        self.t += 1
        done = self.t >= self.episode_length
        return self._get_state(), reward, done, {}

    def render(self, mode='human'):
        print(f"时间: {self.t}, 电储能: {self.SoC_elec:.2f}, 热储能: {self.SoC_heat:.2f}")