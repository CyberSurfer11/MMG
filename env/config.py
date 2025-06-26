# -*- coding: utf-8 -*-
"""
config.py: 配置综合电-热能源强化学习环境的共享参数与专属参数（无 __init__ 类式组织）
"""
import numpy as np
import pandas as pd

class Config:
    # ====== 共享参数 ======
    SHARED_PARAMS = {
        'Hss': np.float32(3407.34),
        'Hhs': np.float32(3164.02),
        'Hms': np.float32(2877.53),
        'Hls': np.float32(2742.5),
        'Hsc': np.float32(2400.0),
        'Hw' : np.float32(104.93),
        'Hbfw': np.float32(642.12),

        'hs_cost': np.float32(0.0164),
        'ms_cost': np.float32(0.01495),
        'ls_cost': np.float32(0.01161),
        'grid_cost': np.float32(0.0821),
        'grid_co2': np.float32(0.4019),

        'Hgt': np.float32(10.0),
        'ng_co2': np.float32(0.48),
        'ng_cost': np.float32(0.0672),

        'Vcin': np.float32(3.0),
        'Vrat': np.float32(12.0),
        'Vcout': np.float32(25.0),
        'Grat': np.float32(1000.0),
        'xwt': np.float32(94),

        'LHV': np.float32(45200.0),
        'Fr': np.float32(0.5573),
        'effSHC': np.float32(0.84),
        'Tfw': np.float32(25.0),
        'Tsat': np.float32(143.61),
        'Tls': np.float32(145.52),
        'cpw': np.float32(4.1819),
        'cpsat': np.float32(2.3175),
        'rw': np.float32(2260.0),

        'fuel_cost': np.float32(0.21085), # kg
        'gfuel' : np.float32(3.2233),
        'ghs': np.float32(0.1991),
        'gms': np.float32(0.1811),
        'gls': np.float32(0.1726),

        'max_step': 24,
        'penalty_weight_e': np.float32(1.0),
        'penalty_weight_h': np.float32(1.0)
    }

    # ====== 数据源配置 ======
    DATA_SOURCES = {
        'wind_speed': {'path': 'data/env_data2.xlsx', 'column': 'wind'},
        'solar_radiation': {'path': 'data/env_data.xlsx', 'column': 'salor'}
    }

    # ====== 场景专属参数 ======
    SCENARIO_PARAMS = {
        'IES1': {
            'Gst_user': np.array([1557, 990, 437, 1001, 308, 51, 41, 78], dtype=np.float32),
            'Mwhrs_ss': np.float32(394002.0),
            'base_r_ss': np.float32(72534.0),
            'base_r_hs': np.float32(89060.0),
            'base_r_ms': np.float32(154653.0),
            'base_r_ls': np.float32(95302.0),
            'clv': np.array([1.079, 1.103, 1.051], dtype=np.float32),
            'ngt': np.float32(0.65),
            'xwt': np.float32(94),
            'Fbmax': np.float32(150000.0),
            'nb' : np.float32(0.97),
            'solar_area': np.float32(50000.0),

            'action_bounds': {
                'ele_low': np.zeros(2, dtype=np.float32),
                'ele_high': np.array([2000, 2000], dtype=np.float32),
                'th_cont_low': np.zeros(17, dtype=np.float32),
                'th_cont_high': np.concatenate([
                    [2e6, 7e5, 6e6, 2e5],
                    [2e6, 6e5, 2e6, 1e6, 6e5],
                    [5e5, 5e5, 6e5, 6e5, 1e6],
                    [1e5, 5e4, 5e4]
                ]).astype(np.float32),
                # 'th_disc_low': np.zeros(8, dtype=np.float32),
                # 'th_disc_high': np.ones(8, dtype=np.float32)
            },
            'obs_bounds': {
                'low': np.zeros(14, dtype=np.float32),
                'high': np.concatenate([
                    np.array([2000], dtype=np.float32),
                    np.array([1e7] + [1e8]*6, dtype=np.float32),
                    np.ones(6, dtype=np.float32)
                ])
            }
        },
        'IES2': {
            'Gst_user': np.array([1557, 990, 437, 1001, 308, 51, 41, 78], dtype=np.float32),
            'Mwhrs_ss': np.float32(394002.0),
            'base_r_ss': np.float32(72534.0),
            'base_r_hs': np.float32(89060.0),
            'base_r_ms': np.float32(154653.0),
            'base_r_ls': np.float32(95302.0),
            'clv': np.array([1.079, 1.103, 1.051], dtype=np.float32),
            'ngt': np.float32(0.65),
            'xwt': np.float32(94),
            'Fbmax': np.float32(150000.0),
            'nb' : np.float32(0.97),
            'solar_area': np.float32(50000.0),

            'action_bounds': {
                'ele_low': np.zeros(2, dtype=np.float32),
                'ele_high': np.array([2000, 2000], dtype=np.float32),
                'th_cont_low': np.zeros(17, dtype=np.float32),
                'th_cont_high': np.concatenate([
                    [2e6, 7e5, 6e6, 2e5],
                    [2e6, 6e5, 2e6, 1e6, 6e5],
                    [5e5, 5e5, 6e5, 6e5, 1e6],
                    [1e5, 5e4, 5e4]
                ]).astype(np.float32),
                # 'th_disc_low': np.zeros(8, dtype=np.float32),
                # 'th_disc_high': np.ones(8, dtype=np.float32)
            },
            'obs_bounds': {
                'low': np.zeros(14, dtype=np.float32),
                'high': np.concatenate([
                    np.array([2000], dtype=np.float32),
                    np.array([1e7] + [1e8]*6, dtype=np.float32),
                    np.ones(6, dtype=np.float32)
                ])
            }
        },
        'IES3': {
            'Gst_user': np.array([1557, 990, 437, 1001, 308, 51, 41, 78], dtype=np.float32),
            'Mwhrs_ss': np.float32(394002.0),
            'base_r_ss': np.float32(72534.0),
            'base_r_hs': np.float32(89060.0),
            'base_r_ms': np.float32(154653.0),
            'base_r_ls': np.float32(95302.0),
            'clv': np.array([1.079, 1.103, 1.051], dtype=np.float32),
            'ngt': np.float32(0.65),
            'xwt': np.float32(94),
            'Fbmax': np.float32(150000.0),
            'nb' : np.float32(0.97),
            'solar_area': np.float32(50000.0),

            'action_bounds': {
                'ele_low': np.zeros(2, dtype=np.float32),
                'ele_high': np.array([2000, 2000], dtype=np.float32),
                'th_cont_low': np.zeros(17, dtype=np.float32),
                'th_cont_high': np.concatenate([
                    [2e6, 7e5, 6e6, 2e5],
                    [2e6, 6e5, 2e6, 1e6, 6e5],
                    [5e5, 5e5, 6e5, 6e5, 1e6],
                    [1e5, 5e4, 5e4]
                ]).astype(np.float32),
                # 'th_disc_low': np.zeros(8, dtype=np.float32),
                # 'th_disc_high': np.ones(8, dtype=np.float32)
            },
            'obs_bounds': {
                'low': np.zeros(14, dtype=np.float32),
                'high': np.concatenate([
                    np.array([3000], dtype=np.float32),
                    np.array([1e7] + [1e8]*6, dtype=np.float32),
                    np.ones(6, dtype=np.float32)
                ])
            }
        },        
    }

    @classmethod
    def get_shared(cls, key):
        return cls.SHARED_PARAMS[key]

    @classmethod
    def get_scenario(cls, scenario, key):
        return cls.SCENARIO_PARAMS[scenario][key]

    @classmethod
    def load_data(cls, name):
        cfg = cls.DATA_SOURCES[name]
        df = pd.read_excel(cfg['path'], engine='openpyxl')
        return df[cfg['column']].values.astype(np.float32)

    @classmethod
    def get_action_bounds(cls, scenario):
        ab = cls.SCENARIO_PARAMS[scenario]['action_bounds']
        low = np.concatenate([ab['ele_low'], ab['th_cont_low'], ab['th_disc_low']])
        high = np.concatenate([ab['ele_high'], ab['th_cont_high'], ab['th_disc_high']])
        return low, high

    @classmethod
    def get_obs_bounds(cls, scenario):
        ob = cls.SCENARIO_PARAMS[scenario]['obs_bounds']
        return ob['low'], ob['high']
