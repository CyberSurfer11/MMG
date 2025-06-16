# 定义多个IES（MG）的成本与排放配置参数
MG_configs = {
    "MG1": {
        "CHP": {
            "type": "FC",
            "cost": 0.413,
            "emission": 0.25,
            "P_rated": 5000,
            "eta_elec": 0.45,
            "eta_heat": 0.43
        },
        "GB": {
            "cost": 0.275,
            "emission": 0.30,
            "P_rated": 3000,
            "eta_heat": 0.90
        },
        "EES": {
            "cost": 0.138,
            "rho": 0.05,
            "eta_c": 0.0095,
            "eta_d": 0.0095,
            "P_rated": 500,
            "SoC_max": 500
        },
        "TES": {
            "cost":0.138,
            "rho": 0.25,
            "eta_c": 0.0095,
            "eta_d": 0.0095,
            "P_rated": 500,
            "SoC_max": 500
        },
        "c_MG":300
    },
    "MG2": {
        "CHP": {
            "type": "MT",
            "cost": 0.413,
            "emission": 0.20,
            "P_rated": 6000,
            "eta_elec": 0.45,
            "eta_heat": 0.43
        },
        "GB": {
            "cost": 0.275,
            "emission": 0.25,
            "P_rated": 3000,
            "eta_heat": 0.90
        },
        "EES": {
            "cost": 0.138,
            "rho": 0.05,
            "eta_c": 0.0095,
            "eta_d": 0.0095,
            "P_rated": 500,
            "SoC_max": 500
        },
        "TES": {
            "cost":0.138,
            "rho": 0.25,
            "eta_c": 0.0095,
            "eta_d": 0.0095,
            "P_rated": 500,
            "SoC_max": 500
        },
        "c_MG":300
    },
    "MG3": {
        "CHP": {
            "type": "MT",
            "cost": 0.413,
            "emission": 0.30,
            "P_rated": 10000,
            "eta_elec": 0.45,
            "eta_heat": 0.43
        },
        "GB": {
            "cost": 0.275,
            "emission": 0.28,
            "P_rated": 5000,
            "eta_heat": 0.90
        },
        "EES": {
            "cost": 0.138,
            "rho": 0.05,
            "eta_c": 0.0095,
            "eta_d": 0.0095,
            "P_rated": 500,
            "SoC_max": 500
        },
        "TES": {
            "cost": 0.138,
            "rho": 0.25,
            "eta_c": 0.0095,
            "eta_d": 0.0095,
            "P_rated": 500,
            "SoC_max": 500
        },
        "c_MG":300
    },
    
    "elec_price_buy": [
        5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8,    # 0–6 Off-peak
        10.2,                                 # 7 Shoulder
        16.4, 16.4, 16.4,                     # 8–10 Peak
        10.2, 10.2,                           # 11–12 Shoulder
        16.4, 16.4, 16.4,                     # 13–15 Peak
        10.2, 10.2,                           # 16–17 Shoulder
        16.4, 16.4, 16.4, 16.4,               # 18–21 Peak
        5.8, 5.8                              # 22–23 Off-peak
    ],

    "carbon_emission_factor_grid" : [
    0.60, 0.50, 0.60, 0.60, 0.70, 0.80, 0.80, 0.80,
    0.80, 0.80, 0.80, 0.90, 0.90, 0.90, 0.90, 1.10,
    1.10, 1.10, 0.90, 0.90, 0.80, 0.80, 0.80, 0.7
    ],

    # 天然气参数（统一配置）
    "gas_price": 41.0,                    # ¢/m³
    "carbon_emission_factor_gas": 0.51,   # kg/m³

    # 上级市场碳税
    "carbon_price_buy": 0.7,   # ¢/kg
    "carbon_price_sell": 0.35 # ¢/kg
    
}