import pandas as pd
import numpy as np
import os
from glob import glob

def calculate_carbon_quota_split(Pe, gamma=1.0,
                                 epsilon_e=0.5, epsilon_h=0.22):
    """
    分别计算电负荷和热负荷对应的碳配额（单位：kgCO₂）
    """
    Pe = np.array(Pe)
    quota_electric = gamma * epsilon_e * Pe
    quota_heat = np.full_like(Pe, gamma * 37000)
    quota_total = quota_electric + quota_heat

    return {
        "quota_electric": quota_electric,
        "quota_heat": quota_heat,
        "quota_total": quota_total
    }

if __name__ == '__main__':
    folder = "./data/load_data/by_area_1h_cleaned"  # 路径按需修改
    files = sorted(glob(os.path.join(folder, "forecast_*.xlsx")))

    results = []

    for file in files:
        area_name = os.path.basename(file).replace("forecast_", "").replace(".xlsx", "")
        df = pd.read_excel(file, usecols=["TIME", "DEMAND_MW"])
        df["TIME"] = pd.to_datetime(df["TIME"])
        df["DATE"] = df["TIME"].dt.date

        # === 用你要求的方式计算每日平均电力负荷 ===
        num_days = df["DATE"].nunique()
        total_kwh = df["DEMAND_MW"].sum()
        avg_kwh_per_day = total_kwh / num_days

        # === 碳配额计算 ===
        quota = calculate_carbon_quota_split([avg_kwh_per_day], [88000])

        results.append({
            "Area": area_name,
            "Days": num_days,
            "Total_Electricity_mw": total_kwh,
            "Avg_Daily_mw": avg_kwh_per_day,
            "Quota_Electric_kgCO2": quota["quota_electric"][0],
            "Quota_Heat_kgCO2": quota["quota_heat"][0],
            "Quota_Total_kgCO2": quota["quota_total"][0]
        })

    # === 结果导出（可选）===
    df_quota = pd.DataFrame(results)
    df_quota.to_excel("carbon_quota_0.5.xlsx", index=False)
    print("✅ 每日碳配额已保存至 avg_daily_carbon_quota_by_area.xlsx")
