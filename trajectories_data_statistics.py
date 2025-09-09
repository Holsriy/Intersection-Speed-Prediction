import pandas as pd
df = pd.read_csv("vehicle_trajectory_data.csv")

# 查看基本信息
print(f"总记录数: {len(df)}")
print(f"车辆数量: {df['track_id'].nunique()}")
print(f"速度范围: {df['speed'].min()}~{df['speed'].max()} km/h")
print(f"方向分布:\n{df['direction'].value_counts()}")

# 查看是否有缺失值
print(f"缺失值情况:\n{df.isnull().sum()}")