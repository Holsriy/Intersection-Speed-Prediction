import learn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


def create_sequences(csv_path, seq_len=10, pred_len=5, save_path="sequence_data"):
    """
    生成模型训练所需的时序序列
    seq_len: 输入序列长度（历史帧数）
    pred_len: 预测序列长度（未来帧数）
    """
    # 加载数据
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=['track_id', 'frame'])

    # 特征标准化
    scaler = StandardScaler()
    features = ['center_x', 'center_y', 'heading_angle', 'speed']
    df[features] = scaler.fit_transform(df[features])
    joblib.dump(scaler, f"{save_path}/scaler.pkl")  # 保存标准化器

    # 方向编码（字符串→整数）
    dir_encoder = LabelEncoder()
    df['direction_encoded'] = dir_encoder.fit_transform(df['direction'])
    joblib.dump(dir_encoder, f"{save_path}/dir_encoder.pkl")

    # 按车辆ID分组生成序列
    X, y, directions = [], [], []
    for track_id, group in df.groupby('track_id'):
        group = group.reset_index(drop=True)
        total_frames = len(group)

        # 生成样本
        for i in range(total_frames - seq_len - pred_len + 1):
            # 输入特征：前seq_len帧的[位置x, 位置y, 航向角, 速度]
            input_seq = group.iloc[i:i + seq_len][features].values
            # 目标值：未来pred_len帧的速度
            target_seq = group.iloc[i + seq_len:i + seq_len + pred_len]['speed'].values
            # 方向标签（当前轨迹的方向）
            dir_label = group.iloc[i]['direction_encoded']

            X.append(input_seq)
            y.append(target_seq)
            directions.append(dir_label)

    # 转换为数组并保存
    X = np.array(X)  # 形状：(样本数, seq_len, 4)
    y = np.array(y)  # 形状：(样本数, pred_len)
    directions = np.array(directions)  # 形状：(样本数,)

    np.savez(f"{save_path}/sequences.npz", X=X, y=y, directions=directions)
    print(f"生成序列完成：{X.shape[0]}个样本，输入长度{seq_len}，预测长度{pred_len}")
    return X, y, directions


if __name__ == "__main__":
    import os

    os.makedirs("sequence_data", exist_ok=True)
    create_sequences(
        csv_path="vehicle_trajectory_data.csv",
        seq_len=10,  # 用前10帧（0.4秒）预测
        pred_len=5  # 预测未来5帧（0.2秒）
    )