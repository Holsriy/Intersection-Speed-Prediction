import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
from direction_aware_model import DirectionAwareTCN

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def load_data_and_model(sequence_path=r"D:\data\PycharmProjects\Curriculum Design 2\sequence_data\sequences.npz",
                        model_path=r"D:\data\PycharmProjects\Curriculum Design 2\models\best_model.pth",
                        scaler_path=r"D:\data\PycharmProjects\Curriculum Design 2\sequence_data\scaler.pkl",
                        encoder_path=r"D:\data\PycharmProjects\Curriculum Design 2\sequence_data\dir_encoder.pkl"):
    """加载评估所需的数据和模型"""
    # 加载序列数据
    data = np.load(sequence_path)
    X, y, directions = data['X'], data['y'], data['directions']

    # 加载标准化器和方向编码器
    scaler = joblib.load(scaler_path)
    dir_encoder = joblib.load(encoder_path)

    # 划分测试集（使用与训练时相同的随机状态）
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test, _, dir_test = train_test_split(
        X, y, directions, test_size=0.3, random_state=42
    )

    # 初始化模型并加载权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = X_test.shape[1]
    pred_len = y_test.shape[1]
    model = DirectionAwareTCN(seq_len=seq_len, pred_len=pred_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return {
        'X_test': X_test,
        'y_test': y_test,
        'dir_test': dir_test,
        'model': model,
        'scaler': scaler,
        'dir_encoder': dir_encoder,
        'device': device
    }


def get_predictions(eval_data):
    """获取模型预测结果"""
    X_test = eval_data['X_test']
    dir_test = eval_data['dir_test']
    model = eval_data['model']
    device = eval_data['device']

    # 转换为Tensor
    X_tensor = torch.FloatTensor(X_test).to(device)
    dir_tensor = torch.LongTensor(dir_test).to(device)

    # 预测
    with torch.no_grad():
        predictions = model(X_tensor, dir_tensor).cpu().numpy()

    return predictions


def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    # 展平数组（将时序维度合并）
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # 计算指标
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))

    return {
        'mae': mae,
        'rmse': rmse,
        'samples': len(y_true_flat)
    }


def evaluate_by_direction(eval_data, y_pred):
    """按三类行驶方向评估模型性能"""
    y_test = eval_data['y_test']
    dir_test = eval_data['dir_test']
    dir_encoder = eval_data['dir_encoder']

    # 解码方向标签
    direction_names = dir_encoder.classes_

    # 按方向分组计算指标
    metrics_by_dir = {}
    for dir_idx, dir_name in enumerate(direction_names):
        # 找到该方向的样本索引
        dir_mask = (dir_test == dir_idx)
        if np.sum(dir_mask) == 0:
            continue

        # 计算指标
        y_true_dir = y_test[dir_mask]
        y_pred_dir = y_pred[dir_mask]
        metrics = calculate_metrics(y_true_dir, y_pred_dir)

        metrics_by_dir[dir_name] = {
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'count': np.sum(dir_mask)
        }

    return metrics_by_dir


def inverse_transform_speed(scaler, speed_data):
    """将标准化的速度转换回原始尺度"""
    # 创建一个空数组，仅恢复速度特征（第3个索引）
    dummy = np.zeros((speed_data.shape[0], speed_data.shape[1], 4))
    dummy[:, :, 3] = speed_data  # 速度在特征中的索引是3
    dummy = scaler.inverse_transform(dummy.reshape(-1, 4)).reshape(dummy.shape)
    return dummy[:, :, 3]


def plot_prediction_examples(eval_data, y_pred, num_examples=3):
    """绘制预测示例对比图"""
    y_test = eval_data['y_test']
    dir_test = eval_data['dir_test']
    dir_encoder = eval_data['dir_encoder']
    scaler = eval_data['scaler']

    # 转换回原始速度尺度
    y_test_original = inverse_transform_speed(scaler, y_test)
    y_pred_original = inverse_transform_speed(scaler, y_pred)

    # 随机选择样本
    np.random.seed(42)
    sample_indices = np.random.choice(len(y_test), num_examples, replace=False)

    # 创建图表
    plt.figure(figsize=(12, 4 * num_examples))

    for i, idx in enumerate(sample_indices):
        # 获取样本数据
        true_speed = y_test_original[idx]
        pred_speed = y_pred_original[idx]
        dir_name = dir_encoder.inverse_transform([dir_test[idx]])[0]

        # 绘制子图
        plt.subplot(num_examples, 1, i + 1)
        plt.plot(true_speed, 'b-', label='真实速度')
        plt.plot(pred_speed, 'r--', label='预测速度')
        plt.title(f'预测示例 {i + 1}（方向：{dir_name}）')
        plt.xlabel('未来帧序号')
        plt.ylabel('速度(km/h)')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('prediction_examples.png', dpi=300)
    plt.close()
    print("预测示例图已保存为prediction_examples.png")


def plot_direction_metrics(metrics_by_dir):
    """绘制不同方向的指标对比图"""
    # 准备数据
    dir_names = list(metrics_by_dir.keys())
    maes = [metrics_by_dir[name]['mae'] for name in dir_names]
    rmses = [metrics_by_dir[name]['rmse'] for name in dir_names]
    counts = [metrics_by_dir[name]['count'] for name in dir_names]

    # 创建图表
    x = np.arange(len(dir_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制MAE和RMSE
    rects1 = ax1.bar(x - width / 2, maes, width, label='MAE')
    rects2 = ax1.bar(x + width / 2, rmses, width, label='RMSE')
    ax1.set_ylabel('误差(km/h)')
    ax1.set_title('不同行驶方向的预测误差')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dir_names)
    ax1.legend()

    # 绘制样本数量
    ax2 = ax1.twinx()
    ax2.plot(x, counts, 'ko-', label='样本数量')
    ax2.set_ylabel('样本数量')
    ax2.legend(loc='upper right')

    # 添加数值标签
    def add_bar_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.2f}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')

    add_bar_labels(rects1)
    add_bar_labels(rects2)

    plt.tight_layout()
    plt.savefig('direction_metrics.png', dpi=300)
    plt.close()
    print("方向误差对比图已保存为direction_metrics.png")


def plot_error_distribution(eval_data, y_pred):
    """绘制误差分布直方图"""
    y_test = eval_data['y_test']
    scaler = eval_data['scaler']

    # 转换回原始速度尺度
    y_test_original = inverse_transform_speed(scaler, y_test)
    y_pred_original = inverse_transform_speed(scaler, y_pred)

    # 计算误差
    errors = (y_pred_original - y_test_original).flatten()

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, bins=30)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'预测误差分布（均值：{errors.mean():.2f}, 标准差：{errors.std():.2f}）')
    plt.xlabel('预测误差(km/h)')
    plt.ylabel('频率')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=300)
    plt.close()
    print("误差分布图已保存为error_distribution.png")


def main():
    # 创建结果目录
    os.makedirs('evaluation_results', exist_ok=True)
    os.chdir('evaluation_results')  # 切换到结果目录

    # 加载数据和模型
    print("加载数据和模型...")
    eval_data = load_data_and_model()

    # 获取预测结果
    print("生成预测结果...")
    y_pred = get_predictions(eval_data)

    # 计算整体指标
    print("计算评估指标...")
    overall_metrics = calculate_metrics(eval_data['y_test'], y_pred)
    print(f"整体评估指标：")
    print(f"MAE (平均绝对误差): {overall_metrics['mae']:.4f} km/h")
    print(f"RMSE (均方根误差): {overall_metrics['rmse']:.4f} km/h")
    print(f"评估样本数: {overall_metrics['samples']}")

    # 按方向评估
    metrics_by_dir = evaluate_by_direction(eval_data, y_pred)
    print("\n按行驶方向评估：")
    for dir_name, metrics in metrics_by_dir.items():
        print(f"{dir_name} - MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, 样本数: {metrics['count']}")

    # 保存指标到文件
    with open('evaluation_metrics.txt', 'w') as f:
        f.write("整体评估指标：\n")
        f.write(f"MAE(平均绝对误差): {overall_metrics['mae']:.4f} km/h\n")
        f.write(f"RMSE(均方根误差): {overall_metrics['rmse']:.4f} km/h\n")
        f.write(f"评估样本数: {overall_metrics['samples']}\n\n")

        f.write("按行驶方向评估：\n")
        for dir_name, metrics in metrics_by_dir.items():
            f.write(
                f"{dir_name} - MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, 样本数: {metrics['count']}\n")

    # 可视化结果
    print("\n生成可视化结果...")
    plot_prediction_examples(eval_data, y_pred)
    plot_direction_metrics(metrics_by_dir)
    plot_error_distribution(eval_data, y_pred)

    print("\n评估完成！所有结果已保存至evaluation_results文件夹")


if __name__ == "__main__":
    main()
