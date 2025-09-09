import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# 配置参数
IMAGE_WIDTH = 1920  # 图像宽度
IMAGE_HEIGHT = 1080  # 图像高度
FPS = 25  # 视频帧率
VEHICLE_LENGTH = 4.5  # 小型汽车平均长度(米)
VEHICLE_WIDTH = 1.8  # 小型汽车平均宽度(米)

# 轨迹补全参数
MAX_FRAME_GAP = 2  # 最大允许帧差
MAX_POSITION_DISTANCE = 80  # 最大允许像素距离(约半个车身)
MAX_ANGLE_DIFFERENCE = 30  # 最大允许航向角差(度)
MIN_TRAJECTORY_LENGTH = 10  # 最短轨迹长度阈值


def calculate_center_coords(bbox):
    """计算旋转边界框的中心坐标"""
    x_coords = bbox[::2]  # 提取所有x坐标
    y_coords = bbox[1::2]  # 提取所有y坐标
    return np.mean(x_coords), np.mean(y_coords)


def calculate_heading_angle(bbox):
    """计算车辆航向角(度)"""
    x1, y1 = bbox[0], bbox[1]
    x2, y2 = bbox[2], bbox[3]
    dx = x2 - x1
    dy = y2 - y1
    angle = np.arctan2(dy, dx) * 180 / np.pi
    return angle if angle >= 0 else angle + 360


def calculate_vehicle_dimensions(bbox):
    """计算车辆在图像中的像素尺寸"""
    x_coords = bbox[::2]
    y_coords = bbox[1::2]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    return width, height


def determine_direction(angle_sequence):
    """根据航向角序列确定车辆行驶方向"""
    if len(angle_sequence) < 2:
        return "straight"

    start_angle = angle_sequence[0]
    end_angle = angle_sequence[-1]
    angle_diff = end_angle - start_angle

    # 标准化角度差(-180到180之间)
    if angle_diff > 180:
        angle_diff -= 360
    elif angle_diff < -180:
        angle_diff += 360

    if angle_diff > 10:  # 左转
        return "left"
    elif angle_diff < -10:  # 右转
        return "right"
    else:
        return "straight"  # 直行


def complete_trajectories(df):
    """轨迹补全算法：自动合并因ID切换导致的断裂轨迹"""
    # 按轨迹长度排序，优先处理较短的轨迹（更可能是断裂的）
    trajectory_lengths = df.groupby('track_id').size().reset_index(name='length')
    short_trajectories = trajectory_lengths[trajectory_lengths['length'] < MIN_TRAJECTORY_LENGTH * 2]
    short_ids = set(short_trajectories['track_id'])

    # 所有轨迹的ID列表
    all_ids = set(df['track_id'].unique())
    # 长轨迹ID（作为合并目标）
    long_ids = all_ids - short_ids

    # 创建轨迹信息字典：存储每个轨迹的首尾帧、位置和角度
    trajectory_info = {}
    for track_id in all_ids:
        traj = df[df['track_id'] == track_id].sort_values('frame')
        if len(traj) == 0:
            continue

        first_frame = traj['frame'].iloc[0]
        last_frame = traj['frame'].iloc[-1]
        first_x, first_y = traj['center_x'].iloc[0], traj['center_y'].iloc[0]
        last_x, last_y = traj['center_x'].iloc[-1], traj['center_y'].iloc[-1]
        first_angle = traj['heading_angle'].iloc[0]
        last_angle = traj['heading_angle'].iloc[-1]

        trajectory_info[track_id] = {
            'first_frame': first_frame,
            'last_frame': last_frame,
            'first_x': first_x,
            'first_y': first_y,
            'last_x': last_x,
            'last_y': last_y,
            'first_angle': first_angle,
            'last_angle': last_angle,
            'length': len(traj)
        }

    # 构建ID映射关系
    id_mapping = {}
    processed_short_ids = set()

    # 遍历短轨迹，尝试与其他轨迹合并
    for short_id in tqdm(short_ids, desc="补全断裂轨迹"):
        if short_id in processed_short_ids:
            continue

        short_info = trajectory_info.get(short_id)
        if not short_info:
            continue

        # 尝试将短轨迹合并到长轨迹
        best_match = None
        best_score = 0

        for target_id in all_ids:
            if target_id == short_id or target_id in processed_short_ids:
                continue

            target_info = trajectory_info.get(target_id)
            if not target_info:
                continue

            # 情况1：短轨迹在目标轨迹之后（短轨迹的首帧接近目标轨迹的末帧）
            frame_gap = short_info['first_frame'] - target_info['last_frame']
            if 1 <= frame_gap <= MAX_FRAME_GAP:
                # 计算位置距离
                pos_distance = np.sqrt(
                    (short_info['first_x'] - target_info['last_x']) ** 2 +
                    (short_info['first_y'] - target_info['last_y']) ** 2
                )

                # 计算角度差
                angle_diff = abs(short_info['first_angle'] - target_info['last_angle'])
                angle_diff = min(angle_diff, 360 - angle_diff)  # 取最小角度差

                # 满足条件
                if pos_distance <= MAX_POSITION_DISTANCE and angle_diff <= MAX_ANGLE_DIFFERENCE:
                    # 计算匹配分数（距离越小、角度差越小、帧差越小，分数越高）
                    score = (1 / (pos_distance + 1)) * (1 / (angle_diff + 1)) * (1 / frame_gap)
                    if score > best_score:
                        best_score = score
                        best_match = (target_id, 'after')

            # 情况2：短轨迹在目标轨迹之前（短轨迹的末帧接近目标轨迹的首帧）
            frame_gap = target_info['first_frame'] - short_info['last_frame']
            if 1 <= frame_gap <= MAX_FRAME_GAP:
                # 计算位置距离
                pos_distance = np.sqrt(
                    (target_info['first_x'] - short_info['last_x']) ** 2 +
                    (target_info['first_y'] - short_info['last_y']) ** 2
                )

                # 计算角度差
                angle_diff = abs(target_info['first_angle'] - short_info['last_angle'])
                angle_diff = min(angle_diff, 360 - angle_diff)  # 取最小角度差

                # 满足条件
                if pos_distance <= MAX_POSITION_DISTANCE and angle_diff <= MAX_ANGLE_DIFFERENCE:
                    # 计算匹配分数
                    score = (1 / (pos_distance + 1)) * (1 / (angle_diff + 1)) * (1 / frame_gap)
                    if score > best_score:
                        best_score = score
                        best_match = (target_id, 'before')

        # 如果找到最佳匹配，记录映射关系
        if best_match:
            target_id, position = best_match
            id_mapping[short_id] = target_id
            processed_short_ids.add(short_id)
            print(f"合并轨迹: ID {short_id} -> ID {target_id} ({position})")

    # 应用ID映射，合并轨迹
    df['track_id'] = df['track_id'].replace(id_mapping)

    # 对合并后的轨迹进行速度平滑处理
    smoothed_df = pd.DataFrame()
    for track_id in df['track_id'].unique():
        traj = df[df['track_id'] == track_id].sort_values('frame').copy()
        if len(traj) < MIN_TRAJECTORY_LENGTH:
            continue  # 过滤掉仍然过短的轨迹

        # 检测帧间隙（合并点）
        traj['frame_gap'] = traj['frame'].diff().fillna(1)
        gap_indices = traj[traj['frame_gap'] > 1].index

        # 平滑间隙处的速度
        for idx in gap_indices:
            # 确保索引有效
            if idx - 1 >= 0 and idx < len(traj):
                # 取前后速度的平均值
                prev_speed = traj.iloc[idx - 1]['speed']
                curr_speed = traj.iloc[idx]['speed']
                smoothed_speed = (prev_speed + curr_speed) / 2
                traj.at[idx, 'speed'] = smoothed_speed

        # 更新方向标签（基于完整轨迹重新计算）
        directions = []
        full_angle_sequence = traj['heading_angle'].values
        overall_direction = determine_direction(full_angle_sequence)

        # 确保方向只能是三类
        valid_directions = ['left', 'right', 'straight']
        if overall_direction not in valid_directions:
            overall_direction = 'straight'  # 异常值归为直行

        # 为轨迹中的所有点设置相同的方向
        traj['direction'] = overall_direction
        smoothed_df = pd.concat([smoothed_df, traj], ignore_index=True)

    print(f"轨迹补全完成: 原始轨迹数 {len(all_ids)}, 补全后轨迹数 {len(smoothed_df['track_id'].unique())}")
    return smoothed_df


def process_labels(labels_dir, output_file):
    """处理像素标签文件并生成轨迹和速度数据"""
    # 收集所有标签文件并按帧号排序
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    # 排序规则：提取文件名中"V_"后的数字作为帧号进行排序
    def extract_frame_number(filename):
        base_name = os.path.splitext(filename)[0]
        v_part = base_name.split("_V_")[-1]
        return int(v_part)

    # 按提取的帧号排序
    label_files.sort(key=extract_frame_number)

    # 存储所有车辆的轨迹数据
    vehicle_data = {}
    frame_number = 0

    # 处理每一帧
    for file in tqdm(label_files, desc="处理标签文件"):
        frame_path = os.path.join(labels_dir, file)

        with open(frame_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 11:  # 像素标签格式为11个字段
                continue

            # 解析数据
            class_id = int(parts[0])
            if class_id != 3:  # 只处理小型汽车(class_id=3)
                continue

            # 直接使用像素坐标
            pixel_bbox = list(map(float, parts[1:9]))
            track_id = int(parts[10])  # 跟踪ID

            # 计算中心坐标
            center_x, center_y = calculate_center_coords(pixel_bbox)

            # 计算航向角
            heading_angle = calculate_heading_angle(pixel_bbox)

            # 计算车辆像素尺寸
            pixel_width, pixel_height = calculate_vehicle_dimensions(pixel_bbox)

            # 存储数据
            if track_id not in vehicle_data:
                vehicle_data[track_id] = {
                    'frames': [],
                    'center_x': [],
                    'center_y': [],
                    'heading_angle': [],
                    'pixel_width': [],
                    'pixel_height': []
                }

            vehicle_data[track_id]['frames'].append(frame_number)
            vehicle_data[track_id]['center_x'].append(center_x)
            vehicle_data[track_id]['center_y'].append(center_y)
            vehicle_data[track_id]['heading_angle'].append(heading_angle)
            vehicle_data[track_id]['pixel_width'].append(pixel_width)
            vehicle_data[track_id]['pixel_height'].append(pixel_height)

        frame_number += 1

    # 计算速度并生成初始数据集
    all_data = []
    for track_id, data in vehicle_data.items():
        if len(data['frames']) < 5:  # 过滤极短轨迹
            continue

        # 计算速度
        for i in range(1, len(data['frames'])):
            time_diff = (data['frames'][i] - data['frames'][i - 1]) / FPS
            dx = data['center_x'][i] - data['center_x'][i - 1]
            dy = data['center_y'][i] - data['center_y'][i - 1]
            pixel_distance = np.sqrt(dx ** 2 + dy ** 2)

            # 像素到米的转换因子
            avg_pixel_length = np.mean([data['pixel_width'][i], data['pixel_width'][i - 1]])
            pixel_to_meter = VEHICLE_LENGTH / avg_pixel_length if avg_pixel_length > 0 else 0.00234375

            # 计算真实速度
            real_distance = pixel_distance * pixel_to_meter
            speed = (real_distance / time_diff) * 3.6 if time_diff > 0 else 0

            # 过滤不合理的速度值
            if 0 < speed < 120:
                all_data.append({
                    'track_id': track_id,
                    'frame': data['frames'][i],
                    'center_x': data['center_x'][i],
                    'center_y': data['center_y'][i],
                    'heading_angle': data['heading_angle'][i],
                    'speed': speed,
                    'direction': 'straight'  # 初始值设为直行
                })

    # 创建初始DataFrame
    initial_df = pd.DataFrame(all_data)

    # 如果没有数据，直接返回
    if initial_df.empty:
        print("没有有效的车辆轨迹数据")
        return initial_df

    # 补全轨迹
    completed_df = complete_trajectories(initial_df)

    # 保存最终结果
    completed_df.to_csv(output_file, index=False)
    print(f"数据预处理完成，共生成{len(completed_df)}条记录，已保存至{output_file}")
    return completed_df


if __name__ == "__main__":
    labels_directory = "dataset/data"
    output_csv = "vehicle_trajectory_data.csv"

    # 处理数据
    process_labels(labels_directory, output_csv)
