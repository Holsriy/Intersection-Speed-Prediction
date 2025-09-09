import os
import re


def extract_last_number(filename):
    """从文件名（如"5-DJI_20230427114336_0002_V_1.txt"）中提取末尾数字，用于按顺序处理文件"""
    name_without_ext = os.path.splitext(filename)[0]
    match = re.search(r'_V_(\d+)', name_without_ext)
    return int(match.group(1)) if match else 0


def yolo_obb_norm2pixel(yolo_input_dir, pixel_output_dir, img_width=1920, img_height=1080):
    """
    将YOLO-OBB归一化标签转换为1920x1080像素坐标标签
    :param yolo_input_dir: YOLO-OBB归一化标签文件所在目录
    :param pixel_output_dir: 像素坐标标签文件输出目录
    :param img_width: 图像宽度（固定为1920）
    :param img_height: 图像高度（固定为1080）
    """
    # 1. 创建输出目录（若不存在）
    os.makedirs(pixel_output_dir, exist_ok=True)

    # 2. 获取所有YOLO-OBB标签文件，并按文件名末尾数字排序（如V_1→V_6853）
    yolo_files = [f for f in os.listdir(yolo_input_dir) if f.endswith('.txt')]
    if not yolo_files:
        print(f"警告：在输入目录 {yolo_input_dir} 中未找到任何.txt标签文件")
        return
    yolo_files.sort(key=extract_last_number)  # 按V_后的数字顺序处理
    print(f"找到 {len(yolo_files)} 个标签文件，开始转换...")

    # 3. 逐个处理标签文件
    for file_idx, yolo_filename in enumerate(yolo_files, 1):
        yolo_file_path = os.path.join(yolo_input_dir, yolo_filename)
        pixel_file_path = os.path.join(pixel_output_dir, yolo_filename)  # 输出文件名与输入一致

        try:
            # 读取YOLO-OBB归一化标签
            with open(yolo_file_path, 'r', encoding='utf-8') as yolo_f:
                lines = yolo_f.readlines()

            # 处理每一行标签（转换坐标）
            pixel_lines = []
            for line in lines:
                line = line.strip()
                if not line:  # 跳过空行
                    continue

                # 解析YOLO-OBB行：字段1=类别ID，字段2-9=8个归一化顶点坐标，字段10=置信度，字段11=跟踪ID
                parts = line.split()
                if len(parts) != 11:
                    print(f"跳过无效行（字段数≠11）：{line}（文件：{yolo_filename}）")
                    continue

                # 提取原始字段（确保类型正确）
                class_id = parts[0]  # 类别ID（字符串，保留原格式）
                norm_coords = parts[1:9]  # 8个归一化坐标（x1,y1,x2,y2,x3,y3,x4,y4）
                extra_field = parts[9]  # 第10个字段（置信度）
                track_id = parts[10]  # 跟踪ID（字符串，保留原格式）

                # 归一化坐标 → 像素坐标（公式：像素坐标 = 归一化坐标 × 图像尺寸）
                pixel_coords = []
                for i, norm_val in enumerate(norm_coords):
                    try:
                        norm_val = float(norm_val)
                        # 偶数索引（0,2,4,6）是x坐标 → 乘图像宽度；奇数索引（1,3,5,7）是y坐标 → 乘图像高度
                        if i % 2 == 0:  # x坐标（第1、3、5、7个坐标值）
                            pixel_x = round(norm_val * img_width, 2)  # 保留2位小数（可调整）
                            pixel_coords.append(str(pixel_x))
                        else:  # y坐标（第2、4、6、8个坐标值）
                            pixel_y = round(norm_val * img_height, 2)
                            pixel_coords.append(str(pixel_y))
                    except ValueError:
                        print(f"跳过无效坐标值：{norm_val}（行：{line}，文件：{yolo_filename}）")
                        break  # 坐标无效，跳过当前行
                else:
                    # 若所有坐标转换成功，重组为像素格式行（类别ID + 8个像素坐标 + 置信度 + 跟踪ID）
                    pixel_line = ' '.join([class_id] + pixel_coords + [extra_field, track_id])
                    pixel_lines.append(pixel_line)

            # 4. 保存像素坐标标签文件
            with open(pixel_file_path, 'w', encoding='utf-8') as pixel_f:
                pixel_f.write('\n'.join(pixel_lines))

            # 打印进度
            progress = (file_idx / len(yolo_files)) * 100
            print(f"[{progress:.1f}%] 已处理：{yolo_filename} → 保存至：{yolo_filename}")

        except Exception as e:
            print(f"处理文件 {yolo_filename} 时出错：{str(e)}，跳过该文件")

    print(f"\n转换完成！所有像素坐标标签已保存至：{pixel_output_dir}")


if __name__ == "__main__":
    # -------------------------- 请根据你的实际路径修改以下参数 --------------------------
    YOLO_OBB_INPUT_DIR = "track/labels"  # YOLO-OBB归一化标签目录
    PIXEL_OUTPUT_DIR = "dataset/data"  # 像素标签输出目录
    # -----------------------------------------------------------------------------------

    # 执行转换（固定图像尺寸为1920x1080）
    yolo_obb_norm2pixel(
        yolo_input_dir=YOLO_OBB_INPUT_DIR,
        pixel_output_dir=PIXEL_OUTPUT_DIR,
        img_width=1920,
        img_height=1080
    )