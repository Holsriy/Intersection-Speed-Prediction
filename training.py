import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from direction_aware_model import DirectionAwareTCN

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 加载数据
data = np.load("sequence_data/sequences.npz")
X, y, directions = data['X'], data['y'], data['directions']

# 划分训练集和验证集（7:3）
X_train, X_val, y_train, y_val, dir_train, dir_val = train_test_split(
    X, y, directions, test_size=0.3, random_state=42
)

# 转换为Tensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train = torch.FloatTensor(X_train).to(device)
X_val = torch.FloatTensor(X_val).to(device)
y_train = torch.FloatTensor(y_train).to(device)
y_val = torch.FloatTensor(y_val).to(device)
dir_train = torch.LongTensor(dir_train).to(device)
dir_val = torch.LongTensor(dir_val).to(device)

# 初始化模型、损失函数和优化器
model = DirectionAwareTCN(
    seq_len=X_train.shape[1],
    pred_len=y_train.shape[1]
).to(device)
criterion = nn.MSELoss()  # 均方误差损失（适合回归任务）
optimizer = optim.Adam(model.parameters(), lr=2e-4)

# 训练参数
epochs = 30
batch_size = 16
best_val_loss = float('inf')
os.makedirs("models", exist_ok=True)

# 记录损失
train_losses = []
val_losses = []

# 训练循环
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    # 分批训练
    for i in tqdm(range(0, len(X_train), batch_size), desc=f"Epoch {epoch + 1}/{epochs}"):
        batch_X = X_train[i:i + batch_size]
        batch_y = y_train[i:i + batch_size]
        batch_dir = dir_train[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(batch_X, batch_dir)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_X.size(0)

    # 验证
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            batch_X = X_val[i:i + batch_size]
            batch_y = y_val[i:i + batch_size]
            batch_dir = dir_val[i:i + batch_size]

            outputs = model(batch_X, batch_dir)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)

    # 计算平均损失
    train_loss /= len(X_train)
    val_loss /= len(X_val)
    # 记录损失
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch + 1} | 训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f}")

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "models/best_model.pth")
        print(f"保存最佳模型（验证损失: {best_val_loss:.6f}）")

# 绘制并保存损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label='训练损失')
plt.plot(range(1, epochs + 1), val_losses, label='验证损失')
plt.xlabel('Epoch')
plt.ylabel('损失值')
plt.title('训练与验证损失曲线')
plt.legend()
plt.grid(True)
plt.savefig('models/loss_curve.png')  # 保存图像
plt.close()  # 关闭图像以释放资源

print("训练完成！最佳模型已保存至models/best_model.pth")
print("损失曲线已保存至models/loss_curve.png")