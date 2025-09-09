import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionAwareTCN(nn.Module):
    def __init__(self, input_dim=4, seq_len=10, pred_len=5,
                 hidden_dim=32, dir_emb_dim=3, num_directions=3):
        super().__init__()
        # 方向嵌入层：将方向标签转换为向量
        self.dir_embedding = nn.Embedding(num_directions, dir_emb_dim)

        # TCN层：提取时序特征
        self.tcn1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim // 2,
            kernel_size=3,
            padding=1
        )
        self.tcn2 = nn.Conv1d(
            in_channels=hidden_dim // 2 + dir_emb_dim,  # 融合方向特征
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )

        # 流向注意力层：不同方向对特征的关注度不同
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 输出层：预测未来速度
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len)
        )

    def forward(self, x, direction):
        """
        x: 输入特征 (batch_size, seq_len, input_dim)
        direction: 方向标签 (batch_size,)
        """
        batch_size, seq_len, _ = x.shape

        # 方向嵌入并扩展到序列长度
        dir_emb = self.dir_embedding(direction)  # (batch_size, dir_emb_dim)
        dir_emb = dir_emb.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, dir_emb_dim)

        # TCN需要通道优先，转换形状
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len)
        x = F.relu(self.tcn1(x))  # (batch_size, hidden_dim//2, seq_len)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, hidden_dim//2)

        # 融合方向特征
        x = torch.cat([x, dir_emb], dim=2)  # (batch_size, seq_len, hidden_dim//2 + dir_emb_dim)

        # 第二层TCN
        x = x.permute(0, 2, 1)  # (batch_size, hidden_dim//2 + dir_emb_dim, seq_len)
        x = F.relu(self.tcn2(x))  # (batch_size, hidden_dim, seq_len)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, hidden_dim)

        # 流向注意力：对序列中每个时间步加权
        att_weights = F.softmax(self.attention(x), dim=1)  # (batch_size, seq_len, 1)
        x = x * att_weights  # (batch_size, seq_len, hidden_dim)

        # 展平特征并预测
        x = x.reshape(batch_size, -1)  # (batch_size, seq_len * hidden_dim)
        out = self.fc(x)  # (batch_size, pred_len)

        return out


# 测试模型输出形状
if __name__ == "__main__":
    model = DirectionAwareTCN(seq_len=10, pred_len=5)
    x = torch.randn(32, 10, 4)  # 32个样本，10帧序列，4个特征
    direction = torch.randint(0, 3, (32,))  # 32个方向标签（0-2）
    output = model(x, direction)
    print(f"模型输出形状：{output.shape}（预期：(32, 5)）")