# Transformer 复现实验笔记

## 实验环境

- 硬件：单卡 RTX 3090 (24GB)
- 软件：PyTorch 2.0 + CUDA 11.8
- 数据集：WMT14 英德翻译子集（100 万句对）

## 复现配置

### 模型超参（复刻 Base 模型）
- 层数：6 层编码器 + 6 层解码器
- 模型维度 d_model = 512
- 多头数 h = 8，每头维度 d_k = 64
- 前馈网络维度 d_ff = 2048
- dropout = 0.1

### 训练超参
- 优化器：Adam（β1=0.9, β2=0.98, ε=1e-9）
- 学习率：warmup_steps=4000，按公式 lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
- 批次大小：4096 tokens
- 训练步数：100k 步

## 踩坑记录

### 坑1：学习率 warmup 容易写错

最初把 warmup 写成线性增长到 peak_lr，结果模型完全不收敛。仔细对照论文公式后才发现：warmup 期间 lr = d_model^(-0.5) * step * warmup^(-1.5)，warmup 之后 lr = d_model^(-0.5) * step^(-0.5)。峰值出现在 warmup_steps 处，约为 0.0007。

### 坑2：缩放因子漏了 sqrt(d_k)

实现缩放点积注意力时漏掉了除以 sqrt(d_k)。在 d_k=64 时虽然能训练但收敛很慢，BLEU 下降 3 个点。这是论文里强调的关键工程细节。

### 坑3：位置编码用错维度

位置编码的维度必须与 d_model 一致（512），最初误写成与序列长度一致导致维度不匹配。位置编码使用 sin/cos 交替：偶数维用 sin，奇数维用 cos。

## 复现结果

- 训练耗时：约 3 天（单卡）
- 英德翻译 BLEU：27.2（论文 Base 报告 27.3，基本对齐）
- 推理速度：每秒 1200 tokens

## 关键代码片段

```python
# 缩放点积注意力（注意 sqrt(d_k)）
scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
attn = F.softmax(scores, dim=-1)
output = torch.matmul(attn, value)
```

## 结论

复现结果与论文报告基本一致，偏差在 0.1 BLEU 以内。核心要点：
1. 学习率 warmup 公式必须严格按论文实现
2. 缩放因子 sqrt(d_k) 不可省略
3. 多头注意力用矩阵并行实现，避免循环
