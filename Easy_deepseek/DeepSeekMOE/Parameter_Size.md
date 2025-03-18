# DeepSeek-671B配置参数：

```json
{
    "vocab_size": 129280,
    "dim": 7168,
    "inter_dim": 18432,
    "moe_inter_dim": 2048,
    "n_layers": 61,
    "n_dense_layers": 3,
    "n_heads": 128,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "n_activated_experts": 8,
    "n_expert_groups": 8,
    "n_limited_groups": 4,
    "route_scale": 2.5,
    "score_func": "sigmoid",
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "dtype": "fp8"
}
```


## 参数规模总结（每个模块）：

| 模块 | 参数张量大小 |
|------|-------------|
| Embedding | $129280 \times 7168$ |
| Attention QKV | $3 \times (7168 \times 16384)$ |
| Attention O | $16384 \times 7168$ |
| LoRA Q | $7168 \times 1536$, $1536 \times 16384$ |
| LoRA KV | $7168 \times 512$, $512 \times 16384$ |
| Dense MLP (每层) | $7168 \times 18432$, $18432 \times 7168$, $7168 \times 18432$ |
| MoE Gate (每MoE层) | $256 \times 7168$ |
| MoE Experts (每个专家) | $7168 \times 2048$, $2048 \times 7168$, $7168 \times 2048$ |
| MoE Shared Expert | $7168 \times 2048$, $2048 \times 7168$, $7168 \times 2048$ |
| LayerNorm (每层) | $7168 \times 2$ |
| LM Head | $7168 \times 129280$ |

---


# MoE门控路由详细示例（一个更加简单的例子）

基于一组更加简单的配置参数：`dim=16, n_experts=32, n_groups=8, topk_groups=2, topk=2`，展示具体的示例，详细追踪每一步的计算过程和具体数值变化。

## 一、Gate类的处理流程

### 1. 初始化阶段
```python
# Gate.__init__调用
self.dim = 16                     # 输入特征维度
self.topk = 2                     # 每个token激活的专家数
self.n_groups = 8                 # 专家分组数量
self.topk_groups = 2              # 每个token激活的组数量
self.score_func = "sigmoid"       # 评分函数类型
self.route_scale = 2.5            # 路由权重的缩放因子

# 初始化参数
self.weight = nn.Parameter(torch.randn(32, 16) * 0.1)  # [n_experts, dim]
# 例如权重矩阵为:
# [[ 0.02,  0.01, -0.03, ...,  0.04]
#  [-0.01,  0.03,  0.02, ..., -0.02]
#  ... 
#  [ 0.04, -0.02,  0.03, ...,  0.01]] 

self.bias = None  # 本例不使用偏置
```

### 2. 前向传播示例

```python
# 输入示例：单个token
x = torch.tensor([[0.5, 0.2, 0.3, -0.1, 0.7, -0.2, 0.4, 0.1, 
                  -0.3, 0.6, -0.4, 0.8, 0.2, -0.5, 0.3, 0.9]])
```

#### 步骤1: 计算亲和度分数
```python
# 执行 scores = Linear(x, self.weight)
scores = x @ self.weight.T  # [1, 16] @ [16, 32] -> [1, 32]

# 计算结果可能为:
scores = tensor([[ 0.21,  0.15,  0.18,  0.27, -0.12,  0.31,  0.24,  0.19,
                   0.23,  0.28,  0.34,  0.17, -0.08,  0.22,  0.29,  0.20,
                   0.16,  0.25,  0.32,  0.14,  0.26,  0.33,  0.18,  0.21,
                   0.30,  0.13,  0.22,  0.19,  0.27,  0.16,  0.35,  0.24]])
```

#### 步骤2: 转换为概率值
```python
# 执行sigmoid激活
scores = scores.sigmoid()
original_scores = scores.clone()  # 保存原始分数副本

scores = tensor([[ 0.55,  0.54,  0.54,  0.57,  0.47,  0.58,  0.56,  0.55,
                   0.56,  0.57,  0.58,  0.54,  0.48,  0.55,  0.57,  0.55,
                   0.54,  0.56,  0.58,  0.53,  0.56,  0.58,  0.54,  0.55,
                   0.57,  0.53,  0.55,  0.55,  0.57,  0.54,  0.59,  0.56]])
```

#### 步骤3: 分组重塑
```python
# 重塑为组结构 - 每组4个专家
scores = scores.view(1, 8, 4)  # [1, n_groups, experts_per_group]

scores = tensor([[[ 0.55,  0.54,  0.54,  0.57],   # 组0
                  [ 0.47,  0.58,  0.56,  0.55],   # 组1
                  [ 0.56,  0.57,  0.58,  0.54],   # 组2
                  [ 0.48,  0.55,  0.57,  0.55],   # 组3
                  [ 0.54,  0.56,  0.58,  0.53],   # 组4
                  [ 0.56,  0.58,  0.54,  0.55],   # 组5
                  [ 0.57,  0.53,  0.55,  0.55],   # 组6
                  [ 0.57,  0.54,  0.59,  0.56]]]) # 组7
```

#### 步骤4: 计算组得分
```python
# 获取每组最大值作为组得分
group_scores = scores.amax(dim=-1)

group_scores = tensor([[ 0.57, 0.58, 0.58, 0.57, 0.58, 0.58, 0.57, 0.59]])
```

#### 步骤5: 选择最佳组
```python
# 选择得分最高的两个组
_, indices = group_scores.topk(2, dim=-1)

# 假设选中组7和组2
indices = tensor([[ 7, 2]])
```

#### 步骤6: 创建掩码
```python
# 创建掩码，选中组为False, 未选中组为True
mask = torch.ones(1, 8, dtype=bool)
mask.scatter_(1, indices, False)

mask = tensor([[ True,  True, False,  True,  True,  True,  True, False]])
```

#### 步骤7: 应用组掩码
```python
# 将未选中组的专家得分设为负无穷
scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf"))

scores = tensor([[[-inf, -inf, -inf, -inf],    # 组0 (未选中)
                  [-inf, -inf, -inf, -inf],    # 组1 (未选中)
                  [0.56, 0.57, 0.58, 0.54],    # 组2 (选中)
                  [-inf, -inf, -inf, -inf],    # 组3 (未选中)
                  [-inf, -inf, -inf, -inf],    # 组4 (未选中)
                  [-inf, -inf, -inf, -inf],    # 组5 (未选中)
                  [-inf, -inf, -inf, -inf],    # 组6 (未选中)
                  [0.57, 0.54, 0.59, 0.56]]])  # 组7 (选中)
```

#### 步骤8: 展平回专家视图
```python
# 恢复专家视图
scores = scores.flatten(1)

scores = tensor([[-inf, -inf, -inf, -inf,    # 组0的4个专家
                  -inf, -inf, -inf, -inf,    # 组1的4个专家
                  0.56, 0.57, 0.58, 0.54,    # 组2的4个专家
                  -inf, -inf, -inf, -inf,    # 组3的4个专家
                  -inf, -inf, -inf, -inf,    # 组4的4个专家
                  -inf, -inf, -inf, -inf,    # 组5的4个专家
                  -inf, -inf, -inf, -inf,    # 组6的4个专家
                  0.57, 0.54, 0.59, 0.56]])  # 组7的4个专家
```

#### 步骤9: 选择顶部专家
```python
# 选择得分最高的2个专家
_, indices = torch.topk(scores, 2, dim=-1)

# 结果：选择了专家30和专家10
indices = tensor([[ 30, 10 ]])
```

#### 步骤10: 获取专家权重
```python
# 从原始分数中提取选中专家的分数
weights = original_scores.gather(1, indices)
weights = tensor([[ 0.59, 0.58 ]])

# 归一化权重
weights /= weights.sum(dim=-1, keepdim=True)
weights = tensor([[ 0.504, 0.496 ]])

# 应用路由缩放
weights *= self.route_scale
weights = tensor([[ 1.26, 1.24 ]])
```

## 二、MoE类的处理流程

### 1. 初始化关键参数
```python
# MoE.__init__调用
self.dim = 16                     # 输入特征维度
self.n_routed_experts = 32        # 路由专家总数
self.n_activated_experts = 2      # 每个token激活的专家数
self.gate = Gate(args)            # 初始化门控机制
self.experts = nn.ModuleList()    # 初始化32个Expert实例
self.shared_experts = MLP(16, 64) # 共享专家网络
```

### 2. 前向传播处理示例

```python
# 输入张量 - 单个token示例
x = torch.tensor([[[0.5, 0.2, 0.3, -0.1, 0.7, -0.2, 0.4, 0.1, 
                   -0.3, 0.6, -0.4, 0.8, 0.2, -0.5, 0.3, 0.9]]])
shape = x.shape  # 保存原始形状 [1, 1, 16]
x = x.view(-1, self.dim)  # 重塑为 [1, 16]
```

#### 步骤1: 获取路由信息
```python
# 调用门控机制决定路由
weights, indices = self.gate(x)
# weights = [[ 1.26, 1.24 ]]
# indices = [[ 30, 10 ]]
```

#### 步骤2: 初始化输出
```python
# 创建空输出张量
y = torch.zeros_like(x)  # [1, 16] 全零张量
```

#### 步骤3: 统计专家负载
```python
# 统计每个专家选中次数
counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)

counts = [0, 0, ..., 1, ..., 1, ...] # 第10位和第30位为1，其他为0
```

#### 步骤4: 分发计算 - 专家10
```python
# 处理专家10
i = 10
idx, top = torch.where(indices == i)  # 找到选择专家10的tokens
# idx = [0], top = [1]  (索引0的token通过其第1个槽位选择了专家10)

expert = self.experts[10]
expert_out = expert(x[idx])  
# 假设输出为: [0.3, -0.2, 0.5, 0.1, -0.4, 0.6, 0.2, -0.3, 0.7, 0.4, -0.1, 0.8, 0.3, -0.5, 0.9, 0.2]

# 应用权重
weight = weights[idx, top, None]  # 权重值: 1.24
weighted_out = expert_out * weight
# weighted_out = [0.372, -0.248, 0.620, 0.124, -0.496, 0.744, 0.248, -0.372, 
#                 0.868, 0.496, -0.124, 0.992, 0.372, -0.620, 1.116, 0.248]

# 累加到输出
y[idx] += weighted_out  # 更新输出张量对应位置
```

#### 步骤5: 分发计算 - 专家30
```python
# 处理专家30
i = 30
idx, top = torch.where(indices == i)  # 找到选择专家30的tokens
# idx = [0], top = [0]  (索引0的token通过其第0个槽位选择了专家30)

expert = self.experts[30]
expert_out = expert(x[idx])  
# 假设输出为: [0.4, 0.1, -0.3, 0.7, 0.2, -0.4, 0.8, 0.3, -0.2, 0.6, 0.5, -0.1, 0.9, 0.4, -0.6, 0.7]

# 应用权重
weight = weights[idx, top, None]  # 权重值: 1.26
weighted_out = expert_out * weight
# weighted_out = [0.504, 0.126, -0.378, 0.882, 0.252, -0.504, 1.008, 0.378,
#                -0.252, 0.756, 0.630, -0.126, 1.134, 0.504, -0.756, 0.882]

# 累加到输出
y[idx] += weighted_out  # 更新输出张量对应位置
```

#### 步骤6: 共享专家计算
```python
# 所有token通过共享专家
z = self.shared_experts(x)

# 假设共享专家输出为:
z = [0.2, -0.1, 0.3, -0.2, 0.4, -0.3, 0.5, -0.4, 0.6, -0.5, 0.7, -0.6, 0.8, -0.7, 0.9, -0.8]
```

#### 步骤7: 合并结果
```python
# 组合路由专家和共享专家的结果
output = y + z

# 合并后结果:
output = [1.076, -0.222, 0.542, 0.806, 0.156, -0.060, 1.756, 0.308, 
          1.216, 0.746, 1.206, 0.266, 2.306, -0.816, 1.260, 0.330]

# 恢复原始形状
output = output.view(shape)  # 形状: [1, 1, 16]
```

## 结论

这个详细的跟踪展示了混合专家模型中的精确计算流程：

1. **门控机制**：首先计算亲和度分数，选择最相关的组，然后从这些组中选择最佳专家
2. **稀疏专家计算**：每个token只激活2个专家(本例为专家10和专家30)，并按权重加权其输出
3. **共享基础处理**：所有token都通过共享专家，确保获得基础处理
4. **动态路由**：整个过程中，token根据内容找到最匹配的计算路径

这种机制显著提高了模型容量，同时保持了计算效率，对于扩展大规模语言模型至关重要。
