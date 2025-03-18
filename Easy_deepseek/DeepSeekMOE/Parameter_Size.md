## 一、Embedding层

- **Token Embedding**：
  - 张量大小：`[vocab_size, dim]`
  - 计算：$129280 \times 7168$

- **输出大小**：`[batch_size, seq_len, dim]`

---

## 二、Attention层（自注意力层）

- **Q (Query) 投影矩阵**：
  - 张量大小：`[dim, n_heads × qk_head_dim]`
  - 计算：$7168 \times (128 \times 128) = 7168 \times 16384$

- **K (Key) 投影矩阵**：
  - 张量大小：`[dim, n_heads × qk_head_dim]`
  - 同上：$7168 \times 16384$

- **V (Value) 投影矩阵**：
  - 张量大小：`[dim, n_heads × v_head_dim]`
  - 计算：$7168 \times (128 \times 128) = 7168 \times 16384$

- **输出投影矩阵 (O)**：
  - 张量大小：`[n_heads × v_head_dim, dim]`
  - 计算：$16384 \times 7168$

---

## 三、LoRA (低秩适配) 参数

- **Q LoRA**：
  - 张量大小：两个矩阵分别为 `[dim, q_lora_rank]` 和 `[q_lora_rank, n_heads × qk_head_dim]`
  - 计算：$7168 \times 1536$ 和 $1536 \times 16384$

- **KV LoRA**：
  - 张量大小：两个矩阵分别为 `[dim, kv_lora_rank]` 和 `[kv_lora_rank, n_heads × qk_head_dim]`
  - 计算：$7168 \times 512$ 和 $512 \times 16384$

---

## 四、MLP (Dense) 层

- **Dense MLP 层**（非MoE层，使用`inter_dim`）：
  - 第一层投影矩阵 (w1)：`[dim, inter_dim]` = $7168 \times 18432$
  - 第二层投影矩阵 (w2)：`[inter_dim, dim]` = $18432 \times 7168$
  - 第三层门控矩阵 (w3)：`[dim, inter_dim]` = $7168 \times 18432$

---

## 五、MoE (混合专家) 层

### (1) Gate 门控模块：

- **门控权重矩阵**：
  - 张量大小：`[n_routed_experts, dim]`
  - 计算：$256 \times 7168$

- **门控偏置向量**（仅当dim=7168时存在）：
  - 张量大小：`[n_routed_experts]`
  - 计算：$256$

### (2) 专家网络 (Expert)：

每个专家网络结构：

- 第一层投影矩阵 (w1)：`[dim, moe_inter_dim]` = $7168 \times 2048$
- 第二层投影矩阵 (w2)：`[moe_inter_dim, dim]` = $2048 \times 7168$
- 第三层门控矩阵 (w3)：`[dim, moe_inter_dim]` = $7168 \times 2048$

共有256个专家，每个专家参数大小相同。

### (3) 共享专家网络 (Shared Expert)：

- 第一层投影矩阵 (w1)：`[dim, moe_inter_dim × n_shared_experts]` = $7168 \times 2048$
- 第二层投影矩阵 (w2)：`[moe_inter_dim × n_shared_experts, dim]` = $2048 \times 7168$
- 第三层门控矩阵 (w3)：`[dim, moe_inter_dim × n_shared_experts]` = $7168 \times 2048$

---

## 六、LayerNorm 层

- 每个LayerNorm层参数：
  - 权重 (gamma)：`[dim]` = $7168$
  - 偏置 (beta)：`[dim]` = $7168$

---

## 七、输出层 (LM Head)

- **输出投影矩阵**：
  - 张量大小：`[dim, vocab_size]`
  - 计算：$7168 \times 129280$

---

## 八、整体模型结构

- 总层数：`n_layers = 61`
- 其中MoE层数：`n_layers - n_dense_layers = 61 - 3 = 58` 层
- Dense层数：`n_dense_layers = 3` 层

---

## 九、数据类型

- 参数数据类型：`fp8` (8-bit浮点数)，每个参数占用1字节。

---

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
