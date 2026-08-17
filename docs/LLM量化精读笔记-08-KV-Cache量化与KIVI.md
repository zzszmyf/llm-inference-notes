# LLM 量化精读笔记 · 08 KV Cache 量化：KIVI 与误差累积

> 对应：KIVI（arXiv:2402.02750，ICML 2024）；Inference Engineering Ch5 的敏感性排序（$KV cache =$中等敏感、误差逐 token 累积）；vLLM PagedAttention 背景。
> 学完本章你应该能：① 手算任意模型的 KV cache 显存公式与实例；② 解释"KV 误差为什么滚雪球"；③ 复述 KIVI 的两个核心洞察（K 按通道、V 按 token）及其理由；④ 说出 KIVI 的实现要点（非对称 2-bit、流式量化、FP16 缓冲）与结果（2-bit 近无损、4x 省显存）。

---

## 目录（本章）

1. 本章目标
2. 为什么 KV cache 是下一个量化对象
3. KV 误差为什么"滚雪球"
4. 设计空间：量化什么、用什么粒度
5. 分布分析：K 稳定、V 多变
6. KIVI 算法
7. 结果与边界
8. 与 KV 优化的其他手段的关系
9. 本章小结
10. 习题与解答
11. 延伸阅读

---

## 1. 本章目标

至此我们已经量化了权重（05/06）和激活（07）。第三个量化对象是 **KV cache**——它是长上下文推理里最大的显存消耗者，也是"错误会跨 token 累积"的独特存在。Inference Engineering 把它排在敏感性排序的中间：**中等敏感，但必须谨慎**。

本章以 KIVI 为主线，回答三个问题：KV 为什么值得量化、KV 量化为什么难、KIVI 怎么在 2-bit 做到近无损。

---

## 2. 为什么 KV cache 是下一个量化对象

### 2.1 KV cache 是什么

解码时，每个 token 的注意力 Key/Value 向量被缓存，供后续所有 token 复用：

KV cache 张量：K、V 各一个，形状$[n_{\text{layers}}, n_{kv_heads}, \text{seq\_len}, \text{head\_dim}]$

显存公式（FP16）：

$$
Bytes = 2 \times n_{\text{layers}} \times n_{kv_heads} \times \text{head\_dim} \times \text{seq\_len} \times 2
$$

### 2.2 实例：Llama-2-70B

$$
n_{\text{layers}} = 80，n_{kv_heads} = 8（GQA），\text{head\_dim} = 128，FP16
$$

每 token：$2 \times 80 \times 8 \times 128 \times 2 B = 327,680 B \approx 0.31 MB$

32K 上下文$\to \approx 10.7 GB$
128K 上下文$\to \approx 43 GB$

对比：70B 权重$FP16 = 140 GB$；$FP8 = 70 GB$

结论：**上下文越长，KV 越接近甚至超过权重成为显存大头**；而且 decode 每一步都要读全量 KV，带宽消耗同样可观。

### 2.3 两个收益

省显存$\to$同样的 GPU 能塞更大 batch / 更长上下文
省带宽$\to decode$读 KV 的时间减少（和权重带宽同源）

---

## 3. KV 误差为什么"滚雪球"

权重量化误差是**静态**的：量化一次，误差固定，且可以被校准补偿（GPTQ/AWQ）。

KV 误差是**动态累积**的：

1. token t 写入时被量化$\to$带误差的$K_{t}$、$V_{t}$进入 cache
2. 之后每个 token（t+1, t+2, …）都要读$K_{t}$、$V_{t}$算注意力
3. 误差不消失，还参与所有后续 softmax/输出计算
4. 每一层、每一步都在重复使用这些被污染的缓存$\to$误差逐 token 累积

更微妙的是 softmax：QK 分数一旦被量化噪声扰动，softmax 的指数放大会把"小扰动"变成"注意力权重偏差"，进而直接改变输出分布。这就是 Inference Engineering 说"errors compound token-to-token"的机理。

---

## 4. 设计空间：量化什么、用什么粒度

KV 量化有四个自由参数：

| 维度 | 选项 | 直觉 |
|---|---|---|
| 量化对象 | K 或 V 或两者 | K 影响 softmax 分数，V 直接进输出 |
| 粒度 | per-token / per-channel / per-tensor | 与 04 章同构 |
| 位宽 | 2/4/8-bit | 每 bit 省 2 倍显存 |
| 对称性 | 对称 / 非对称 | 分布是否有偏 |

朴素方案（per-tensor 4-bit）为什么不行：KV 的通道/时间分布差异大，per-tensor 范围利用率低（04 章），且 V 的分布随时间漂移。

---

## 5. 分布分析：K 稳定、V 多变

KIVI 论文先做了分布统计（这是它最大的贡献）：

### 5.1 Key 的分布：跨 token 稳定、按通道有规律

Key 张量：$shape [tokens, heads, \text{head\_dim}]$

发现 1：同一通道（dim 维度）的 key 值跨 token 分布稳定（模式固定）
发现 2：不同通道的量级/偏斜差异大（有通道级 outlier）

含义：按通道选 scale（per-channel）是准的，而且 scale 可以"持续更新"而不怕时间漂移

### 5.2 Value 的分布：随时间漂移

Value 张量：同一通道的 value 分布随 token 变化明显

含义：per-channel 的 scale 会"过期"；per-token 的 scale 才能跟上当前分布

**这是 KIVI 的命名式结论：K 按通道量化、V 按 token 量化。**

为什么不对称是合理的：

QK 分数= $q\cdot k$：key 的量化误差通过点积进入 softmax，需要稳定的通道级精度
V 直接加权进输出：value 的误差被注意力权重"平均"，per-token 精度更划算

---

## 6. KIVI 算法

### 6.1 方案总览

K cache：2-bit，per-channel 非对称量化（每 head 每通道一个 scale + zero-point）
V cache：2-bit，per-token 非对称量化（每个 token 一个 scale + zero-point）
免调参：不需要微调，即插即用

### 6.2 流式实现要点

KV 是**在线生成**的（一个 token 一个 token 写），所以量化必须流式：

1. 保留一小段最近的 token 在 FP16 缓冲（避免反复量化、保证近期精度）
2. K：以一小段 token 为单元，统计该段每个通道的$\max/\min \to per-channel$非对称量化$\to$写 2-bit
3. V：每 token 到达即按该 token 自己的 max/min 量化（per-token）
4. 解码时：2-bit 数据按需反量化回 FP16 参与 attention

实现细节（块大小、缓冲长度）是工程参数，见 [KIVI 官方代码](https://github.com/jy-yuan/KIVI)。

### 6.3 为什么 2-bit 还能近无损

1. 粒度对：K 的通道规律 + V 的 token 适配，让 2-bit 网格利用率高
2. 非对称：KV 分布往往有偏，zero-point 把网格对准分布（01 章 3.7）
3. 误差去处：V 误差被注意力权重平均；K 误差虽然进 softmax，但通道级精度保住了主要结构
4. 免调参不意味着免验证：论文在多个模型上逐长度验证 PPL 差异落在噪声内

---

## 7. 结果与边界

### 7.1 论文结果

- **模型**：Llama-2 7B/13B/70B、OPT、Mistral 等。
- **质量**：2-bit KV 在论文测试长度（数千到上万 token）下 perplexity 与 FP16 差异在噪声范围内；4-bit 更稳。
- **内存**：KV 省 4 倍$\to$同一 GPU 可支持更大 batch 或更长上下文。
- **即插即用**：无需微调、无需额外校准训练。

### 7.2 边界与生产建议

1. 2-bit 是研究甜点；生产上 KV 量化通常从 4-bit 或 FP8 起步，先跑长上下文评测
2. 超长上下文（几十万 token）下误差累积更明显，需要按实际长度验证
3. 与 GQA 天然兼容（KV head 少，量化对象更少）
4. vLLM / TensorRT-LLM 已提供 KV cache 的 FP8/INT8 量化选项

---

## 8. 与 KV 优化的其他手段的关系

KV 优化是一个组合拳，量化只是其中一块：

| 手段 | 思路 | 与量化的关系 |
|---|---|---|
| PagedAttention（vLLM） | 显存管理（分页，避免碎片/预分配浪费） | **正交**，可叠加 |
| 前缀缓存 / KV 复用 | 相同前缀只算一次 | **正交**，可叠加 |
| 淘汰（H2O / StreamingLLM） | 丢不重要的旧 token | 可叠加；量化省所有 token 的位 |
| ThinK | 裁剪 key 的冗余通道 | 可与 KIVI 叠加（论文报告过） |
| KV 量化（KIVI 等） | 每个 KV 用更少 bit | 本章主题 |

> 工程排序：先做 PagedAttention + 前缀缓存（无损、成熟），再考虑 KV 量化（有损但收益大），最后才考虑淘汰（改变语义）。

---

## 9. 本章小结

1. **KV 是长上下文的显存大头**：公式 `2·L·H·d·S·bytes`，$70B @32K \approx 10.7 GB$。
2. **误差滚雪球**：KV 被反复读取，softmax 放大扰动$\to$逐 token 累积。
3. **K 按通道、V 按 token**：K 分布稳定（通道模式），V 分布漂移（随时间变）——KIVI 的全部设计都从这里来。
4. **2-bit 非对称 + 流式量化 + 免调参**：即插即用，4x 省显存，近无损。
5. **生产建议**：4-bit/FP8 起步、按实际长度评测、与 PagedAttention/前缀缓存叠加。

> 一句话记忆：**"Key 像星座（稳定、按通道有规律），Value 像天气（随时变），所以 Key 按通道记、Value 按天记。"**

---

## 10. 习题与解答

### 题 1（计算）：KV 显存

模型：64 层、4 个 KV head、$\text{head\_dim} 128$。计算 FP16 与 2-bit 下 64K 上下文的 KV 显存。

<details>
<summary>题 1 解答</summary>

每$token = 2\times 64\times 4\times 128\times 2 = 131,072 B = 128 KB$。
$64K = 65536 token \to FP16 \approx 8.59 GB$；$2-bit \approx 2.15 GB$（省 4 倍）。
</details>

### 题 2（思考）：误差为什么逐 token 累积

用"KV 被读多少次"回答：第 t 个 token 写入的$K_{t}$，在序列长度为 T 时会被后续多少步读取？量化误差因此被放大多少次？

<details>
<summary>题 2 解答</summary>

$K_{t}$被 token t+1 … T 每一步读取，共 T−t 次；每次读取都参与一层 attention 且可能影响后续所有 token 的生成（输出 token 会继续影响后面）。所以误差既被"重复使用"，又被"传播放大"——这就是累积。
</details>

### 题 3（设计）：为什么 V 不用 per-channel

如果 V 也用 per-channel（scale 用前 1000 个 token 统计），长对话后 V 分布漂移，会发生什么？

<details>
<summary>题 3 解答</summary>

scale 过期：新 token 的 V 分布可能远超/远低于旧 scale 覆盖范围$\to$大量截断或有效位宽浪费（02 章）；per-token scale 始终跟当前 token 对齐，截断可控。K 因为分布稳定才敢用 per-channel。
</details>

### 题 4（编程）：per-token vs per-channel 对比

生成随时间漂移的 V 数据（前一半 N(0,1)，后一半 N(10,1)），比较 per-token 与 per-channel（用前半统计）2-bit 量化的总误差。

<details>
<summary>题 4 解答要点</summary>

per-channel 的 scale 固定$\to$后一半全部落在网格边缘/外，误差大；per-token 每 token 自适应，误差接近理论最优。验证 KIVI 的结论：**分布漂移时 per-token 完胜**。
</details>

### 题 5（开放）：KV 量化的验收

给 KV 2-bit 方案设计验收实验：模型、长度、指标、对照组各是什么？

<details>
<summary>题 5 解答要点</summary>

模型：生产用模型（如 70B 级）；长度：覆盖线上最大上下文（如 32K/128K）；指标：perplexity（WikiText-2 长段）+ 长上下文任务（如 RULER/大海捞针）+ 自定义业务评测；对照组：FP16 KV 与 4-bit KV，跑多遍确认差异在噪声内（10 章的方法论）。
</details>

---

## 11. 延伸阅读

1. [KIVI（arXiv:2402.02750）](https://arxiv.org/abs/2402.02750)：本章主文献；[代码](https://github.com/jy-yuan/KIVI)
2. [PagedAttention（vLLM 论文，arXiv:2309.06180）](https://arxiv.org/abs/2309.06180)：KV 显存管理（正交手段）
3. [ThinK（ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8edb116d5b288b6a9bba4c16ab647702-Abstract-Conference.html)：key 通道剪枝 + KIVI 叠加
4. 上一篇：[07 激活量化](./LLM量化精读笔记-07-激活量化-LLM-int8与SmoothQuant.md)；下一篇：**[09 QAT 与训练内量化：STE、QLoRA、BitNet b1.58]**——从"事后补偿"转向"让模型学会忍受量化"。

