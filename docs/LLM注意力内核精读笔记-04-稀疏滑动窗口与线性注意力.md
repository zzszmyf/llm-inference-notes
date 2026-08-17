# LLM 注意力与计算内核精读笔记 · 04 稀疏、滑动窗口与线性注意力

> 对应：Xiao et al., *Efficient Streaming Language Models with Attention Sinks*（StreamingLLM，arXiv:2309.17453，2023）；Zhang et al., *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*（arXiv:2306.14048，NeurIPS 2023）；Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*（arXiv:2006.16236，ICML 2020）；Choromanski et al., *Rethinking Attention with Performers*（arXiv:2009.14794，ICLR 2021）；Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*（arXiv:2312.00752，2023）；Mistral AI, *Mistral 7B*（arXiv:2310.06825，2023）。
> 前置：01 章（$O(L^2)$ 复杂度）、02 章（FlashAttention 的精确路线）。学完本章你应该能：① 说出对抗 $L^2$ 的三条路线（少看位置/少存 KV/换计算范式）与代表方法；② 解释滑动窗口注意力的局部性假设与 StreamingLLM 的 attention sink 修复；③ 写出线性注意力的结合律推导（$O(L d^2)$）与 Performers 的核近似思路；④ 说清 Mamba 的选择性 SSM 与线性注意力的本质区别；⑤ 用复杂度表格判断"什么场景该上哪条路线"。

---

## 目录（本章）

1. 本章目标
2. 三条路线：对抗 $L^2$ 的三种策略
3. 滑动窗口注意力（SWA）
4. Attention Sink 与 StreamingLLM
5. 驱逐类方法：H2O
6. 线性注意力：把 softmax 换成核
7. Performers：随机特征近似
8. Mamba：选择性状态空间模型
9. 统一视角：稀疏 vs 线性
10. 复杂度对照与数值算例
11. 实验数据
12. 本章小结
13. 习题与解答
14. 延伸阅读

---

## 2. 三条路线：对抗 $L^2$ 的三种策略

01 章说注意力 FLOPs $= 4L^2d$、KV 显存 $= O(L)$。02/03 章把"计算怎么快、KV 怎么小"做到了极致，但 **$L^2$ 的平方本质没动**。本章的三种策略从三个方向绕开它：

```
路线 A：少看位置（稀疏）
  每个 query 只看一部分 key（窗口、top-k、检索）
  → 复杂度 O(L·W)，W ≪ L

路线 B：少存 KV（驱逐）
  显存预算固定，动态决定哪些 token 值得留
  → KV 显存 O(预算)，与 L 解耦

路线 C：换计算范式（线性/状态空间）
  用结合律或循环把 L² 换成 d²
  → 复杂度 O(L·d²)，d ≪ L 时线性
```

注意路线 C 通常是**近似**的（softmax 被替换）；02 章 FlashAttention 是精确路线，两类互补。

---

## 3. 滑动窗口注意力（SWA）

### 3.1 定义

假设"远距离 token 对当前 token 的贡献可忽略"，每个 query 只看最近 $W$ 个 key：

$$
o_i = \sum_{j = i-W+1}^{i} a_{ij}\, v_j
$$

FLOPs 从 $4L^2d$ 降到 $4LWd$（$W \ll L$），KV cache 也只需存窗口 $W$。代表：Longformer（Beltagy et al. 2020）、Mistral 7B。

### 3.2 Mistral 的做法

```
窗口 W = 4096；GQA 8 个 KV 头（03 章）
每一层的"有效感受野" = 层数 × W（多层叠加后远处信息能间接到达）
论文结论：SWA 让"任意长度序列 + 降低推理成本"兼得
```

### 3.3 代价

```
优点：复杂度与 KV 显存都只跟 W 有关
缺点：局部性假设对"需要远距离精确引用"的任务（长文档检索式问答）不友好；
     训练窗口有限 → 推理超过训练长度时质量崩（下一节的痛点）
```

---

## 4. Attention Sink 与 StreamingLLM

### 4.1 窗口注意力为什么会崩

把窗口直接用于"流式长对话"（文本长度超过训练长度、超过缓存）：只保留最近 $W$ 个 KV，初始 token 被踢出。论文发现**质量崩坏**——不是语义问题，而是 softmax 的"垃圾收集"失效。

### 4.2 Attention Sink 现象

观察：模型给初始 token（尤其第一个 token）分配**异常高的注意力分数**，即使它与当前内容无关。这些 token 像"下水道"一样吸走多余的注意力质量——所以叫 **attention sink**。证据：保留初始 token 的 KV，窗口注意力的性能基本恢复。

### 4.3 StreamingLLM 方案

$$
\text{KV cache} = \underbrace{\text{前 } S \text{ 个 sink token}}_{\text{固定保留}} + \underbrace{\text{最近 } W \text{ 个 token}}_{\text{窗口}}
$$

```
要点：
① 不需要微调，直接用现成模型（Llama-2、MPT、Falcon、Pythia）
② 稳定流式到 400 万 token 以上
③ 比"窗口 + 重算"基线快最多 22.2 倍
④ 预训练时加一个占位 token 当专用 sink，效果更好
```

代价：这是**近似**——它丢掉了窗口外的历史信息，但实验表明 sink + 窗口已经足够稳定。

---

## 5. 驱逐类方法：H2O

### 5.1 观察

论文统计发现：**一小部分 token 贡献了注意力分数的绝大部分**——"重击手"（Heavy Hitters, $H_2$）。它们与文本中的高频共现词强相关；删掉它们性能显著下降。

### 5.2 H2O 驱逐策略

KV 预算固定（如 20% 的 token），动态维护两类：

$$
\text{保留} = \underbrace{\text{最近 } R \text{ 个 token}}_{\text{局部性}} + \underbrace{\text{累计注意力分数最高的 } H_2 \text{ token}}_{\text{重要性}}
$$

驱逐被建模为**动态子模最大化**问题，论文给出近似保证。注意驱逐后 KV 不再完整，注意力是近似的（且驱逐不可逆）。

### 5.3 数据

```
20% 预算下：吞吐比 DeepSpeed-Zero-Inference / HF Accelerate 高 29 倍，
          比 FlexGen 高 3 倍（OPT-6.7B/30B）；同 batch 延迟降 1.9 倍
```

---

## 6. 线性注意力：把 softmax 换成核

### 6.1 结合律：$L^2$ 从哪来、怎么消失

标准注意力的 $\mathbf{P}\mathbf{V}$ 不能交换顺序（softmax 的归一化依赖整行）。若把注意力分数换成**可分解核**：

$$
\mathrm{Att}(q_i, k_j) = \frac{\varphi(q_i)^\top \varphi(k_j)}{\sum_{j'} \varphi(q_i)^\top \varphi(k_{j'})}
$$

则输出可写成：

$$
o_i = \frac{\varphi(q_i)^\top \sum_{j \le i} \varphi(k_j) v_j^\top}{\varphi(q_i)^\top \sum_{j \le i} \varphi(k_j)}
$$

关键在于**结合律**（Katharopoulos et al. 2020）：

$$
\varphi(\mathbf{Q})\,\big(\varphi(\mathbf{K})^\top \mathbf{V}\big)
= \big(\varphi(\mathbf{Q})\,\varphi(\mathbf{K})^\top\big)\,\mathbf{V}
$$

先算 $\varphi(\mathbf{K})^\top \mathbf{V} \in \mathbb{R}^{d \times d}$（与 $L$ 无关！），再左乘 $\varphi(\mathbf{Q})$。复杂度：

$$
O(L^2 d) \;\to\; O(L d^2)
$$

推理时变成**循环更新**：维护状态 $\mathbf{S}_i = \mathbf{S}_{i-1} + \varphi(k_i) v_i^\top$（$d \times d$）与归一化向量 $\mathbf{z}_i$——这正是"Transformer 是 RNN"的含义。

### 6.2 代价

```
优点：训练 O(L d²)、推理 O(d²)/步，长序列友好
缺点：softmax 被替换 → 不再是精确 softmax 注意力；
     无界核的数值稳定性需要额外处理
```

---

## 7. Performers：随机特征近似

要"线性"又要"接近 softmax"，可以用**核技巧**：softmax 的核 $e^{q^\top k}$ 可以用随机特征近似：

$$
e^{q^\top k} \approx \mathbb{E}\left[\varphi(q)^\top \varphi(k)\right], \qquad
\varphi(x) = \frac{1}{\sqrt{m}}\, f(x)\, \odot\, \text{random cos/sin features}
$$

Performers（FAVOR+）用**正随机特征**保证方差小，能在 $O(L d^2)$ 内近似 softmax 注意力，误差有界。后续大量"线性 Transformer"（Performer、Linear Transformer、Random Features Attention 等）都走这条路线。

---

## 8. Mamba：选择性状态空间模型

### 8.1 从线性注意力到 SSM

线性注意力的循环形式是"无状态选择"的：$\mathbf{S}_i$ 的更新对所有 token 一视同仁。状态空间模型（SSM）把状态更新写成：

$$
h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \qquad
y_t = C_t h_t
$$

### 8.2 关键：选择性（Selectivity）

前人 SSM 的 $\bar{A}, \bar{B}, C$ 是**输入无关**的常数，导致无法做"基于内容的推理"（该记住的记不住、该忘的忘不掉）。Mamba 的贡献：

```
让 A, B, C 都依赖当前输入 x_t（选择性）
→ 模型可以按内容决定"传播还是遗忘"
→ 代价：不能再高效用卷积，需要专门的硬件感知并行扫描算法
```

### 8.3 数据

```
Mamba-3B 在语言建模上超过同尺寸 Transformer、打平两倍尺寸的 Transformer
推理吞吐比同尺寸 Transformer 高 5 倍；序列长度线性扩展，实测到百万级
```

---

## 9. 统一视角：稀疏 vs 线性

```
稀疏（路线 A/B）：保留 softmax 注意力，但"看得少 / 存得少"
  → 语义可解释、与 FlashAttention 兼容、质量接近精确
  → 但信息被硬性截断（窗口外/被驱逐的永远看不到）

线性（路线 C）：换核 / 换状态模型，复杂度真正线性
  → 任意位置都能"影响"当前输出（无硬截断）
  → 但近似了 softmax，长程精确引用能力存疑
```

工程经验：

```
长上下文但需要精确引用 → FlashAttention + 稀疏/驱逐（StreamingLLM、H2O）
极致吞吐、可接受近似 → 线性注意力 / Mamba
当前主流 LLM 服务 → 仍是"精确注意力 + KV 优化"（02/03/05 章），
                    稀疏/线性更多是研究前沿与端侧选择
```

---

## 10. 复杂度对照与数值算例

| 方法 | 每层 FLOPs | KV 显存/token | 精确性 |
|---|---|---|---|
| 标准注意力 | $4L^2 d$ | $2 d_{\text{kv}}$ | 精确 |
| FlashAttention | $4L^2 d$（省 HBM 访问） | $2 d_{\text{kv}}$ | 精确 |
| 滑动窗口 | $4LWd$ | $2 d_{\text{kv}}$（窗口内） | 近似（截断） |
| StreamingLLM | $4L(W+S)d$ | $2(W+S)d_{\text{kv}}$ | 近似 |
| H2O | $4L^2d$（计算不变，省显存） | 预算固定 | 近似（驱逐） |
| 线性注意力 | $4Ld^2$ | $d^2$（状态） | 近似（换核） |
| Mamba | $O(L d N)$ | $O(dN)$（状态） | 架构不同 |

数值例（$d=64$、$L=32K$、$W=4K$，单头）：

$$
\text{标准： } 4 \times 32768^2 \times 64 \approx 275\ \text{GFLOP}
$$

$$
\text{滑窗： } 4 \times 32768 \times 4096 \times 64 \approx 34\ \text{GFLOP} \quad(\times 8\ \text{省})
$$

$$
\text{线性： } 4 \times 32768 \times 64^2 \approx 0.54\ \text{GFLOP} \quad(\times 500\ \text{省})
$$

数字差距巨大，但**别忘了精确性**：线性路线省的是"近似后的注意力"。

---

## 11. 实验数据

| 方法 | 已核实数据 | 出处 |
|---|---|---|
| StreamingLLM | 稳定流式 4M+ token；比滑窗重算基线快 22.2x；无需微调 | 摘要/§4 |
| H2O | 20% 预算：吞吐比 Zero-Inference/HF-Accelerate 高 29x、比 FlexGen 高 3x；延迟降 1.9x | 摘要 |
| Mistral SWA | 窗口 4096 + GQA；超 Llama 2 13B 且推理成本更低 | 摘要 |
| Mamba | 吞吐比 Transformer 高 5x；线性扩展；Mamba-3B 打平 2 倍尺寸 Transformer | 摘要 |
| Katharopoulos 线性注意力 | $O(L d^2)$ 训练、$O(d^2)$/步推理；等价 RNN 形式 | 论文 §3 |

---

## 12. 本章小结

1. **三条路线**：少看（稀疏/滑窗）、少存（驱逐）、换范式（线性/SSM）。
2. **Attention Sink**：初始 token 是 softmax 的"垃圾收集器"，窗口注意力踢掉它就会崩；保留 sink + 窗口即可流式 4M token。
3. **H2O**：按"累计注意力分数"驱逐，20% 预算带来 29x 吞吐提升。
4. **线性注意力**：$\varphi(Q)(\varphi(K)^\top V)$ 的结合律把 $L^2$ 换成 $d^2$；Performer 用随机特征近似 softmax。
5. **Mamba**：选择性 SSM 让状态按内容决定记忆/遗忘，吞吐 5x、线性扩展。
6. **工程提醒**：这些大多是近似；精确注意力 + KV 优化仍是主流服务的选择。

> 一句话记忆：**"想让注意力跑赢平方，要么看得少（窗口/驱逐），要么换个算法（核/状态空间）——前者保留 softmax 的味道，后者把 softmax 换掉，省下来的都是时间，付出去的是精确性。"**

---

## 13. 习题与解答

### 题 1（推导）：线性注意力的结合律

写出 $\varphi(\mathbf{Q})(\varphi(\mathbf{K})^\top \mathbf{V})$ 与 $(\varphi(\mathbf{Q})\varphi(\mathbf{K})^\top)\mathbf{V}$ 的维度，说明为什么前者只需 $O(L d^2)$，并推导逐位置循环更新式。

<details>
<summary>题 1 解答</summary>

$\varphi(\mathbf{K})^\top \mathbf{V}$ 是 $d \times d$；$\varphi(\mathbf{Q})$ 是 $L \times d$，左乘得 $L \times d$。先算 $d\times d$ 矩阵（$O(L d^2)$），再乘 $L\times d$（$O(L d^2)$），总 $O(L d^2)$；而右侧先算 $L\times L$ 是 $O(L^2 d)$。循环形式：$\mathbf{S}_i = \mathbf{S}_{i-1} + \varphi(k_i)v_i^\top$，$\mathbf{z}_i = \mathbf{z}_{i-1} + \varphi(k_i)$，$o_i = \varphi(q_i)^\top \mathbf{S}_i / \varphi(q_i)^\top \mathbf{z}_i$。
</details>

### 题 2（计算）：滑窗 vs 标准

$L=32K$、$W=4K$、$d=64$：算标准与滑窗的单头 FLOPs 并给比值；若 $W=512$ 呢？

<details>
<summary>题 2 解答</summary>

标准 $4\times32768^2\times64 \approx 275$ GFLOP；$W=4K$ 时 $4\times32768\times4096\times64 \approx 34.4$ GFLOP，省 8 倍；$W=512$ 时 $\approx 4.3$ GFLOP，省 64 倍。注意 KV 显存同样只跟 $W$ 相关。
</details>

### 题 3（思考）：attention sink 为什么存在

从 softmax 的归一化性质解释"初始 token 吸收注意力质量"这一现象，并说明 StreamingLLM 为什么"踢掉 sink 就崩、留两个 token 就恢复"。

<details>
<summary>题 3 解答要点</summary>

softmax 把注意力质量归一化到 1；当窗口内没有"稳定的高分 token"时，分布被迫把质量分散到任意位置，数值上不稳定。初始 token 在训练里长期充当"兜底的高分目标"，模型学会了把多余质量倾泻到它身上（sink）。流式时若窗口不含 sink，归一化失去锚点；保留前几个 token 即恢复。这也是为什么预训练加专用占位 sink token 能进一步稳定。
</details>

### 题 4（设计）：H2O 的驱逐策略

设计一个"累计注意力分数"的在线度量（如何增量维护、如何防止早期 token 永远占优），并说明为什么"最近 token + Heavy Hitters"的组合优于纯最近/纯高频。

<details>
<summary>题 4 解答要点</summary>

每个 token 维护"被作为 key 时收到的注意力分数之和"（可在每步并行累加，代价可摊销）。要防早期占优：用滑动窗口内的累计值或分数归一化。纯最近丢长程、纯高频丢局部上下文；混合保留兼顾两者，且驱逐可建模为动态子模问题近似最优。
</details>

### 题 5（对比）：线性注意力 vs Mamba

列出线性注意力（Katharopoulos）与 Mamba 在"状态形式、选择性、训练并行性、精确性"四个维度上的差异。

<details>
<summary>题 5 解答要点</summary>

线性注意力：状态 $\mathbf{S}$（$d\times d$）无选择性，训练可并行（矩阵乘法），是 softmax 的核近似；Mamba：状态 $h$（$dN$）有选择性（A/B/C 依赖输入），训练需硬件感知并行扫描，是全新架构而非注意力近似。选择性让 Mamba 能做内容相关记忆，代价是失去纯卷积式并行。
</details>

### 题 6（编程）：窗口 + sink 模拟

实现一个 toy：给定注意力分数序列，比较 ① 全注意力、② 纯窗口、③ 窗口 + sink（保留前 2 个 token）三种方案的"有效归一化质量"，并画出长度增长时的困惑度趋势，复现 StreamingLLM 的核心结论。

<details>
<summary>题 6 解答要点</summary>

用 toy 分数分布模拟：纯窗口在长度超过窗口后，softmax 分母失去 sink 锚点，输出分布漂移；加 sink 后分母稳定。困惑度（或分布 KL）趋势应呈现"纯窗口骤升、sink+窗口平缓"——这就是 StreamingLLM 论文 Fig. 2 的 toy 版。
</details>

---

## 14. 延伸阅读

1. [Efficient Streaming Language Models with Attention Sinks（arXiv:2309.17453）](https://arxiv.org/abs/2309.17453)：StreamingLLM 与 attention sink 的出处。
2. [H2O: Heavy-Hitter Oracle（arXiv:2306.14048）](https://arxiv.org/abs/2306.14048)：KV 驱逐策略与吞吐数据。
3. [Transformers are RNNs: Linear Attention（arXiv:2006.16236）](https://arxiv.org/abs/2006.16236)：线性注意力与循环等价形式。
4. [Rethinking Attention with Performers（arXiv:2009.14794）](https://arxiv.org/abs/2009.14794)：随机特征近似 softmax。
5. [Mamba（arXiv:2312.00752）](https://arxiv.org/abs/2312.00752)：选择性 SSM 与硬件感知算法。
6. 上一篇：[03 注意力头变体：MQA/GQA/MLA](./LLM注意力内核精读笔记-03-注意力头变体-MQA-GQA-MLA.md)；下一篇：**05 PagedAttention 与 KV 显存管理**——不砍 KV 内容，而是把 KV 从"连续大块内存"变成"分页小片"，让显存利用率接近 100%。
