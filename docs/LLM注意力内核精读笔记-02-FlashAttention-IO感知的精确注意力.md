# LLM 注意力与计算内核精读笔记 · 02 FlashAttention：IO 感知的精确注意力

> 对应：Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*（arXiv:2205.14135，NeurIPS 2022）；Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*（arXiv:2307.08691，ICLR 2024）；Shah et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*（arXiv:2407.08608，2024）。
> 前置：01 章（注意力复杂度、prefill/decode 形态、max-subtraction）。学完本章你应该能：① 说清朴素注意力"快不起来"的真正瓶颈是 HBM 读写而不是 FLOPs；② 写出 FlashAttention 的三个核心技巧（tiling、online softmax、recompute）及各自解决的问题；③ 推导 online softmax 的增量更新公式并手算验证与整体 softmax 一致；④ 复述 IO 复杂度定理（$O(N^2d^2/M)$ vs $\Omega(Nd+N^2)$）与最优性结论；⑤ 对比 FA1/FA2/FA3 的改进路线与实测数据；⑥ 解释"重计算多花 FLOPs 反而更快"和"FlashAttention 是精确算法"这两个关键论断。

---

## 目录（本章）

1. 本章目标
2. 01 章的遗留问题：朴素注意力"卡"在哪
3. IO 复杂度视角：瓶颈是 HBM，不是 FLOPs
4. 技巧一：Tiling（分块）
5. 技巧二：Online Softmax（增量 softmax）
6. 技巧三：Recomputation（反向重算）
7. 算法与定理
8. 数值算例：online softmax 手算
9. FlashAttention-2：并行与工作划分
10. FlashAttention-3：Hopper 上的异步与低精度
11. 实验数据汇总
12. 衔接：decode 侧、FP8 注意力与量化系列
13. 本章小结
14. 习题与解答
15. 延伸阅读

---

## 2. 01 章的遗留问题：朴素注意力"卡"在哪

01 章算出注意力 FLOPs $= 4L^2d$、显存 $O(L^2)$。朴素 PyTorch 实现的三步：

$$
\mathbf{S} = \mathbf{Q}\mathbf{K}^\top, \qquad
\mathbf{P} = \mathrm{softmax}(\mathbf{S}), \qquad
\mathbf{O} = \mathbf{P}\mathbf{V}
$$

三个浪费：

```
① 显存：S 和 P 都是 N×N 矩阵，必须完整落在 HBM
   L=128K 时一个 S 就是 32GB（FP16）→ 直接爆显存
② IO：S 写一次、softmax 读一次、P 写一次、PV 再读一次
   每个元素被搬进搬出 HBM 多次
③ 白算：因果掩码在矩阵算完之后才应用，上三角 FLOPs 全浪费
```

FlashAttention 的答案：**不让 $N \times N$ 矩阵落地**——把它拆成能塞进片上 SRAM 的块，逐块算、逐块合并。难点只有一个：softmax 的归一化依赖整行，怎么分块还算得**精确**？这就是 online softmax。

---

## 3. IO 复杂度视角：瓶颈是 HBM，不是 FLOPs

现代 GPU 有两层存储：

```
HBM（主存）：容量大（几十 GB）、带宽有限（A100 约 2TB/s）
SRAM（片上）：容量小（每 SM 192KB）、带宽极高（~19TB/s 量级）
```

一次 HBM 访问的成本远高于一次浮点运算。所以算法的快慢不由 FLOPs 决定，而由 **HBM 访问次数**决定——这正是"IO 感知"的含义。

### 3.1 标准注意力的 HBM 账本

$$
\Omega(Nd + N^2)
$$

解读：读 $\mathbf{Q},\mathbf{K},\mathbf{V}$ 各 $Nd$ 是不可避免的；但 $\mathbf{S}$ 和 $\mathbf{P}$ 的**写入再读出**贡献了 $N^2$ 量级的 HBM 访问（每个元素至少 3–4 次搬移）。

### 3.2 FlashAttention 的目标

$$
O\!\left(\frac{N^2 d^2}{M}\right)
$$

其中 $M$ 是 SRAM 容量。当 $M \gg d^2$ 时，这个系数远小于标准实现的常数。论文实测：典型配置下 HBM 访问**最多减少 9 倍**（Fig. 2）。

---

## 4. 技巧一：Tiling（分块）

把 $\mathbf{Q}$ 按行切成 $T_r$ 块、$\mathbf{K},\mathbf{V}$ 按行切成 $T_c$ 块：

$$
\mathbf{Q}_i \in \mathbb{R}^{B_r \times d}, \qquad
\mathbf{K}_j, \mathbf{V}_j \in \mathbb{R}^{B_c \times d}
$$

每次只把一块 $\mathbf{Q}_i$、一块 $\mathbf{K}_j$、一块 $\mathbf{V}_j$ 和中间的 $\mathbf{S}_{ij} = \mathbf{Q}_i \mathbf{K}_j^\top$ 放进 SRAM，算完合并到输出后丢弃。**$N \times N$ 的矩阵从头到尾不落 HBM**。

块大小怎么定？SRAM 里同时要放 $\mathbf{Q}$ 块（$B_r d$）、$\mathbf{K}$ 块（$B_c d$）、$\mathbf{V}$ 块（$B_c d$）、$\mathbf{S}$ 块（$B_r B_c$）四项，所以：

$$
B_c = \left\lceil \frac{M}{4d} \right\rceil, \qquad
B_r = \min\left(\left\lceil \frac{M}{4d} \right\rceil, d\right)
$$

（$\approx M/4d$ 的直觉：四项各占 $\approx M/4$。）

---

## 5. 技巧二：Online Softmax（增量 softmax）

分块后，第 $j$ 块的 softmax 归一化依赖"已经看过的 $1..j-1$ 块"，不能独立归一化。解法是维护三个运行量：

$$
m^{(j)} = \max \text{ of scores so far}, \qquad
\ell^{(j)} = \sum \exp(s - m^{(j)}), \qquad
\mathbf{O}^{(j)} = \text{unnormalized output so far}
$$

处理新块 $j+1$（块内最大 $\tilde{m}$、块内指数和 $\tilde{\ell}$、块内贡献 $\tilde{\mathbf{P}}\mathbf{V}$）时，按以下三步更新：

$$
m^{(j+1)} = \max(m^{(j)},\ \tilde{m})
$$

$$
\ell^{(j+1)} = e^{m^{(j)} - m^{(j+1)}}\, \ell^{(j)} + e^{\tilde{m} - m^{(j+1)}}\, \tilde{\ell}
$$

$$
\mathbf{O}^{(j+1)} = \frac{1}{\ell^{(j+1)}}\left(
e^{m^{(j)} - m^{(j+1)}} \ell^{(j)} \mathbf{O}^{(j)} + e^{\tilde{m} - m^{(j+1)}} \tilde{\mathbf{P}} \mathbf{V}
\right)
$$

直觉：如果新的最大值比旧的大（$m^{(j+1)} > m^{(j)}$），旧块的指数都要按 $e^{m^{(j)} - m^{(j+1)}} < 1$ **打折**；新块按 $e^{\tilde{m} - m^{(j+1)}}$ 缩放。全部块处理完后，$\mathbf{O} = \mathbf{O}^{(T_c)}$ 已经是正确归一化的输出。

### 5.1 为什么这是"精确"算法

每一步的缩放都是软max恒等式的直接应用：

$$
\mathrm{softmax}(s) = \frac{\exp(s - m)}{\sum_{j}\exp(s_j - m)}
$$

对任意 $m$ 成立。Online softmax 只是"换了个 $m$"并等比缩放之前的累计——**结果与一次性算整行 softmax 完全相同**（只差浮点舍入）。这是 FlashAttention 与近似注意力（稀疏/线性）的本质区别：**快，但不改结果**。

---

## 6. 技巧三：Recomputation（反向重算）

反向传播需要 $\mathbf{S}$ 和 $\mathbf{P}$ 的梯度。朴素实现把它们存下来（$O(N^2)$ 显存）；FlashAttention 选择：

```
前向：只存每块的归一化因子 ℓ 和行最大 m（O(N)）
反向：从 HBM 重新读 Q/K/V 块，在 SRAM 里重算 S 和 P
```

代价是反向 FLOPs 大约翻倍；收益是**免掉 $O(N^2)$ 的中间矩阵落盘**，反向的 HBM 访问从 $O(N^2)$ 降到 $O(N^2d^2/M)$。这就是"**用 FLOPs 换 HBM 访问**"——GPU 的算力是富余的，带宽才是稀缺的。

---

## 7. 算法与定理

### 7.1 前向伪代码（FA1 Algorithm 1 精简版）

```
输入：Q, K, V ∈ R^{N×d}（HBM），SRAM 容量 M
1. B_c = ⌈M/4d⌉，B_r = min(⌈M/4d⌉, d)
2. 初始化 O = 0, ℓ = 0, m = −∞（HBM）
3. for j = 1..T_c:                    # 遍历 K/V 块
4.   载入 K_j, V_j 到 SRAM
5.   for i = 1..T_r:                  # 遍历 Q 块
6.     载入 Q_i, O_i, ℓ_i, m_i 到 SRAM
7.     S_ij = Q_i K_j^T               # 片上算
8.     m̃ = rowmax(S_ij), P̃ = exp(S_ij − m̃), ℓ̃ = rowsum(P̃)
9.     m_i^new = max(m_i, m̃)
10.    ℓ_i^new = e^{m_i−m_i^new}ℓ_i + e^{m̃−m_i^new}ℓ̃
11.    O_i ← (e^{m_i−m_i^new}ℓ_i O_i + e^{m̃−m_i^new}P̃ V_j) / ℓ_i^new
12. 返回 O
```

### 7.2 两条定理

**定理 1（正确性与资源）**：上述算法返回精确的 $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$，FLOPs 为 $O(N^2d)$，**额外显存 $O(N)$**。

**定理 2（IO 复杂度与最优性）**：FlashAttention 需要 $O(N^2 d^2 / M)$ 次 HBM 访问，而标准注意力需要 $\Omega(Nd + N^2)$ 次；并且**不存在渐近更优的精确注意力算法**（对所有 SRAM 大小）。

---

## 8. 数值算例：online softmax 手算

复用 01 章 $q_3$ 的分数行 $s = (1.414,\ 0.707,\ 0.707)$，$V$ 取第一坐标 $v = (1,\ 0,\ 1)$。把行分两块：块 1 = 前两个分数，块 2 = 最后一个。

**块 1**：

$$
\tilde{m} = 1.414, \quad \tilde{\mathbf{P}} = \exp(s-\tilde{m}) = (1,\ 0.493), \quad \tilde{\ell} = 1.493
$$

$$
m^{(1)} = 1.414, \quad \ell^{(1)} = 1.493, \quad
\mathbf{O}^{(1)} = \frac{1\cdot 1 + 0.493 \cdot 0}{1.493} = 0.670
$$

**块 2**：

$$
\tilde{m} = 0.707, \quad \tilde{\mathbf{P}} = (1), \quad \tilde{\ell} = 1
$$

$$
m^{(2)} = \max(1.414, 0.707) = 1.414
$$

$$
\ell^{(2)} = e^{0}\cdot 1.493 + e^{0.707-1.414}\cdot 1 = 1.493 + 0.493 = 1.986
$$

$$
\mathbf{O}^{(2)} = \frac{1.493 \times 1 \times 0.670 + 0.493 \times 1 \times 1}{1.986}
= \frac{1.0 + 0.493}{1.986} = 0.752
$$

**对照整体 softmax**：$a = (0.5035,\ 0.2483,\ 0.2483)$，输出 $= 0.5035\cdot1 + 0.2483\cdot0 + 0.2483\cdot1 = 0.7518 \approx 0.752$ ✓。

> 注意：$\mathbf{O}^{(1)} = 0.670$ 是"只看前两块"的部分输出；合入块 2 时按 $\ell$ 正确重归一化——分块没有改变最终结果。

---

## 9. FlashAttention-2：并行与工作划分

FA1 已经比标准实现快 2–4 倍，但论文实测其前向只达到 A100 理论峰值 FLOPs 的 30–50%（反向 25–35%），而优化过的 GEMM 能到 80–90%。差距来自**工作划分**：

```
FA2 的三处改进：
① 序列长度维并行：每个 thread block 负责一个 (query 块, key 块) 对，减少串行依赖
② 减少非 matmul 运算：不再频繁重归一化，把 rescale 推迟到块末做一次
③ warp 内划分：KV 的列维在 warp 间均分，避免重复读共享内存，支持 d 到 256
```

结果：

$$
\text{FA2} \approx 2\times \text{FA1}, \qquad \text{峰值} \approx 230\ \text{TFLOPs/s} = 73\%\ \text{理论峰值（A100）}
$$

---

## 10. FlashAttention-3：Hopper 上的异步与低精度

FA2 在 H100 上只到约 35% 利用率（GEMM 能到 80–90%）。H100 的新硬件给了 FA3 三个杠杆：

```
① TMA 异步搬运：Tensor Memory Accelerator 专做 HBM↔SRAM 拷贝，
   让数据搬运与计算重叠（warp specialization：生产者 warp 搬运、消费者 warp 计算）
② 软max 藏在 GEMM 下面：把 softmax 的指数/归一等低吞吐运算
   与 WGMMA（warpgroup 矩阵乘）交错，用"乒乓"调度隐藏掉
③ FP8 张量核：块级量化 + incoherent processing（旋转/置换打散离群值），
   把精度损失压到最低
```

实测（H100 SXM5）：

$$
\text{FA3 FP16} = 1.5\text{–}2.0\times \text{FA2（前向）}, \quad \text{最高 740 TFLOPs/s（75% 利用率）}
$$

$$
\text{FA3 FP8} \approx 1.2\ \text{PFLOPs/s}, \quad \text{数值误差比基线 FP8 注意力低 2.6 倍}
$$

---

## 11. 实验数据汇总

| 版本/指标 | 数据 | 出处 |
|---|---|---|
| FA1 vs 标准注意力 | 2–4x；GPT-2 上最高 7.6x | FA1 Fig. 1 |
| FA1 内存 | 由 $O(N^2)$ 降到 $O(N)$，省 10–20 倍 | FA2 引言 |
| FA1 HBM 访问 | 最多减少 9 倍；$O(N^2d^2/M)$ vs $\Omega(Nd+N^2)$ | FA1 Theorem 2 |
| FA1 训练提速 | BERT-large 15%（vs MLPerf 1.1 记录）；GPT-2 3x；LRA 2.4x | FA1 §4.1 |
| FA1 长序列收益 | GPT-2 ppl −0.7；长文档分类 +6.4；Path-X 16K 61.4%、Path-256 64K 63.1% | FA1 §4.2 |
| FA2 | 约 2x vs FA1；前向 73% 理论峰值（230 TFLOPs/s on A100） | FA2 摘要/§4 |
| FA3 | FP16 1.5–2.0x vs FA2，740 TFLOPs/s；FP8 ≈1.2 PFLOPs/s；误差低 2.6x | FA3 摘要/§5 |

共同主题：**每一步都在压 HBM 访问、提硬件利用率，而不是改注意力数学**——三版都是精确算法。

---

## 12. 衔接：decode 侧、FP8 注意力与量化系列

### 12.1 Decode 形态（FlashDecoding）

decode 时 query 只有 1 行、key 有 $L$ 行，$L$ 很长而 batch 小，GPU 并行度不够。FlashDecoding 把 KV **按行切块**并行算部分输出，再做一次 online-softmax 式合并——复用本章的数学，把 decode 的长序列注意力也并行化。

### 12.2 与量化系列的呼应

01 章说过"注意力是量化最敏感的区域之一"；FA3 给出了低精度注意力的两条工程经验：

```
块级量化（per-block scale）比 per-tensor 更稳
incoherent processing（打散离群值）能把 FP8 注意力误差再降一截
```

这与量化系列的"粒度 + 离群值"主线完全一致——**Attention 内核与量化在精度策略上共享同一套语言**。

---

## 13. 本章小结

1. **瓶颈是 IO**：朴素注意力的 $N^2$ 矩阵在 HBM 反复搬移；FA 的 IO 复杂度 $O(N^2d^2/M)$ vs 标准 $\Omega(Nd+N^2)$。
2. **三个技巧**：tiling（分块塞 SRAM）、online softmax（增量归一化，精确）、recompute（反向重算换带宽）。
3. **精确性**：FlashAttention 的结果与朴素 softmax 逐位等价（浮点舍入内），不是近似算法。
4. **演进**：FA1 提出 IO 感知框架（2–4x）；FA2 修工作划分（再 2x、73% 峰值）；FA3 吃 Hopper 硬件（再 1.5–2x、FP8）。
5. **定理**：$O(N)$ 额外显存、$O(N^2d^2/M)$ HBM 访问，且对精确注意力是最优的。

> 一句话记忆：**"不要造一张 N×N 的纸（中间矩阵），把它剪成能放进口袋（SRAM）的碎片，边算边记账（online softmax），反向时再撕一遍（recompute）——纸没了，账一分不少。"**

---

## 14. 习题与解答

### 题 1（推导）：online softmax 保持精确

证明：若第 $j$ 块前 $m^{(j)}, \ell^{(j)}, \mathbf{O}^{(j)}$ 满足 $\ell^{(j)} = \sum_{k \le j}\exp(s_k - m^{(j)})$、$\mathbf{O}^{(j)} = \mathrm{softmax}(s_{1:j})\mathbf{V}_{1:j}$，则第 5 节的更新公式使相同性质对 $j+1$ 成立。

<details>
<summary>题 1 解答</summary>

$\ell^{(j+1)} = e^{m^{(j)}-m^{(j+1)}}\ell^{(j)} + e^{\tilde{m}-m^{(j+1)}}\tilde{\ell} = \sum_{k\le j}\exp(s_k - m^{(j+1)}) + \sum_{k>j}\exp(s_k - m^{(j+1)})$，即全行以 $m^{(j+1)}$ 为基准的指数和。$\mathbf{O}^{(j+1)}$ 分子 $= e^{m^{(j)}-m^{(j+1)}}\ell^{(j)}\mathbf{O}^{(j)} + e^{\tilde{m}-m^{(j+1)}}\tilde{\mathbf{P}}\mathbf{V} = \sum_{k\le j+1}\exp(s_k - m^{(j+1)})v_k$，除以 $\ell^{(j+1)}$ 即 softmax 加权和。归纳成立。
</details>

### 题 2（计算）：重做第 8 节算例

把 $s = (1.414, 0.707, 0.707)$ 按"块 1 = 第一个分数、块 2 = 后两个"切分，重新走一遍 online softmax，验证结果仍是 0.752。

<details>
<summary>题 2 解答</summary>

块 1（单个分数 $1.414$）：$\tilde{m}=1.414$，$\tilde{\mathbf{P}}=\exp(1.414-1.414)=(1)$，$\tilde{\ell}=1$。更新：$m^{(1)}=1.414$，$\ell^{(1)} = e^{-\infty-1.414}\cdot0 + e^{0}\cdot1 = 1$，$O^{(1)} = (0 + 1\cdot 1\cdot v_1)/1 = v_1 = 1$。

块 2（两个分数 $0.707, 0.707$）：$\tilde{m}=0.707$，$\tilde{\mathbf{P}}=\exp(0.707-0.707)=(1,1)$，$\tilde{\ell}=2$。更新：

$$
m^{(2)} = \max(1.414, 0.707) = 1.414
$$

$$
\ell^{(2)} = e^{0}\cdot 1 + e^{0.707-1.414}\cdot 2 = 1 + 0.493\times 2 = 1.986
$$

$$
O^{(2)} = \frac{1\cdot 1\cdot 1 + e^{-0.707}\cdot (1\cdot v_2 + 1\cdot v_3)}{1.986}
= \frac{1 + 0.493\times(0+1)}{1.986} = \frac{1.493}{1.986} = 0.752
$$

与整体 softmax 一致 ✓。关键点：**块内 $\tilde{\mathbf{P}}$ 必须相对块内最大值 $\tilde{m}$ 计算**（这里是 $(1,1)$），合入时再统一缩放到全局基准 $m^{(2)}$。
</details>

### 题 3（推导）：块大小为什么是 M/4d

说明 $B_c = \lceil M/4d \rceil$ 的来源：SRAM 里同时驻留哪四项？

<details>
<summary>题 3 解答</summary>

同时驻留 $\mathbf{Q}_i$（$B_r d$）、$\mathbf{K}_j$（$B_c d$）、$\mathbf{V}_j$（$B_c d$）、$\mathbf{S}_{ij}$（$B_r B_c$）。取 $B_r = B_c = B$，四项约各占 $M/4$：$B d \approx M/4 \Rightarrow B \approx M/4d$。$B_r$ 额外限制不超过 $d$（Q 块不必比 head 维还宽）。
</details>

### 题 4（思考）：为什么"多算 FLOPs 反而更快"

FA 反向的 FLOPs 约为标准实现的两倍，却更快。用"IO 稀缺、FLOPs 富余"解释，并说明什么条件下这个结论会反转。

<details>
<summary>题 4 解答要点</summary>

反向重算省掉 $O(N^2)$ 矩阵的落盘与回读；HBM 访问次数是墙钟的主导项，FLOPs 只要没超硬件上限就不值钱。反转条件：$d$ 很大而 $M$ 很小（$d^2/M$ 接近 1），或设备算力本身成为瓶颈（如极低功耗设备）——此时重算的开销可能超过省下的 IO。
</details>

### 题 5（对比）：FA1/FA2/FA3 各自解决什么

用一句话分别总结 FA1、FA2、FA3 的核心贡献，并说明三版为什么都保持"精确"。

<details>
<summary>题 5 解答要点</summary>

FA1：提出 IO 感知框架（tiling + online softmax + recompute），把 HBM 访问从 $O(N^2)$ 降到 $O(N^2d^2/M)$；FA2：修 thread block/warp 的工作划分，把利用率从 30–50% 提到 73%；FA3：用 TMA/wgmma 的异步与 FP8 张量核，在 H100 上再提 1.5–2x。三版都不改注意力数学，只改计算的组织方式，所以都精确。
</details>

### 题 6（编程）：分块 softmax 验证

实现：① 朴素 softmax（整行）；② online softmax（按任意块大小切分）；③ 对比两者在随机输入（float32 与 float16）下的最大绝对误差；④ 画出误差随块大小 $B \in \{2, 8, 32, 128\}$ 的变化，并解释 float16 误差来源。

<details>
<summary>题 6 解答要点</summary>

①②输出应逐元素一致（float32 下误差 $\sim 10^{-7}$ 量级）；③ float16 下误差变大，主要来自 $e^{m-m^{new}}$ 缩放与累加的舍入；④ 块越大（越接近整行）误差越小——但即使块很小，online 版仍比"分块后各自独立 softmax"（错误做法）精确得多。这就是 FA 精度故事的代码级验证。
</details>

---

## 15. 延伸阅读

1. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（arXiv:2205.14135）](https://arxiv.org/abs/2205.14135)：本章算法（Algorithm 1）、定理 1/2、实验的出处。
2. [FlashAttention-2（arXiv:2307.08691）](https://arxiv.org/abs/2307.08691)：并行与工作划分的细节。
3. [FlashAttention-3（arXiv:2407.08608）](https://arxiv.org/abs/2407.08608)：Hopper 异步、FP8 注意力。
4. [FlashDecoding（Dao et al., 2023）](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)：decode 侧的并行化（12.1 节的展开）。
5. 上一篇：[01 注意力机制基础与复杂度分析](./LLM注意力内核精读笔记-01-注意力机制基础与复杂度分析.md)；下一篇：**03 注意力头变体：MQA / GQA / MLA**——把 KV cache 的显存账本从 $d$ 砍到 $d_{\text{kv}}$。
