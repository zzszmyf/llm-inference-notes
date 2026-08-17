# LLM 注意力与计算内核精读笔记 · 01 注意力机制基础与复杂度分析

> 对应：Vaswani et al., *Attention Is All You Need*（2017）；Inference Engineering Ch5 的 Attention 部分。
> 前置：量化系列 03 章（带宽模型）、推测解码系列 01 章（解码过程）。学完本章你应该能：① 写出缩放点积注意力的完整定义并解释 $\sqrt{d_{\text{head}}}$ 的作用；② 推导注意力的 FLOPs 公式 $4L^2 d$，并说明为什么与头数无关；③ 用 $L/(2d)$ 判据判断"长到什么程度注意力反超 FFN"；④ 推导 KV cache 的大小公式，手算 7B 模型 128K 上下文的显存；⑤ 说清 prefill 与 decode 两种注意力形态（计算密集 vs 访存密集）；⑥ 解释 softmax 为什么要做 max-subtraction，以及低精度下的风险。

---

## 目录（本章）

1. 本章目标
2. 注意力的位置：Transformer 里唯一的 token 交互通道
3. 形式化定义：缩放点积注意力
4. 复杂度分析：O(L²) 从哪来
5. 因果掩码与 KV Cache
6. Prefill 与 Decode：两种形态
7. 数值稳定性：max-subtraction 与低精度
8. 数值算例
9. 本章小结
10. 习题与解答
11. 延伸阅读

---

## 2. 注意力的位置：Transformer 里唯一的 token 交互通道

Transformer 的每一层由两类子层组成：

```
token-wise 子层（MLP、LayerNorm、激活）：每个 token 独立处理，互不通信
注意力子层：唯一让 token 之间交换信息的通道
```

这个结构决定了两个重要事实：

1. **序列长度 $L$ 只会让注意力变贵**：MLP 的 FLOPs 对 $L$ 是线性的，注意力的 FLOPs 是 $L^2$——长上下文的计算瓶颈一定先在注意力出现；
2. **token 的"记忆"都压在注意力上**：第 $t$ 个 token 要知道前面的任何信息，只能通过注意力去"看"前面 token 的表示。KV cache 之所以存在，正是为了省掉 decode 阶段重复计算这些"看"的键值。

---

## 3. 形式化定义：缩放点积注意力

给定输入序列的表示矩阵，先通过三个投影得到查询、键、值：

$$
\mathbf{Q} = X W_Q, \qquad \mathbf{K} = X W_K, \qquad \mathbf{V} = X W_V
$$

缩放点积注意力：

$$
\mathrm{Att}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_{\text{head}}}}\right)\mathbf{V}
$$

其中 $\mathbf{Q}, \mathbf{K} \in \mathbb{R}^{L \times d_{\text{head}}}$、$\mathbf{V} \in \mathbb{R}^{L \times d_{\text{head}}}$。逐位置写法：位置 $i$ 的输出是所有位置的值的加权平均，

$$
o_i = \sum_{j=1}^{L} a_{ij}\, v_j, \qquad
a_{ij} = \frac{\exp(s_{ij})}{\sum_{j'} \exp(s_{ij'})}, \qquad
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_{\text{head}}}}
$$

### 3.1 为什么要除以 $\sqrt{d_{\text{head}}}$

假设 $q, k$ 的各分量独立同分布、均值 0 方差 1，则点积的方差等于维度：

$$
\mathrm{Var}(q \cdot k) = d_{\text{head}}
$$

除以 $\sqrt{d_{\text{head}}}$ 后方差回到 1。若不缩放，维度越大点积数值越极端，softmax 越容易饱和（一个接近 1，其余接近 0），梯度消失。缩放是让注意力"温度"与维度无关。

---

## 4. 复杂度分析：O(L²) 从哪来

### 4.1 FLOPs 推导

标准实现（不缓存、不掩码优化）分三步：

$$
\mathbf{S} = \mathbf{Q}\mathbf{K}^\top / \sqrt{d_{\text{head}}}, \qquad
\mathbf{P} = \mathrm{softmax}(\mathbf{S}), \qquad
\mathbf{O} = \mathbf{P}\mathbf{V}
$$

FLOPs：

```
QK^T：L×d_head 与 d_head×L 相乘 → 2L²·d_head
softmax：L×L 行归一化 → O(L²)，常数项可忽略
PV：L×L 与 L×d_head 相乘 → 2L²·d_head
合计（每头）：≈ 4L²·d_head
```

全部 $h$ 头加起来：

$$
\text{FLOPs}_{\text{attn}} = h \times 4L^2 d_{\text{head}} = 4L^2 (h\, d_{\text{head}}) = 4L^2 d
$$

**结论：注意力 FLOPs 只依赖 $L$ 和 $d$，与切成多少个头无关**（$h\,d_{\text{head}} = d$ 是不变量）。切头是为了并行和表示能力，不是为了省算力。

### 4.2 与 FFN 的比值：长上下文判据

每层 FFN（两段线性，中间扩到 $4d$）的 FLOPs：

$$
\text{FLOPs}_{\text{FFN}} = 8L\,d^2
$$

（每个 token 两段各 $2 \times d \times 4d = 8d^2$ FLOPs，共 $L$ 个 token。）于是：

$$
\frac{\text{FLOPs}_{\text{attn}}}{\text{FLOPs}_{\text{FFN}}}
= \frac{4L^2 d}{8L d^2}
= \frac{L}{2d}
$$

**判据：$L = 2d$ 时两者相等；$L > 2d$ 时注意力反超 FFN，成为 prefill 的主要计算。** 例如 $d = 4096$ 的 7B 级模型：$L = 8192$ 持平，$L = 32K$ 时注意力是 FFN 的 4 倍。这解释了为什么长上下文优化的主战场是注意力（02–04 章）。

---

## 5. 因果掩码与 KV Cache

### 5.1 因果掩码

自回归模型只允许位置 $i$ 看 $j \le i$。实现上把注意力分数加上掩码矩阵：

$$
\mathbf{S} \leftarrow \mathbf{S} + \mathbf{M}, \qquad
M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$

注意：**naive 实现先算完整的 $L \times L$ 再掩码，上三角的 FLOPs 是白花的**（约占一半）；02 章 FlashAttention 用因果块只算下三角。

### 5.2 KV Cache：decode 为什么能省一半计算

解码第 $t$ 步时，注意力需要：

$$
q_t \quad \text{与所有历史} \quad (k_1, v_1), \dots, (k_{t-1}, v_{t-1})
$$

关键观察：**$k_j, v_j$ 只依赖 $x_1..x_j$，与未来的 query 无关**——所以算过一次就缓存，decode 时直接读取，不用重新前向。这就是 KV cache。

每步新算 $k_t, v_t$ 并追加进缓存。缓存大小：

$$
\text{KV bytes} = 2 \times L \times n_{\text{layers}} \times d_{\text{kv}} \times b
$$

其中因子 2 是 K 和 V 各一份，$b$ 是精度字节数（BF16 为 2）。

### 5.3 算例：7B 模型的 KV 显存

配置：$d = 4096$、32 层、$d_{\text{kv}} = 4096$（无 GQA）、BF16。

$$
\text{每 token} = 2 \times 4096 \times 2 \times 32 = 524{,}288\ \text{B} = 512\ \text{KB}
$$

| 上下文长度 | KV cache 总量 | 相对 7B 权重（约 13GB BF16） |
|---:|---:|---:|
| 4K | 2 GiB | 15% |
| 32K | 16 GiB | 1.2 倍 |
| 128K | 64 GiB | 5 倍 |

结论：**KV cache 是"上下文越长越贵"的显存项**，这正是 03 章 MQA/GQA/MLA 和 05 章 PagedAttention 要解决的问题。

---

## 6. Prefill 与 Decode：两种形态

同一个注意力，在两个阶段呈现完全不同的计算形态：

| | Prefill（处理用户输入） | Decode（逐 token 生成） |
|---|---|---|
| 一次处理的 token 数 | $L$（整段输入） | 1（新 query） |
| 注意力矩阵 | $L \times L$ | $1 \times L$ |
| 计算量 | $O(L^2)$，**计算密集** | $O(L)$ 次乘加，但要读全部 KV |
| 瓶颈 | 计算（FLOPs） | 带宽（KV 读取 + 权重读取） |
| 优化主力 | FlashAttention（02 章） | GQA/MLA（03 章）+ 分页（05 章） |

decode 每步要读的 KV 字节数：

$$
\text{每步 KV 读取} = 2 \times L \times n_{\text{layers}} \times d_{\text{kv}} \times b
$$

7B 模型算例：4K 上下文时每步读 $2 \times 4096 \times 32 \times 4096 \times 2 = 2\ \text{GiB}$（约 1ms @ 2TB/s），与权重读取（13GB，约 6.5ms）相比约 15%；**32K 上下文时每步读 16 GiB，超过权重读取**——长上下文 decode 的带宽瓶颈从"权重"转移到了"KV"。

---

## 7. 数值稳定性：max-subtraction 与低精度

### 7.1 朴素 softmax 的溢出问题

$\exp(x)$ 在 $x$ 稍大时就溢出（FP16 最大约 65504，对应 $x \approx 11$）。数值稳定写法：

$$
\mathrm{softmax}(s)_i = \frac{\exp(s_i - m)}{\sum_j \exp(s_j - m)}, \qquad m = \max_j s_j
$$

减去行最大值后，指数里最大是 0，不会溢出。02 章 FlashAttention 的 online softmax 就是这个技巧的"分块版"。

### 7.2 低精度下的风险

```
BF16：动态范围大，softmax 数值安全，但精度低（尾数 8 位）
FP16：动态范围小，softmax 容易溢出，需要 max-subtraction 和钳制
FP8：尾数更少（E4M3），注意力分数与概率的量化误差放大
```

经验：**注意力（尤其 softmax 区域）是量化最敏感的地方之一**（量化系列 08 章的结论）。低精度注意力必须做数值与质量双重验收（07 章展开）。

---

## 8. 数值算例

$L = 3$、$d_{\text{head}} = 2$：

$$
\mathbf{Q} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}, \quad
\mathbf{K} = \begin{pmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
\mathbf{V} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}, \quad
\sqrt{d_{\text{head}}} = \sqrt{2}
$$

**以第 3 个 query $q_3 = (1,1)$ 为例**：

$$
s = \frac{q_3 \cdot k_j}{\sqrt{2}} = \left(\frac{2}{\sqrt2}, \frac{1}{\sqrt2}, \frac{1}{\sqrt2}\right) = (1.414, 0.707, 0.707)
$$

max-subtraction：$m = 1.414$，

$$
s - m = (0,\ -0.707,\ -0.707), \qquad
\exp = (1,\ 0.493,\ 0.493), \qquad
\text{sum} = 1.986
$$

$$
a_3 = (0.504,\ 0.248,\ 0.248)
$$

输出：

$$
o_3 = 0.504 \cdot (1,0) + 0.248 \cdot (0,1) + 0.248 \cdot (1,1) = (0.752,\ 0.497)
$$

**验证**：位置 3 与位置 1 的相似度（$q_3 \cdot k_1 = 2$）最高，所以 $v_1 = (1,0)$ 在输出里占比最大（0.504）——注意力是"按相关性加权求和"。

完整三行输出（留给读者核对习题 2）：

$$
o_1 \approx (0.802,\ 0.599), \qquad
o_2 \approx (0.599,\ 0.599), \qquad
o_3 \approx (0.752,\ 0.497)
$$

---

## 9. 本章小结

1. **注意力是唯一 $O(L^2)$ 的组件**：FLOPs $= 4L^2d$，与头数无关；$L/(2d)$ 判据决定它何时反超 FFN。
2. **KV cache 来自"键值只依赖历史"**：大小 $= 2Ln_{\text{layers}}d_{\text{kv}}b$，7B 模型 128K 上下文约 64 GiB。
3. **prefill 计算密集、decode 访存密集**：长上下文时 decode 的带宽瓶颈从权重转移到 KV。
4. **softmax 需要 max-subtraction**；低精度下注意力是质量风险最高的区域之一。

> 一句话记忆：**"注意力是 Transformer 里唯一会让 token 互相看的地方，也是唯一随序列长度平方变贵的地方——KV cache 是 decode 的账本，$L^2$ 是 prefill 的账单。"**

---

## 10. 习题与解答

### 题 1（推导）：FLOPs 与头数无关

从每头 $4L^2 d_{\text{head}}$ 出发，推导多头注意力总 FLOPs $= 4L^2 d$，并说明为什么切头不省算力。

<details>
<summary>题 1 解答</summary>

多头 = $h$ 个独立的 $d_{\text{head}}$ 维注意力并行，总 FLOPs $= h \times 4L^2 d_{\text{head}} = 4L^2 (h d_{\text{head}}) = 4L^2 d$。$h\,d_{\text{head}} = d$ 是投影后的总维度不变式，所以切头只改并行方式，不改 FLOPs。
</details>

### 题 2（计算）：手算完整一行

用第 8 节的 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$，完整手算 $o_1$（含 max-subtraction），核对 $o_1 \approx (0.802, 0.599)$。

<details>
<summary>题 2 解答</summary>

$q_1=(1,0)$：$s = (1/\sqrt2,\ 0,\ 1/\sqrt2) = (0.707, 0, 0.707)$；max $= 0.707$；$s-m = (0, -0.707, 0)$（注意第三个位置 $0.707-0.707=0$）；$\exp = (1, 0.493, 1)$；和 $= 2.493$；$a_1 = (0.401, 0.198, 0.401)$。$o_1 = 0.401(1,0) + 0.198(0,1) + 0.401(1,1) = (0.802, 0.599)$ ✓ 与第 8 节一致。
</details>

### 题 3（推导）：长上下文判据

证明注意力与 FFN 的 FLOPs 比为 $L/(2d)$；给定 $d = 8192$，求两者持平的 $L^*$，并说明 $L = 64K$ 时注意力占比。

<details>
<summary>题 3 解答</summary>

比值 $= 4L^2d / 8Ld^2 = L/(2d)$。$L^* = 2d = 16384$。$L=64K$ 时比值 $= 65536/16384 = 4$，即注意力 FLOPs 是 FFN 的 4 倍、占总计算（attention + FFN）的 $4/5 = 80\%$。
</details>

### 题 4（计算）：KV cache 账本

配置：$d = 5120$、48 层、$d_{\text{kv}} = 5120$、FP8（1 字节）。求：① 每 token KV 字节数；② $L = 128K$ 的总显存；③ 若改用 GQA 使 $d_{\text{kv}} = 640$，同样长度下省多少。

<details>
<summary>题 4 解答</summary>

① $2 \times 5120 \times 48 \times 1 = 491{,}520\ \text{B} = 480\ \text{KB/token}$。② $480\ \text{KB} \times 131072 = 60\ \text{GiB}$。③ $d_{\text{kv}}=640$ 时 $2 \times 640 \times 48 = 61{,}440\ \text{B} = 60\ \text{KB/token}$，总量 7.5 GiB——**GQA 把 KV 显存降到原来的 1/8**（03 章详述）。
</details>

### 题 5（思考）：为什么 decode 是访存密集

结合量化系列 03 章的带宽模型，解释 decode 每步"读权重 + 读全部 KV"为什么是瓶颈；长上下文下 KV 读取为何会反超权重读取。

<details>
<summary>题 5 解答要点</summary>

decode 每步只产出 1 个 token，但必须搬全部权重（固定 13GB）和全部 KV（随 $L$ 线性增长：7B 在 4K 是 2GB、32K 是 16GB）。GPU 的 FLOPs 远用不满，瓶颈在 HBM 带宽。$L$ 增大到 KV 读取 ≥ 权重读取时，优化重点从"省权重带宽"（量化）转向"省 KV 带宽/显存"（GQA、分页、稀疏）。
</details>

### 题 6（编程）：causal attention 实现

实现：① naive 全矩阵 + 掩码；② 只算下三角的因果版；③ 数值稳定 softmax（max-subtraction）。验证两种实现在随机输入下输出一致，并统计 $L = 256, 1024, 4096$ 时 FLOPs/显存/时间的增长是否近似 $O(L^2)$。

<details>
<summary>题 6 解答要点</summary>

① 先算 $\mathbf{S}$，加 $\mathbf{M}$（上三角 $-\infty$），再 softmax、乘 V。② 循环逐行或块状只算 $j \le i$。③ 每行减行最大。三者对因果输出一致。计时/显存随 $L$ 增长约 4 倍（$L$ 翻倍）即 $O(L^2)$；到很大 $L$ 时显存先爆——这正是 02 章 FlashAttention 的动机。
</details>

---

## 11. 延伸阅读

1. [Attention Is All You Need（arXiv:1706.03762）](https://arxiv.org/abs/1706.03762)：本章公式出处（缩放点积注意力、多头、因果掩码）。
2. [FlashAttention（arXiv:2205.14135）](https://arxiv.org/abs/2205.14135)：下一篇的主角——为什么 $O(L^2)$ 的中间矩阵可以不落显存。
3. [量化系列 03 章（数值格式与硬件）](./LLM量化精读笔记-03-数值格式与硬件.md)：带宽模型与本系列 decode 分析互相印证。
4. [Inference Engineering Ch5](https://inferenceengineering.tech/chapters/techniques/)：教材正文（Attention 与系统部分）。
5. 下一篇：**02 FlashAttention：IO 感知的精确注意力**——把 max-subtraction 变成 online softmax，把 $L^2$ 矩阵留在片上。
