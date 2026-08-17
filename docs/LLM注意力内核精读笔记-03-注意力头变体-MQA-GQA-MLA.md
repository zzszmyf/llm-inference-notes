# LLM 注意力与计算内核精读笔记 · 03 注意力头变体：MQA / GQA / MLA

> 对应：Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*（arXiv:1911.02150，2019）；Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*（arXiv:2305.13245，EMNLP 2023）；DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*（arXiv:2405.04434，2024）。
> 前置：01 章（KV cache 大小公式 $2Ln_{\text{layers}}d_{\text{kv}}b$）、02 章（prefill/decode 两种形态）。学完本章你应该能：① 写清 MQA/GQA/MLA 三种方案"每 token 缓存什么"的数学定义与 KV 元素数公式；② 解释 GQA 的 up-training 配方（mean-pool + 5% 预训练算力）；③ 推导 MLA 的低秩联合压缩公式，并解释 RoPE 与低秩压缩的冲突及 decoupled RoPE 解法；④ 手算 70B 与 DeepSeek-V2 的 KV 显存账本；⑤ 说出"KV 缓存压缩谱系"的统一视角。

---

## 目录（本章）

1. 本章目标
2. 问题重述：decode 的 KV 带宽账本
3. MQA：一个 KV 头管所有 query 头
4. GQA：中间地带与 Uptraining
5. MLA：低秩 KV 联合压缩
6. 统一视角：KV 缓存压缩谱系
7. 数值算例：三种方案的显存账本
8. 质量与工程权衡
9. 本章小结
10. 习题与解答
11. 延伸阅读

---

## 2. 问题重述：decode 的 KV 带宽账本

01 章的结论：decode 每步要读全部 KV，字节数

$$
\text{每步 KV 读取} = 2 \times L \times n_{\text{layers}} \times d_{\text{kv}} \times b
$$

$d_{\text{kv}}$ 是 KV 总维度。对 MHA（每头独立 K/V）：

$$
d_{\text{kv}} = h \times d_{\text{head}}
$$

长上下文时这一步的带宽会反超权重读取（01 章 32K 算例）。本章的三个方案都在做同一件事：**把 $d_{\text{kv}}$ 变小**。区别只在于"怎么变"：

```
MQA：h 个 KV 头 → 1 个（共享）
GQA：h 个 KV 头 → g 个（分组共享）
MLA：干脆不缓存 K/V，缓存一个低秩潜向量，需要时再展开
```

---

## 3. MQA：一个 KV 头管所有 query 头

### 3.1 定义

Multi-Query Attention（Shazeer 2019）：所有 query 头**共享同一组 K、V**。

$$
d_{\text{kv}} = d_{\text{head}}, \qquad
\text{KV 元素/token/层} = 2\, d_{\text{head}}
$$

相比 MHA（$2h\,d_{\text{head}}$），KV cache 缩小 $h$ 倍。

### 3.2 为什么省的是带宽

Shazeer 的性能分析：decode 每步的"访存:计算"比约为

$$
\Theta\!\left(\frac{1}{d} + \frac{n}{d h} + \frac{1}{b}\right)
$$

其中 $n$ 是序列长度、$b$ 是 batch。中间的 $n/(dh)$ 项正来自"逐位置读 KV"；MQA 把它除以 $h$，把"序列越长越贵"的项直接砍掉一个数量级。

### 3.3 代价

```
优点：KV 显存/带宽 ÷ h；训练也更快（KV 投影计算减少）
缺点：单组 K/V 容量不足，质量退化（尤其在翻译、摘要等任务上）；
     训练不稳定（GQA 论文明确指出）
```

因此 MQA 适合"已经训好的模型想快速换推理"，但训练新模型时它通常不是最优解。

---

## 4. GQA：中间地带与 Uptraining

### 4.1 定义

Grouped-Query Attention（Ainslie et al. 2023）：把 $h$ 个 query 头分成 $g$ 组，**每组共享一个 KV 头**：

$$
d_{\text{kv}} = g \times d_{\text{head}}, \qquad
\text{KV 元素/token/层} = 2\, g\, d_{\text{head}}
$$

两个端点：

$$
\text{GQA-}1 = \text{MQA}, \qquad \text{GQA-}h = \text{MHA}
$$

论文实验（T5 XXL 上对 5 个摘要集、WMT 翻译、TriviaQA 的评测）：**GQA 质量接近 MHA、速度接近 MQA**——中间组数是最优权衡。

### 4.2 Uptraining：从 MHA 检查点转换

不重新训练也能拿到 GQA/MQA 模型：

```
步骤 1：把 K/V 投影按组做 mean-pooling（比选单头或随机初始化好）
步骤 2：用原预训练配方继续训练 α = 5% 的步数（T5 XXL 约 600 TPUv3 chip-days）
```

于是"已有 MHA 权重 + 5% 算力"就能得到可部署的 GQA 模型。

### 4.3 为什么大模型更划算

$$
\text{KV cache} \propto d, \qquad \text{FLOPs} \propto d^2
$$

模型越大，KV 相对 FLOPs 越便宜、头越多，MQA 的"一刀切"越激进；GQA 让带宽削减与模型规模保持同比例。此外，大模型张量并行时每个 KV 头会被复制到每个 partition，GQA 恰好消除这种浪费。所以工业界大模型几乎都选了 GQA：

| 模型 | query 头 | KV 头（组） | 类型 |
|---|---:|---:|---|
| LLaMA-2 7B/13B | 32 | 32 | MHA |
| LLaMA-2 70B | 64 | 8 | GQA-8 |
| LLaMA-3 8B | 32 | 8 | GQA-8（4 组） |
| Mistral 7B | 32 | 8 | GQA-8 |
| Falcon 40B | 128 | 8 | MQA |

---

## 5. MLA：低秩 KV 联合压缩

GQA/MQA 是"砍头数"；DeepSeek-V2 的 MLA（Multi-head Latent Attention）更进一步：**把每层的 K、V 压进一个低秩潜向量，推理只缓存这个向量**。

### 5.1 定义

设第 $t$ 个 token 的注意力输入为 $h_t$。MLA 用三个低秩投影：

$$
c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}, \qquad
k_t = W^{UK} c_t^{KV}, \qquad
v_t = W^{UV} c_t^{KV}
$$

其中 $d_c \ll h\,d_{\text{head}}$。**推理时只缓存 $c_t^{KV}$**（$d_c$ 维），K、V 按需从潜向量展开——展开所需的 $W^{UK}$、$W^{UV}$ 在推理时被吸收进 $W^Q$ 的左右两侧，不增加每步计算。

Query 也做低秩压缩（$c_t^Q = W^{DQ} h_t$，$q_t = W^{UQ} c_t^Q$），目的是省训练时的激活显存——不省 KV。

### 5.2 RoPE 的冲突与 decoupled RoPE

旋转位置编码（RoPE）对 K、Q 都施加一个**随位置变化**的矩阵。若对展开后的 $k_t$ 施加 RoPE，那么"$W^{UK}$ 之后接 RoPE 矩阵"就无法再吸收进 $W^Q$（矩阵乘法不可交换，位置相关的矩阵卡在中间）。

解法：**decoupled RoPE**——位置信息不走潜向量，而是由一组额外的低维头承载：

$$
q_{t,i}^R = \mathrm{RoPE}(W^{QR} c_t^Q), \qquad
k_t^R = \mathrm{RoPE}(W^{KR} h_t)
$$

于是每 token 缓存的 KV = 潜向量 $d_c$ + 共享 RoPE key $d_h^R$：

$$
\text{MLA KV 元素/token/层} = d_c + d_h^R
$$

### 5.3 DeepSeek-V2 的配置与收益

$$
d_c = 512,\quad d_h^R = 64 \quad\Rightarrow\quad \text{每层每 token } 576 \text{ 个元素}
$$

论文对照表（每 token KV 元素数）：

| 机制 | KV 元素/token/层 | 容量 |
|---|---|---|
| MHA | $2\, n_h d_h$ | 强 |
| GQA | $2\, n_g d_h$ | 中等 |
| MQA | $2\, d_h$ | 弱 |
| MLA | $d_c + d_h^R \approx \frac{9}{2}d_h$ | **更强** |

DeepSeek-V2 的 MLA 等价于"只有 2.25 组的 GQA"，但论文报告其效果**超过 MHA**。整模型收益：相比 DeepSeek 67B，**KV cache 减少 93.3%，生成吞吐提升 5.76 倍，训练成本省 42.5%**。

---

## 6. 统一视角：KV 缓存压缩谱系

四个方案本质上是"每 token 存什么"的选择：

```
MHA：存 h 份 (k, v)             → 2 h d_h 元素，容量最强
GQA：存 g 份 (k, v)             → 2 g d_h 元素，容量中等
MQA：存 1 份 (k, v)             → 2 d_h 元素，容量弱
MLA：存 1 个低秩潜向量 + 位置头   → d_c + d_h^R 元素，容量反超
```

MLA 的洞察：**"存 K/V" 只是"存它的压缩表示"的特例**——GQA/MQA 是在"原始表示"里砍头数，MLA 是学一个更紧的表示。表示维度越小，容量损失越小，因为压缩是学习出来的而不是截断。

---

## 7. 数值算例：三种方案的显存账本

### 7.1 LLaMA-2 70B（GQA）

配置：80 层、$d = 8192$、64 个 query 头、8 个 KV 头、$d_{\text{head}} = 128$、BF16。

$$
\text{GQA：} 2 \times 80 \times 8 \times 128 \times 2 = 320\ \text{KB/token}
$$

$$
\text{若换成 MHA：} 2 \times 80 \times 64 \times 128 \times 2 = 2.5\ \text{MiB/token}
$$

| 上下文 | GQA KV | 等价 MHA KV |
|---:|---:|---:|
| 4K | 1.25 GiB | 10 GiB |
| 32K | 10 GiB | 80 GiB |

### 7.2 DeepSeek-V2（MLA）

配置：60 层、$d_c = 512$、$d_h^R = 64$、BF16。

$$
\text{MLA：} (512 + 64) \times 60 \times 2 = 67.5\ \text{KiB/token}
$$

$$
\text{等价 MHA：} 2 \times 128 \times 128 \times 60 \times 2 = 3.75\ \text{MiB/token}
$$

$$
\text{32K 上下文：} \text{MLA } 2.1\ \text{GiB} \quad\text{vs}\quad \text{MHA } 120\ \text{GiB} \quad(\approx 57\times)
$$

---

## 8. 质量与工程权衡

```
质量：MLA ≳ MHA > GQA ≳ MQA（论文各自报告，具体任务有波动）
速度：MLA ≈ MQA > GQA > MHA
实现复杂度：MLA 最高（decoupled RoPE、投影吸收、训练 RMSNorm 处理）
生态：GQA 是当前开源主流（LLaMA-2/3、Mistral）；MLA 是前沿大模型趋势（DeepSeek 系列）
```

工程决策：

```
已有 MHA 权重、想快速提速 → up-training 成 GQA（5% 算力）
训练新模型、极致省 KV → MLA（但要处理 RoPE 与数值稳定性）
显存预算固定 → 先算 01 章 KV 账本，再决定砍到哪一档
```

---

## 9. 本章小结

1. **MQA**：$h$ 个 KV 头 → 1 个，KV ÷ h，质量退化。
2. **GQA**：$h$ → $g$ 个，质量近 MHA、速度近 MQA；可用 mean-pool + 5% 算力从 MHA 转换。
3. **MLA**：缓存低秩潜向量 $c^{KV}$（$d_c$ 维）+ decoupled RoPE 位置头（$d_h^R$ 维）；DeepSeek-V2 每层每 token 仅 576 元素，KV 减少 93.3%、吞吐 5.76x。
4. **统一视角**：四者都是"每 token 存什么"的谱系；MLA 是学出来的压缩，容量损失最小。

> 一句话记忆：**"MHA 给每个头都配 K/V，MQA 让所有头共用一副，GQA 按组共用，MLA 干脆只存一张'底片'（潜向量），要用时再冲印成 K/V。"**

---

## 10. 习题与解答

### 题 1（推导）：KV 元素数公式

写出 MHA、GQA-g、MQA、MLA 每 token 每层 KV 元素数公式，并说明各自对应 $g$ 的取值。

<details>
<summary>题 1 解答</summary>

MHA：$2h d_h$（$g=h$）；GQA-g：$2g d_h$；MQA：$2d_h$（$g=1$）；MLA：$d_c + d_h^R$。GQA 的 $g$ 是 query 头分组数，$1 \le g \le h$，两个端点分别退化为 MQA 与 MHA。
</details>

### 题 2（推导）：MQA 的访存:计算比

从 Shazeer 的 $\Theta(1/d + n/(dh) + 1/b)$ 出发，说明 MQA 把哪一项缩小了多少倍，以及为什么"序列越长 MQA 相对 MHA 越划算"。

<details>
<summary>题 2 解答</summary>

中间项 $n/(dh)$ 来自逐位置读 KV，MQA 把 $d_{\text{kv}}$ 从 $hd_h$ 缩到 $d_h$，该项除以 $h$。$n$ 越大该项占比越高，MQA 的收益越大——长上下文下 MQA/GQA 的优势被放大。
</details>

### 题 3（计算）：GQA 组数换算

LLaMA-2 70B 有 64 个 query 头、8 个 KV 头：① 这是 GQA-几？② 每个 KV 头服务几个 query 头？③ KV cache 是等价 MHA 的几分之一？

<details>
<summary>题 3 解答</summary>

① GQA-8（8 组）；② $64/8 = 8$ 个 query 头共用 1 个 KV 头；③ $2\times8\times128 / 2\times64\times128 = 1/8$。
</details>

### 题 4（思考）：RoPE 为什么与低秩 KV 压缩冲突

解释"RoPE 矩阵插在 $W^{UK}$ 之后会阻止投影吸收"的数学原因，并说明 decoupled RoPE 为什么能绕开。

<details>
<summary>题 4 解答要点</summary>

推理时注意力分数是 $q^\top k$；若 $k = \mathrm{RoPE}(W^{UK}c)$，则分数里出现 $q^\top R\, W^{UK} c$。要把 $W^{UK}$ 吸进 $q$ 需要 $\tilde{q} = (W^{UK})^\top q$，但 $R$（随生成位置变化）夹在中间且矩阵乘法不可交换，无法统一吸收。decoupled RoPE 把位置信息放到独立的 $q^R, k^R$ 上，潜向量分支保持"无位置"的纯低秩形式，两分支各自可吸收。
</details>

### 题 5（计算）：DeepSeek-V2 KV 账本

DeepSeek-V2：60 层、$d_c=512$、$d_h^R=64$、BF16。① 每 token KV 字节数；② 32K 上下文总量；③ 与等价 MHA 相比是几分之一。

<details>
<summary>题 5 解答</summary>

① $(512+64)\times60\times2 = 69{,}120\ \text{B} = 67.5\ \text{KiB}$；② $67.5\ \text{KiB}\times32768 = 2.1\ \text{GiB}$；③ 等价 MHA $= 2\times128\times128\times60\times2 = 3.75\ \text{MiB/token}$，总 $120\ \text{GiB}$——MLA 约为其 $1/57$。
</details>

### 题 6（编程）：KV 尺寸函数

实现函数 `kv_bytes_per_token(mechanism, h, g, d_h, d_c, d_r, layers, bytes)`，对 MHA/GQA/MQA/MLA 分别返回每 token KV 字节数；画一条"上下文长度 vs KV 总量"的曲线（4 条线），验证 MLA 在长上下文下斜率最低。

<details>
<summary>题 6 解答要点</summary>

按题 1 公式实现；MLA 的每 token 尺寸与 $h$ 无关（只依赖 $d_c+d_h^R$），所以随上下文增长最慢；GQA 次之，MHA 最快。画到 128K 时 MHA 曲线会"起飞"，MLA 仍在可管理范围——这就是 03 章的全部动机。
</details>

---

## 11. 延伸阅读

1. [Fast Transformer Decoding: One Write-Head is All You Need（arXiv:1911.02150）](https://arxiv.org/abs/1911.02150)：MQA 的原始定义与带宽分析。
2. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints（arXiv:2305.13245）](https://arxiv.org/abs/2305.13245)：GQA 定义、up-training 配方、质量/速度实验。
3. [DeepSeek-V2（arXiv:2405.04434）](https://arxiv.org/abs/2405.04434)：MLA 公式、decoupled RoPE、配置（§2.1.3）与 KV 对照表（Table 1）。
4. [DeepSeek-V3（arXiv:2412.19437）](https://arxiv.org/abs/2412.19437)：MLA 的后续工程化（不压缩 Q 的变体），了解 MLA 演进。
5. 上一篇：[02 FlashAttention](./LLM注意力内核精读笔记-02-FlashAttention-IO感知的精确注意力.md)；下一篇：**04 稀疏、滑动窗口与线性注意力**——不砍 KV 维度，而是"少看"——把 $O(L^2)$ 的注意力本身变便宜。
