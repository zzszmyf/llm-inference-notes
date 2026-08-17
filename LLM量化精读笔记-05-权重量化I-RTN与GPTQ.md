# LLM 量化精读笔记 · 05 权重量化 I：RTN 与 GPTQ（含二阶误差补偿推导）

> 对应：GPTQ 论文（arXiv:2210.17323，ICLR 2023）；OBQ（Frantar & Alistarh 2022）；OBS（Hassibi et al. 1993）；MIT 6.5940 Lecture 5。
> 学完本章你应该能：① 说明 RTN 为什么在 4-bit 会翻车；② 从"最小化逐元素误差"升级到"最小化层输出误差"；③ 完整推导 OBS/OBQ 的最优补偿公式$\delta F = -(w_{q} - \hat{w}_q)/[H^{-1}]_{qq} \cdot H^{-1}_{:,q}$；④ 说清 GPTQ 的三个工程化观察（顺序无关、行并行、Cholesky）及其复杂度含义；⑤ 手算一个 2×2 的补偿例子。

---

## 目录（本章）

1. [本章目标](#1-本章目标)
2. [问题设定：weight-only 量化](#2-问题设定weight-only-量化)
3. [RTN：最朴素的基线](#3-rtn最朴素的基线)
4. [目标函数升级：从逐元素误差到输出误差](#4-目标函数升级从逐元素误差到输出误差)
5. [OBS：删除一个权重的代价](#5-obs删除一个权重的代价)
6. [OBQ：把"删除"换成"量化"](#6-obq把删除换成量化)
7. [GPTQ：把 OBQ 规模化](#7-gptq把-obq-规模化)
8. [手算例：补偿是怎么发生的](#8-手算例补偿是怎么发生的)
9. [GPTQ 的局限与后续](#9-gptq-的局限与后续)
10. [本章小结](#10-本章小结)
11. [习题与解答](#11-习题与解答)
12. [延伸阅读](#12-延伸阅读)

---

## 1. 本章目标

04 章我们学会了"选 scale、防 outlier"——那本质还是**每个权重独立四舍五入**（RTN）。本章回答一个更聪明的问题：

> **量化必然引入误差；能不能让"还没被量化的权重"去补偿"已经被量化权重"的误差？**

这就是 OBQ/GPTQ 的核心思想：**量化一个、全局补偿**。理解这一章，就理解了一半的 PTQ 文献（AWQ、QuIP# 都是在"补偿策略"上做文章）。

---

## 2. 问题设定：weight-only 量化

### 2.1 场景

权重量化（W4A16 / W4A8）只压缩权重，激活保持高精度：

存储/带宽收益：权重位宽减半$\to$模型文件减半、decode 搬运减半（03 章模型）
计算收益：解码时仍可与低精度激活配合（W4A8），或权重复用高精度激活（W4A16）

这也是推理服务最常见的起点：**先量化权重，因为权重最不敏感、收益最大、风险最小**。

### 2.2 形式化

设一层权重$W \in \mathbb{R}^{d_{\text{row}} \times d_{\text{col}}}$，输入激活$X \in \mathbb{R}^{d_{\text{col}} \times N}$（N 个校准样本），量化后 $\hat{W}$。目标：

$$
\min \|WX - \hat{W}X\|^{2}_F
$$
约束：$\hat{W}$ 的每个元素属于量化网格（如 INT4 + group scale）

注意目标函数里**没有单独惩罚逐元素误差**——我们关心的是层输出（进而整个模型）的误差。

---

## 3. RTN：最朴素的基线

### 3.1 定义

RTN（Round-to-Nearest）：对每个权重独立做"就近舍入到量化网格"：

$$
\hat{W}_ij = \operatorname{clamp}(\operatorname{round}(W_{\text{ij}} / s_{\text{channel}}), q_{\min}, q_{\max}) \times s_{\text{channel}}
$$

scale 可以是 per-channel 或 per-group（04 章）。实现成本几乎为零。

### 3.2 为什么 4-bit 会翻车

RTN 的三个盲区：

1. **忽略分布形状**：只保证每个权重离自己最近的网格点，不保证整体输出误差小。
2. **忽略层间作用**：每层独立量化，误差沿层累积（02 章）。
3. **ignoring outlier 关联**：RTN 不利用"哪些通道重要"（04 章的激活幅度）和"哪些权重动起来代价小"（Hessian）。

实验事实：$FP16 \to INT8 RTN$通常无损；INT4 RTN 在 7B 以上模型明显掉点；INT3 以下基本不可用。**RTN 是天花板最低、地板也最低的方法。**

---

## 4. 目标函数升级：从逐元素误差到输出误差

考虑量化单个权重$w_{q}$（W 的第 q 列）对输出的影响。设$L(W) = \frac{1}{2}\|WX - \hat{W}X\|^{2}_F$（$\frac{1}{2}$便于求导）。

把 L 在最优权重处做二阶泰勒展开（最优处梯度$g = 0$）：

$$
\Delta L \approx \frac{1}{2} \delta w^{T} H \delta w
$$

其中 H 是 Hessian。对矩阵乘的 MSE 目标：

$H = \partial ^{2}L/\partial w^{2} = X X^{T}$（对列向量形式的权重；$\frac{1}{2}$消掉 2）

GPTQ 论文记为$H = 2XX^{T}$（不写$\frac{1}{2}$），并加阻尼项：

$$
H = 2 X X^{T} + \lambda I, \lambda = 0.01 \times \operatorname{mean}(\operatorname{diag}(2XX^{T}))
$$

**H 的直觉**：H 编码了"输入各通道之间的相关性"。H 的对角线 = 每个输入通道的激活能量（谁更重要）；非对角线 = 通道间的相关（误差能否被别处补偿）。

> H 的维度是$d_{\text{col}} \times d_{\text{col}}$（按输入通道），不是参数个数。一层 12288 通道$\to H$约$12288^{2} \times 4 B \approx 600 MB$，可接受。

---

## 5. OBS：删除一个权重的代价

OBS（Optimal Brain Surgeon, Hassibi et al. 1993）问：**删掉一个权重，如何调整其余权重让损失恢复最多？**

设我们要把$w_{q}$设成目标值$\hat{w}_q$（删除= $\hat{w}_q = 0$），约束：

$$
e_{q}^{T} \delta w = \hat{w}_q - w_{q}
$$

其中$e_{q}$是第 q 个单位向量（$\delta w$只在第 q 个分量上取定值）。求：

$$
\begin{aligned}
min_\delta w \frac{1}{2} \delta w^{T} H \delta w
s.t. e_{q}^{T} \delta w &= \hat{w}_q - w_{q}
\end{aligned}
$$

用拉格朗日乘子法：

$$
\begin{aligned}
L &= \frac{1}{2} \delta w^{T} H \delta w + \lambda (e_{q}^{T} \delta w - (\hat{w}_q - w_{q}))
\partial L/\partial \delta w &= H \delta w + \lambda e_{q} = 0 \to \delta w = -\lambda H^{-1} e_{q}
\end{aligned}
$$
代入约束：$e_{q}^{T} \delta w = -\lambda [H^{-1}]_{qq} = \hat{w}_q - w_{q}$
$$
\to \lambda = -(\hat{w}_q - w_{q}) / [H^{-1}]_{qq}
$$

于是最优调整：

$$
\delta w = (\hat{w}_q - w_{q}) / [H^{-1}]_{qq} \times H^{-1}_{:,q}
$$

写成论文里的形式（只更新未处理权重 F）：

$$
\delta F = -(w_{q} - \hat{w}_q) / [H^{-1}]_{qq} \times H^{-1}_{:,q}
$$

对应的最小损失增量：

$$
\Delta L_{\text{min}} = \frac{1}{2} (w_{q} - \hat{w}_q)^{2} / [H^{-1}]_{qq}
$$

**三个关键结论**：

1. **补偿量正比于$H^{-1}$的第 q 列**：误差往"与 q 相关性强"的权重上摊。
2. **分母$[H^{-1}]_{qq}$越小，量化 q 的代价越大**：$H^{-1}$对角大 = 该权重"孤立且关键"，动它补不回来。
3. **$\Delta L$公式给了敏感性度量**：哪个权重量化最便宜，一目了然。

---

## 6. OBQ：把"删除"换成"量化"

OBQ（Optimal Brain Quantizer, Frantar & Alistarh 2022）把 OBS 的"删掉"换成"量化到$\hat{w}_q$"：

算法（逐权重）：
1. 计算$H^{-1}$（含阻尼$\lambda$）
2. 对每个待量化权重 q：
   a. 计算量化误差$\text{err}_q = w_{q} - \operatorname{quant}(w_{q})$
   b. 更新其余未量化权重：$\delta F = -\text{err}_q / [H^{-1}]_{qq} \times H^{-1}_{:,q}$
   c. 记录量化后的值
   d. 更新$H^{-1}$（去掉第 q 维，rank-1 修正）
3. 贪心顺序：每次选$\Delta L = \frac{1}{2} \text{err}_q^{2} / [H^{-1}]_{qq}$最小的 q

贪心顺序的含义：**先量化"代价最小"的权重，让后面的补偿空间最大化**。

复杂度：每量化一个权重要更新整列$H^{-1}$（$O(d_{\text{col}}^{2})$），共$d_{\text{row}} \times d_{\text{col}}$个权重$\to$**$O(d_{\text{row}} \cdot d_{\text{col}}^{3})$**。对$4096^{2}$的层都嫌慢，更别说$12288^{2}$。

---

## 7. GPTQ：把 OBQ 规模化

GPTQ（Frantar et al. 2023）基于三个观察把 OBQ 变成可落地的算法：

### 7.1 观察一：量化顺序几乎不影响结果

论文实验发现：对 LLM 权重，**贪心选序和固定顺序的结果几乎一样**。于是：

去掉贪心：按固定列顺序量化（如从左到右）
省掉：每次比较所有候选权重的$\Delta L$（$O(d_{\text{col}})$的排序开销）

### 7.2 观察二：行之间可以并行

补偿公式$\delta F = -\text{err}_q/[H^{-1}]_{qq} \times H^{-1}_{:,q}$里，**H 和$H^{-1}$只依赖输入 X，不依赖权重行**。所以 W 的$d_{\text{row}}$行共享同一套$H^{-1}$，各行独立量化、独立补偿，可以并行。

### 7.3 观察三：Cholesky 一次性分解

$H^{-1}$只需要**算一次**，并且用 Cholesky 分解$H^{-1} = LL^{T}$缓存起来；量化时按 **128 列一块**批量更新，避免逐列更新$H^{-1}$：

复杂度：$O(d_{\text{col}}^{3})$（一次 Cholesky）$+ O(d_{\text{row}} \cdot d_{\text{col}}^{2})$（批量补偿）
$$
\approx O(d_{\text{row}} \cdot d_{\text{col}}^{2})
$$
对比 OBQ：$O(d_{\text{row}} \cdot d_{\text{col}}^{3})$

### 7.4 GPTQ 伪代码（逐层）

```
输入：权重 W（d_row × d_col），校准激活 X（d_col × N），位宽 b，分组大小 G

for layer in model:
    H     = 2 X Xᵀ + λI                    # λ = 0.01·mean(diag(2XXᵀ))
    H_inv = cholesky_inverse(H)             # 一次分解，H⁻¹ = LLᵀ
    W_hat = W.clone()
    for block in range(0, d_col, 128):
        for q in block:                     # 固定顺序
            w_q = W[:, q]
            ŵ_q = quantize_group(w_q, b, G) # 分组 RTN
            err = w_q − ŵ_q
            W_hat[:, q] = ŵ_q
            # 补偿本块内剩余列
            W[:, block 中 q 之后] −= (err / H_inv[q,q])[:, None] × H_inv[q, 剩余列][None, :]
        # 用 Cholesky 因子批量补偿块之后的列
        W[:, block_end:] −= batch_update(W, W_hat, H_inv, block)
    # 替换层权重为 W_hat（连同 scale 一起保存）
```

（具体实现见 GPTQ 官方仓库；这里保留核心数学结构，块内/块间的 Cholesky 批量更新细节略去。）

### 7.5 数值与业界地位

- 一次量化 OPT-175B / BLOOM-176B 约 **4 GPU 小时**（A100）。
- **$INT4 + group=128$**：perplexity 与 FP16 差异在噪声范围附近（论文 Table 2；具体值因模型而异）。
- **INT3**：仍可用，但已有可感知退化；**INT2**：明显退化（催生了 QuIP#，06 章）。
- 落地：vLLM、TensorRT-LLM、HuggingFace（AutoGPTQ、GPTQModel）均有生产级 kernel。

---

## 8. 手算例：补偿是怎么发生的

设一层只有一行权重$W = [2.0, 1.0]$（$d_{\text{row}}=1$，$d_{\text{col}}=2$），校准激活：

$$
X = [[1.0, 0.5],
$$
     [0.5, 1.0]]        # 两个样本、两个输入通道，正相关

计算 H（用 GPTQ 的记法，忽略$\frac{1}{2}$）：

$$
\begin{aligned}
H &= 2 X X^{T} = 2 \times [[1.25, 1.0],
 [1.0, 1.25]] &= [[2.5, 2.0],
 [2.0, 2.5]]
H^{-1} &= (1/2.25) \times [[2.5, -2.0],
 [-2.0, 2.5]] &= [[1.111, -0.889],
 [-0.889, 1.111]]
\end{aligned}
$$

现在量化第 0 列：$w_0 = 2.0 \to \hat{w}_0 = 1.0$（假设网格$step=1$）：

$$
\begin{aligned}
err_0 &= 2.0 - 1.0 = 1.0
\delta w_1 &= -err_0 / [H^{-1}]_{00} \times [H^{-1}]_{01}
 &= -1.0 / 1.111 \times (-0.889) = +0.80
w_1 &= 1.0 + 0.80 = 1.80
\end{aligned}
$$

解读：$w_0$被量化小了 1.0，但输入通道 0 和 1 正相关（x₀ 大时 x₁ 通常也大），所以**把$w_1$调大 0.8 可以部分抵消输出的损失**。

该步的损失增量：

$$
\Delta L = \frac{1}{2} err_{0}^{2} / [H^{-1}]_{00} = \frac{1}{2} \times 1.0 / 1.111 \approx 0.45
$$

如果两个通道完全独立（H 对角），$[H^{-1}]_{01} = 0 \to \delta w_1 = 0$，补偿失效——**补偿能力来自输入通道之间的相关性**。

---

## 9. GPTQ 的局限与后续

### 9.1 局限

1. **二次假设**：目标函数只在最优附近近似二次；量化误差大（低 bit）时，补偿公式不再最优。
2. **校准集依赖**：H 来自校准数据；校准集分布偏了，H 就偏了。
3. **需要重建（reconstruction）**：逐层前向 + 线性代数，量一次要几分钟到几小时（远慢于 RTN 的几秒）。
4. **没有用激活幅度信息**：H 用到了激活的二阶统计，但 AWQ 证明"一阶幅度 + 简单缩放"在很多场景同样有效且更省。

### 9.2 方法谱系位置

```
RTN（零补偿，最简单）
  ↓ 加入二阶补偿
GPTQ（量化一个、全局补偿）
  ├─ AWQ（换成"激活幅度缩放"补偿，更简单，06 章）
  └─ QuIP#（先"洗牌"再量化，2-bit 天花板更高，06 章）
```

---

## 10. 本章小结

1. **RTN 是最朴素基线**：逐元素独立舍入，4-bit 翻车。
2. **目标函数**：$\min \|WX - \hat{W}X\|^{2}——$误差要看"输出"，不是"每个数"。
3. **二阶工具**：$H = 2XX^{T} + \lambda I$编码输入通道的相关性；$H^{-1}$决定补偿方向。
4. **OBS/OBQ 公式**：$\delta F = -(w_{q} - \hat{w}_q)/[H^{-1}]_{qq} \cdot H^{-1}_{:,q}$；$\Delta L = \frac{1}{2}(w_{q}-\hat{w}_q)^{2}/[H^{-1}]_{qq}$。
5. **GPTQ 三件套**：固定顺序（贪心没必要）+ 行并行（H 共享）+ Cholesky 批量更新$\to O(d_{\text{row}}\cdot d_{\text{col}}^{2})$，175B 模型 4 GPU 小时。
6. **补偿的本质**：利用输入通道相关性，用没量化的权重"背"已量化权重的误差。

> 一句话记忆：**"$GPTQ =$量化一个权重，让其他权重用 Hessian 告诉它的方向，把误差补回来；RTN 就是各扫门前雪。"**

---

## 11. 习题与解答

### 题 1（推导）：OBS 公式

重新推导$\delta F = -(w_{q} - \hat{w}_q)/[H^{-1}]_{qq} \times H^{-1}_{:,q}$，写出拉格朗日函数、驻点条件和$\lambda$的求解。

<details>
<summary>题 1 解答</summary>

$L = \frac{1}{2}\delta w^{T}H\delta w + \lambda (e_{q}^{T}\delta w - (\hat{w}_q-w_{q}))$。驻点：$H\delta w + \lambda e_{q} = 0 \to \delta w = -\lambda H^{-1}e_{q}$。约束：$-\lambda [H^{-1}]_{qq} = \hat{w}_q - w_{q} \to \lambda = -(\hat{w}_q-w_{q})/[H^{-1}]_{qq}$。代回：$\delta w = (\hat{w}_q-w_{q})/[H^{-1}]_{qq}\cdot H^{-1}_{:,q}$。写成$-(w_{q}-\hat{w}_q)/[H^{-1}]_{qq}\cdot H^{-1}_{:,q}$等价。
</details>

### 题 2（手算）：补偿方向

延续 8 节的例子，若激活改为**负相关**$X = [[1, -0.5], [-0.5, 1]]$，量化$w_0 = 2.0 \to 1.0$后，$w_1$应该怎么调？直觉上为什么？

<details>
<summary>题 2 解答</summary>

$H = 2[[1.25, -1],[-1, 1.25]]$，$H^{-1} = (1/2.25)[[2.5, 1],[1, 2.5]]$。$\delta w_1 = -1.0/1.111 \times 0.889 = -0.80 \to w_1 = 0.20$。
直觉：通道 0 与 1 负相关（x₀ 大时 x₁ 小），$w_0$被调小后，输出偏低主要发生在 x₀ 大的样本，此时 x₁ 小，所以$w_1$也要调小（而不是调大）才能匹配。
</details>

### 题 3（思考）：H 的 λ 阻尼

为什么 GPTQ 要在 H 的对角加$\lambda = 0.01\cdot \operatorname{mean}(\operatorname{diag}(H))$？

<details>
<summary>题 3 解答</summary>

校准数据里某些输入方向可能能量极低（近乎零方差），导致 H 奇异、$H^{-1}$爆炸；加阻尼项让 H 正定、数值稳定（ridge 回归的思路）。$\lambda$取对角线均值的 1% 是经验值：太小不起作用，太大扭曲补偿。
</details>

### 题 4（编程）：小规模 GPTQ 核心

实现 8 节的例子（$W=[2,1]$，X 正相关），并验证：量化 col0 后$w_1$的更新、$\Delta L$；再实现"无补偿"版本对比层输出误差 ‖WX−ŴX‖。

<details>
<summary>题 4 解答要点</summary>

有补偿：$\hat{W} = [1, 1.8]$，输出误差$\|(W-\hat{W})X\|^{2} = \|[1, -0.8]X\|^{2}$（X 两列），应显著小于无补偿$\hat{W}=[1,1]$的误差。再验证$\Delta L \approx 0.45$与手算一致。
</details>

### 题 5（开放）：GPTQ 与 AWQ 的取舍

读 AWQ 论文后回答：为什么 AWQ 声称"无需重建、几分钟量化"，却能在 4-bit 追平 GPTQ？它把"二阶补偿"换成了什么？（提示：04 章 7.2 的激活幅度。）

<details>
<summary>题 5 解答要点</summary>

AWQ 观察到"通道重要性 ∝ 激活幅度"，用$s=(\max|X|)^\alpha$的 per-channel 缩放直接保护重要通道，等效于把误差预算定向到不敏感处，不需要逐权重求解。代价：没有真正的"误差补偿"，2-bit 以下不如 QuIP#。详见 06 章。
</details>

---

## 12. 延伸阅读

1. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers（arXiv:2210.17323）](https://arxiv.org/abs/2210.17323)：本章主文献
2. OBQ：*Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning*（Frantar & Alistarh, NeurIPS 2022）
3. OBS：Hassibi, Stork & Wolff, *Optimal Brain Surgeon and general network pruning*（1993）：二阶补偿的源头
4. [GPTQ 官方代码](https://github.com/IST-DASLab/gptq)：块更新与 Cholesky 的工程实现
5. 上一篇：[04 量化粒度、校准与离群值](./LLM量化精读笔记-04-量化粒度校准与离群值.md)；下一篇：**[06 权重量化 II：AWQ、SqueezeLLM、QuIP#]**——三种"不用重建"或"2-bit 更强"的路线。
