# LLM 量化精读笔记 · 02 量化问题的形式化与均匀量化理论

> 对应：MIT 6.5940 Lecture 5（Quantization Part I）前半部分；Inference Engineering Ch 5 的数学基础。
> 学完本章你应该能：① 把任何量化方法归入"编码器 + 解码器 + 误差"的框架；② 徒手写出对称/非对称均匀量化的正向与反向公式；③ 推导并解释"每$bit \approx 6 dB$"；④ 区分舍入误差与截断误差，并指出它们分别由什么造成；⑤ 理解为什么"范围利用率"是量化的第一性原理。

---

## 目录（本章）

1. 本章目标
2. 量化问题的一般形式
3. 均匀（线性）量化
4. 量化误差理论
5. 量化器设计的四个旋钮
6. 为什么神经网络能容忍量化
7. PTQ 与 QAT：两条路线的雏形
8. 本章小结与"一句话记忆"
9. 习题与解答
10. 延伸阅读

---

## 1. 本章目标

量化（quantization）在推理优化里承担的角色可以用一句话概括：**在"表示一个数要花多少 bit"和"这个数被算错多少"之间做交易。**

本章不讨论任何具体模型，只建立一个严格的数学地基。后面的章节（04–07）全部是在这个地基上回答同一个问题：**给定一个 LLM 的权重/激活/KV 分布，如何用最少的 bit、最小的质量损失把它表示出来，并且让硬件跑得快。**

---

## 2. 量化问题的一般形式

### 2.1 定义

令$r \in \mathbb{R}$为一个需要表示的实数（可以是权重 w、激活 a 或 KV cache 的值）。量化是这样一个映射：

编码器（量化）$: f : \mathbb{R} \to Q, Q = \{q_0, q_1, ..., q_{M-1}\} \subset \mathbb{Z}, M = 2^b$
解码器（反量化）$: g : Q \to \mathbb{R}, g(q_{i}) = \hat{r}_i$

其中 b 是位宽，$M = 2^b$是可表示的**不同取值数量**。量化后的数值集合 Q 远小于$\mathbb{R}$，因此这是一个**有损压缩**过程：除了恰好落在某个$\hat{r}_i$上的值，其余值都会引入误差$\varepsilon = r - \hat{r}$。

> 信息论直觉：用一个 b bit 的码字表示一个连续量，等价于把实数轴切成$2^b$个区间，每个区间里的值都"坍缩"到同一个代表值。区间越粗（b 越小），每个值的失真越大。这就是率失真（rate-distortion）的雏形：rate（位宽）与 distortion（失真）不可兼得。

### 2.2 在神经网络里的具体化

一个线性层$y = Wx$在量化后变成：

原始$: y = W x$
量化$: \hat{y} \approx dequant(\operatorname{quant}(W)) \cdot dequant(\operatorname{quant}(x))$
$$
= \hat{W} \cdot \hat{x}
$$

于是每一层的输出都携带了上一层权重和激活的量化误差。误差会沿网络传播（第 07/08 章会分析传播方式）。**量化的目标不是让$\hat{W} = W$，而是让整个模型的任务指标（perplexity、MMLU 分数……）退化到统计上与噪声不可区分。**

### 2.3 两种量化对象：数据分布 vs 计算

论文里经常出现两组说法，容易混淆：

| 说法 | 含义 |
|---|---|
| weight-only quantization | 只把权重存成低精度（W4A16 等），激活仍是高精度 |
| W8A8 | 权重和激活都量化到 8-bit，矩阵乘在低精度 Tensor Core 上做 |
| KV cache quantization | 把解码时缓存的 Key/Value 张量降精度（省显存，08 章） |
| 存 vs 算 | 权重可以"只省存储"（weight-only），也可以"省存储 + 加速计算"（W8A8） |

为什么要区分"存"和"算"？因为推理的 decode 阶段是**访存密集**的：权重每用一次都要从 HBM 搬到寄存器，位宽减半 ≈ 带宽翻倍；而 prefill 阶段是**计算密集**的，需要低精度 Tensor Core 的 FLOPS 翻倍才有效果（详见 03 章硬件部分）。这一区分决定了"只量化权重"还是"权重+激活一起量化"的选择。

---

## 3. 均匀（线性）量化

### 3.1 定义

**均匀量化（uniform / linear quantization）**：所有量化层级等距排列，相邻两层间距恒为$\Delta$（称为步长 step size）。

两种最常用的形式：

**（a）对称量化（symmetric）**：零点$z = 0$，量化值域对称：

$$
\begin{aligned}
q &= \operatorname{clamp}(\operatorname{round}(r / s), q_{\min}, q_{\max})
\hat{r} &= q \cdot s
\end{aligned}
$$
其中$s = \max|r| / (2^{b-1} - 1)$（常用，值域对称，0 有精确表示）
或$s = \max|r| / 2^{b-1}$（个别 kernel 用，把$\pm 2^{b-1}$都利用上）

对 INT8：$q_{\min} = -128$，$q_{\max} = 127$；若用 $s = \max|r| / 127$，则$q \in [-127, 127]$，−128 不用。

**（b）非对称量化（affine / asymmetric）**：零点$z \ne 0$：

$$
\begin{aligned}
s &= (r_{\max} - r_{\min}) / (q_{\max} - q_{\min})
z &= \operatorname{round}(q_{\min} - r_{\min} / s)
q &= \operatorname{clamp}(\operatorname{round}(r / s) + z, q_{\min}, q_{\max})
\hat{r} &= (q - z) \cdot s
\end{aligned}
$$

非对称量化把 `[rmin, rmax]` 完整映射到 `[qmin, qmax]`，不浪费整数网格。代价是：矩阵乘时每个量化值要减去 z（多一次加法），硬件实现略贵。

### 3.2 mid-tread 与 mid-rise

对称量化器还可以按"零点是不是量化层级"分成两类：

| 类型 | 零点 | 例子（3-bit） | 特点 |
|---|---|---|---|
| mid-tread | 0 是一个量化值 |$q \in \{-4, -3, -2, -1, 0, 1, 2, 3\}$| 能精确表示 0；适合权重（0 语义重要，如稀疏权重） |
| mid-rise | 0 落在两级之间 |$q \in \{-4, -3, -2, -1, 1, 2, 3, 4\}$| 偶数个层级，无精确 0；适合满量程正弦类信号 |

神经网络里默认选 mid-tread：权重里的 0 有语义（很多权重本来就接近 0，量化后变成 0 可以配合稀疏性）。

### 3.3 为什么默认用均匀量化

均匀量化的实现极其简单，完全贴合硬件：

1. **量化**：一次除法 + round + clamp；
2. **反量化**：一次乘法；
3. **矩阵乘**：整数乘加（或 FP8 低精度 Tensor Core），累加器用 FP32，几乎不损失累加精度。

非均匀量化（如 k-means 量化、对数量化）用更少的码字逼近真实分布，但推理时需要**查表**或**特殊解码**，位宽不整齐、内存布局不友好。现代 LLM 量化论文（GPTQ/AWQ/QuIP#）基本都回到"均匀网格 + 更聪明的网格选择/缩放"，只是把功夫下在 scale 和 outlier 上。

> 一句话：均匀量化 = "等间距网格"；非均匀 = "按分布密度布点"。硬件喜欢前者，精度喜欢后者，所以业界在两者之间折中（见 04 章粒度、06 章 QuIP# 的格码本）。

### 3.4 伪代码（与 MIT Lab 2 对齐）

```python
def get_quantized_range(b):
    """对称整数量化范围，如 b=8 -> (-128, 127)"""
    return -(2 ** (b - 1)), 2 ** (b - 1) - 1

def quantize_symmetric(r, b, s=None):
    """r: 实数张量; b: 位宽; s: 缩放因子（None 则从数据算）"""
    qmin, qmax = get_quantized_range(b)
    if s is None:
        s = max(abs(r)) / (2 ** (b - 1) - 1)   # 对称缩放
    q = round_(r / s)          # 逐元素 round
    q = clamp(q, qmin, qmax)
    return q, s

def dequantize_symmetric(q, s):
    return q * s

def quantize_affine(r, b, rmin=None, rmax=None):
    qmin, qmax = get_quantized_range(b)
    if rmin is None: rmin = min(r)
    if rmax is None: rmax = max(r)
    s = (rmax - rmin) / (qmax - qmin)
    z = round(qmin - rmin / s)
    q = clamp(round(r / s) + z, qmin, qmax)
    return q, s, z

def dequantize_affine(q, s, z):
    return (q - z) * s
```

（$round_$ 用 banker's rounding 或 round-half-away-from-zero 都可，工程上注意与 kernel 一致即可；MIT Lab 2 的实现与此等价。）

### 3.5 数值算例 1：8-bit 对称量化

输入：

$$
r = [0.52, -1.31, 2.05, -0.87], b = 8, q_{\max} = 127
$$

计算：

$$
\begin{aligned}
s &= 2.05 / 127 \approx 0.01614
r/s &= [32.21, -81.16, 127.00, -53.90]
q &= [32, -81, 127, -54]
\hat{r} &= [0.5165, -1.3075, 2.0500, -0.8717]
\varepsilon &= r - \hat{r} = [0.0035, -0.0025, 0, 0.0017]
\end{aligned}
$$

最大误差$|\varepsilon |\max \approx 0.0035 < \Delta /2 = s/2 \approx 0.00807 ✓$（满足舍入误差上界，因为所有值都在量化范围内，没有截断）。

### 3.6 数值算例 2：4-bit 对称量化（同样的数据）

$b = 4, q_{\max} = 7$（对称时$2^{4-1}-1 = 7$）
$$
\begin{aligned}
s &= 2.05 / 7 \approx 0.2929
r/s &= [1.78, -4.47, 7.00, -2.97]
q &= [2, -4, 7, -3]
\hat{r} &= [0.5857, -1.1714, 2.0500, -0.8786]
\varepsilon &= [-0.0657, -0.1386, 0, 0.0086]
\end{aligned}
$$

最大误差$|\varepsilon |\max \approx 0.1386$，仍是$\Delta /2 = s/2 \approx 0.1464$以内。**位宽从 8 降到 4，误差上限扩大了约 16 倍（$\Delta$放大了 16 倍：$0.01614 \to 0.2929$）。** 注意：噪声功率理论上是$4^{4} = 256$倍（−24 dB），但这里样本只有 4 个数，均匀噪声假设不成立，实际比值会偏离——这正是"统计模型需要足够多样本"的例子（见 4.2）。

### 3.7 数值算例 3：非对称量化"省位"

考虑 ReLU 之后的激活，全部非负：

$$
r = [0.1, 0.9, 0.05, 2.0]
$$

**对称 4-bit**：$s = 2.0/7 \approx 0.2857$，整数网格覆盖$[-2.0, 2.0]$，但数据只用了一半$\to$量化层级浪费一半：

$$
q = [0, 3, 0, 7] \to \hat{r} = [0, 0.857, 0, 2.0]
$$

**非对称 4-bit**：$s = (2.0 - 0.05)/15 = 0.13$，整数网格覆盖$[0.05, 2.0]$：

$$
q = [1, 7, 0, 15] \to \hat{r} = [0.130, 0.910, 0, 1.950]
$$

误差从对称时的最大 0.1 降到约 0.05。结论：**分布不对称时，非对称量化等于免费拿回被浪费的位**——这是后面 SmoothQuant/激活量化的基础直觉之一。

---

## 4. 量化误差理论

### 4.1 误差分解：舍入 + 截断

量化误差由两部分组成：

$$
\varepsilon = r - \hat{r} = \varepsilon_{	ext{round}} + \varepsilon_{	ext{clip}}
$$

- **舍入误差$\varepsilon_{	ext{round}}$**：当 r 落在量化范围 [rmin, rmax] 内时，r 被就近映射到最近层级，误差≤ $\Delta /2$。
- **截断误差$\varepsilon_{	ext{clip}}$**：当 r 超出 [rmin, rmax] 时，被 clamp 到边界，误差= $|r - r_{\min}|$或 |r − rmax|，**可以任意大**，与位宽无关。

这是整个量化理论里最重要的一个划分，值得停下来想清楚：

> **位宽决定"格子细不细"（舍入误差）；范围决定"边界在哪"（截断误差）。** 大多数翻车事故不是舍入误差，而是截断误差——某个 outlier 值把范围撑得极大，导致大部分正常值的有效位宽被稀释，或者干脆被 clamp 掉。

LLM 激活里恰好存在大量 outlier（少数维度值巨大），这正是 LLM.int8()、SmoothQuant、AWQ 全部要解决的**同一个敌人**（04、06、07 章）。

### 4.2 加性噪声模型

当 r 在量化范围内且值域被"充分激励"（样本足够多、分布不极端）时，可以把量化器建模为：

$$
\hat{r} = r + e, e \sim Uniform(-\Delta /2, \Delta /2)
$$

两个性质：

$E[e] = 0$（无偏）
$Var[e] = \Delta ^{2} / 12$（均匀分布方差：$\int x^{2}/\Delta dx over [-\Delta /2, \Delta /2]$）

推导：均匀分布在$[-\Delta /2, \Delta /2]$的方差= $(\Delta /2 - (-\Delta /2))^{2}/12 = \Delta ^{2}/12$。$\Delta ^{2}/12$是量化噪声的**教科书公式**，后面所有 SNR 计算都从它出发。

这个模型成立的条件是：舍入误差在各区间内均匀分布（对充分随机的信号近似成立）。小样本、周期性信号、有截断时都不成立——这也是为什么例 2 的噪声功率与理论有偏差。

### 4.3 SNR 推导：每 bit ≈ 6 dB

**设定**：b-bit 均匀量化器（mid-rise，$2^b$个层级）覆盖满量程 [−A, A]；信号 r 在 [−A, A] 上均匀分布。

步长：

$$
\Delta = 2A / 2^b
$$

信号方差（均匀分布）：

$$
\sigma _s^{2} = (2A)^{2} / 12 = A^{2} / 3
$$

噪声方差（4.2 的公式）：

$$
\sigma _e^{2} = \Delta ^{2} / 12 = (2A / 2^b)^{2} / 12 = A^{2} / (3 \cdot 4^b)
$$

信噪比：

$$
\begin{aligned}
SNR &= \sigma _s^{2} / \sigma _e^{2} = (A^{2}/3) / (A^{2}/(3\cdot 4^b)) = 4^b
SNR(dB) &= 10\cdot \log_{10}(4^b) = b \cdot 10\cdot \log_{10}(4) \approx 6.02 \cdot b dB
\end{aligned}
$$

**结论：位宽每增加 1 bit，信噪比提升约 6 dB（噪声功率降为原来的 1/4）。** 这就是"6 dB per bit"的来历。

两个常用的变体（推导完全相同，只是信号方差不同）：

| 信号模型 | 信号方差$\sigma _s^{2}$| SNR(dB) |
|---|---|---|
| 满量程均匀分布 |$A^{2}/3$| 6.02·b |
| 满量程正弦（振幅 A） |$A^{2}/2$| 6.02·b + 1.76 |
| 半量程均匀分布（只用到$[-A/2, A/2]$） |$A^{2}/12$| 6.02·(b−1) |

最后一行非常关键：**量化范围只用一半，等于白丢 1 bit。** 推广：如果信号实际只占量化范围的$1/2^k$，就损失约 k·6 dB（k 个有效位）。这把"范围利用率"和"有效位宽"直接挂钩——04 章 outlier 问题的数学根源就在这里。

### 4.4 直觉：1 bit 到底意味着什么

- 噪声功率：×1/4（6 dB 约等于"音量减半"）
- 权重矩阵：存储减半、访存带宽等效翻倍
- 计算：低精度 Tensor Core FLOPS 翻倍

所以"量化降一级 30–50% 性能提升"（Inference Engineering Ch5 的结论）本质上就是：**用少一半的存储/带宽/计算精度，换取近似翻倍的吞吐，代价是 6 dB 的噪声预算。**

---

## 5. 量化器设计的四个旋钮

一个均匀量化器由四个旋钮完全确定，后面所有论文都是在调这四个旋钮：

### 5.1 位宽 b

决定噪声上限（6 dB/bit）和压缩率。INT8 是默认起点；4-bit 是当前 LLM 权重的激进甜点；1.58-bit（BitNet）是极限探索。

### 5.2 范围 [rmin, rmax]（等价于 s、z）

决定截断误差与范围利用率。范围怎么选？从**校准数据**（calibration set）统计：

朴素：$\min/\max \to$对 outlier 极敏感（范围被撑大，有效位宽被稀释）
稳妥：分位数（如 99.99% 分位）$\to$牺牲少量截断误差，换正常值更高的精度
最优：per-channel / per-group 分别选范围（04 章）

### 5.3 对称 vs 非对称

见 3.7：分布是否含符号、是否接近 0 决定选择。硬件上对称更便宜（无 z 减法），非对称更省位。

### 5.4 粒度（granularity）

一个 scale 覆盖多少个元素：

per-tensor（整层一个 s）$\to per-channel$（每行/列一个 s）$\to per-group$（每 G 个元素一个 s）

粒度越细，对分布适应越好、误差越小，但 scale 本身要占存储、kernel 要处理非均匀布局。**粒度是"精度 vs 开销"的连续光谱**，04 章单独展开。

---

## 6. 为什么神经网络能容忍量化

既然每个数都被污染了 6–24 dB 的噪声，为什么网络还能工作？四个层次的解释：

1. **冗余性**：模型参数量远超信息需求。一个 70B 模型的大部分权重对输出的影响极小，删掉/量化它们几乎不可感知（这也是剪枝 work 的原因，MIT 6.5940 Lecture 3/4）。
2. **平均化效应**：单个数被污染影响小，很多数一起被污染时，误差在求和/求平均中部分抵消（随机误差的 √N 平均）。线性层的输出是成千上万个乘积之和，误差不是简单叠加。
3. **鲁棒的优化目标**：训练时 loss 已经对权重噪声有一定容忍度；QAT 更进一步让模型"学着忍受"量化噪声（09 章）。
4. **不同张量的敏感度不同**：权重量化误差是"静态污染"，可以用校准数据事后补偿（GPTQ/AWQ）；激活是"动态污染"，逐 token 变化，更难对付；KV cache 误差会跨 token 累积（08 章）；attention/softmax 对数值范围极其敏感。于是有了 Inference Engineering Ch5 的敏感性排序：

权重（最不敏感）< 激活 < KV cache < attention/softmax（最敏感）

这个排序决定了后文的攻坚顺序：**先量化权重（04/05），再碰激活（06），谨慎处理 KV（07），最后才考虑 attention（10）。**

---

## 7. PTQ 与 QAT：两条路线的雏形

### 7.1 PTQ（Post-Training Quantization）

训练完成后直接量化：

流程：加载 FP16/BF16 权重$\to$用少量校准数据统计范围$\to$量化$\to$部署
优点：不需要训练、不需要原始训练数据、几小时内完成
缺点：误差"事后发生"，模型没有机会适应；只能靠更好的量化算法弥补

本系列 05–08 章全部是 PTQ 算法（RTN/GPTQ/AWQ/SmoothQuant/KIVI 等）。

### 7.2 QAT（Quantization-Aware Training）

训练过程中就模拟量化误差：

流程：前向时把权重/激活"假装"量化（$\operatorname{quantize} \to \operatorname{dequantize}$）$\to$反向传播更新
关键：量化是分段常数函数，梯度几乎处处为$0 \to$用 STE（straight-through estimator）把梯度近似传过去
优点：质量上限高，模型学会了容忍噪声
缺点：需要训练数据与算力

09 章会完整推导 STE 并比较 PTQ/QAT 的精度-成本曲线。

> 现在的工程常态：**先 PTQ 用默认配方（FP8/W8A8）上线，质量不达标再升级**（更好的 PTQ 算法$\to KV$量化验证$\to$局部$QAT \to$全量 QAT）。不要一上来就 QAT。

---

## 8. 本章小结与"一句话记忆"

1. **量化 = 有损压缩**：$2^b$个码字表示实数轴，b 是唯一决定噪声上限的旋钮。
2. **均匀量化 = 等距网格**：对称（$z=0$，省事）vs 非对称（$z\ne 0$，省位）；mid-tread 保 0。
3. **误差 = 舍入 + 截断**：位宽管舍入（$\Delta /2$上界），范围管截断（无上界）。outlier 主要制造截断误差。
4. **每$bit \approx 6 dB$**：噪声功率 ×1/4；量化范围只用一半 = 白丢 1 bit。
5. **四个旋钮**：位宽、范围、对称性、粒度——所有论文都在调这四个旋钮。
6. **敏感性排序**：权重 < 激活 < KV cache < attention，决定进攻顺序。

> 一句话记忆：**"先选格子粗细（b），再选格子放哪（范围），最后决定谁跟谁共用一个格子（粒度）。"**

---

## 9. 习题与解答

### 题 1（手算）：对称量化

对$r = [3.2, -1.7, 0.05, -4.9]$：

(a) 用 8-bit 对称量化，写出 s、q、$\hat{r}$、最大误差。
(b) 用 4-bit 对称量化，重复 (a)。
(c) 比较两者的$\Delta$和最大误差，验证"位宽减 4，$\Delta$放大 16 倍"。

<details>
<summary>题 1 解答</summary>

(a) $s = 4.9/127 \approx 0.03858$；$r/s \approx [82.9, -44.1, 1.3, -127.0] \to q = [83, -44, 1, -127]$；$\hat{r} \approx [3.202, -1.698, 0.039, -4.900]$；$|\varepsilon |\max \approx 0.0113 < s/2 \approx 0.0193$。

(b) $s = 4.9/7 = 0.7$；$r/s \approx [4.57, -2.43, 0.071, -7.0] \to q = [5, -2, 0, -7]$；$\hat{r} = [3.5, -1.4, 0, -4.9]$；$|\varepsilon |\max = 0.3$。

(c) $\Delta _8 = 2s_8 \approx 0.0772$，$\Delta _4 = 2s_4 = 1.4$；$1.4/0.0772 \approx 18.1$（小样本下略偏离理论 16，因为$127/7 \approx 18.1——$注意这里对称量化用$2^{b-1}-1$作分母，所以严格说$\Delta$比值是 127/7）。理论值用$2^{b-1}$作分母时为 16。两个约定都要会。
</details>

### 题 2（推导）：6 dB/bit 的另一种推法

不用均匀分布假设，直接从$\Delta$出发：证明把位宽从 b 增加到 b+1（范围不变），量化噪声功率降为原来的 1/4，SNR 提升 6 dB。

<details>
<summary>题 2 解答</summary>

范围 [−A, A] 不变时，$\Delta _{b+1} = 2A/2^{b+1} = \Delta _b / 2$。噪声功率$\propto \Delta ^{2}$，所以$\sigma _e^{2}(b+1) = \sigma _e^{2}(b)/4$。SNR(dB) 提升= $10\cdot \log_{10}(4) \approx 6.02 dB$。与分布无关，只依赖"均匀量化 + 范围固定 + 无截断"三个假设。
</details>

### 题 3（思考）：mid-tread vs mid-rise

为什么权重量化几乎总是用 mid-tread？如果权重矩阵有 95% 的元素是 0，两种量化器分别会发生什么？

<details>
<summary>题 3 解答</summary>

mid-tread 的 0 是精确量化层级：$0 \to 0$，无误差，且与稀疏性天然兼容。mid-rise 的 0 会被量化到$\pm \Delta /2$，95% 的元素全部产生$\pm \Delta /2$的误差，噪声功率会非常高，而且破坏稀疏结构（0 变成非 0）。所以对称整数量化默认 mid-tread。
</details>

### 题 4（编程，建议在 MIT Lab 2 环境里做）

实现 $\text{quantize\_symmetric}$ / $\text{quantize\_affine}$ / `dequantize`（3.4 的伪代码），然后：

(a) 对 N(0,1) 随机张量（$N=100000$）测$SQNR = 10\cdot \log_{10}(\sum r^{2}/\sum (r-\hat{r})^{2})$，位宽 4/6/8/10/12，画 SQNR vs b，验证斜率≈ $6 dB/bit$。
(b) 把同样的张量先放大 100 倍（模拟 outlier 撑大范围），再量化到 8-bit，观察 SQNR 掉了多少 dB，并解释（对应 4.3 的"半量程 −6 dB"）。
(c) 对比对称与非对称量化在$[0, 5]$均匀分布上的 SQNR（对应 3.7）。

<details>
<summary>题 4 解答要点</summary>

(a) 满量程均匀信号的 SQNR 应近似 6.02·b dB；N(0,1) 是高斯，非满量程均匀，斜率仍约 6 dB/bit，但常数项不同（高斯重尾，偶有截断）。
(b) 放大 100 倍后范围$[-100\sigma , 100\sigma ]$，正常值只占很小比例$\to$有效位宽减少$\log_2(100)\approx 6.6 bit$，SQNR 掉约 40 dB（放大倍数越大越明显）。
(c) 对称量化覆盖$[-5,5]$，一半网格浪费$\to$有效位宽 −1 bit，SQNR 比非对称低约 6 dB。
</details>

### 题 5（挑战）：范围利用率的数学

证明：若信号均匀分布在$[-A/2, A/2]$，而量化器范围是 [−A, A]，则量化后等效于用 b−1 bit 量化满量程信号（即$SQNR = 6.02(b-1) dB$）。

<details>
<summary>题 5 解答</summary>

$\sigma _s^{2} = (A/2)^{2}/3 = A^{2}/12$；$\sigma _e^{2}$不变（$\Delta$由量化器范围决定）= $A^{2}/(3\cdot 4^b)$；$SNR = (A^{2}/12)/(A^{2}/(3\cdot 4^b)) = 4^b/4 = 4^{b-1} \to 6.02(b-1) dB$。**范围利用率 k 分之$1 =$丢 log2(k) 个有效位。**
</details>

---

## 10. 延伸阅读

1. [MIT 6.5940 Lecture 5 – Quantization Part I（Fall 2024，Class Central）](https://www.classcentral.com/course/youtube-efficientml-ai-lecture-5-quantization-part-i-mit-6-5940-fall-2024-340161)：线性量化、bitwidth、PTQ 总览
2. [MIT 6.5940 Lab 2：Quantization](https://github.com/CalebDu/MIT6.5940-EfficientML/blob/master/Lab2-quantization/Lab2.ipynb)：linear quantize / k-means 量化的动手实现（对应本系列习题环境）
3. [Inference Engineering Ch 5](https://inferenceengineering.tech/chapters/techniques/)：教材正文（格式总览、敏感性排序、质量评估）
4. Gersho & Gray, *Vector Quantization and Signal Compression*：均匀量化误差理论的经典教材（$\Delta ^{2}/12$、6 dB/bit 的出处）
5. 上一篇：**[01 数值编码与计算机表示基础](./LLM量化精读笔记-01-数值编码与计算机表示基础.md)**——整数/浮点/舍入/截断的计算机表示，本章的数学工具都建立在其上
6. 下一篇：**[03 数值格式：FP16/BF16/FP8/FP4/MXFP/NVFP4 与硬件]**——把"格子"具体到硬件支持的浮点格式，回答"为什么 FP8 是甜点、BF16 为什么适合训练、FP4 为什么必须配块缩放"。
