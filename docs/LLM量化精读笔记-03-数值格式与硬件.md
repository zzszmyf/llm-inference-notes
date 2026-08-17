# LLM 量化精读笔记 · 03 数值格式：FP16 / BF16 / FP8 / FP4 / MXFP8 / NVFP4 与硬件

> 对应：Inference Engineering Ch5 的 Number Formats 部分；NVIDIA/Arm/Intel《FP8 Formats for Deep Learning》白皮书；OCP Microscaling Formats (MX) 规范；MIT 6.5940 Lecture 5 的数据类型总览。
> 学完本章你应该能：① 从位布局推导任意浮点格式的最大值、最小正常值、machine epsilon；② 解释 E4M3 与 E5M2 的区别及各自适用场景；③ 说明 MXFP8/NVFP4 的"块缩放"到底解决了什么问题；④ 用带宽模型手算"70B 模型 decode 每 token 至少多少毫秒"；⑤ 论证"FP8/MXFP8 是生产甜点"。

---

## 目录（本章）

1. 本章目标
2. 从 IEEE 754 到推理格式：一张家族谱系
3. 16-bit 的两种哲学：FP16 与 BF16
4. FP8 深入：E4M3 与 E5M2
5. FP4 与块缩放：NVFP4、MXFP4
6. Microscaling（MX）格式：共享指数缩放
7. 硬件：Tensor Core、FLOPS 与带宽模型
8. 格式选择决策框架：为什么 FP8 是甜点
9. 格式速查表
10. 本章小结
11. 习题与解答
12. 延伸阅读

---

## 1. 本章目标

01 章我们建立了"浮点 = 符号位 + 偏置指数 + 尾数，指数换范围、尾数换精度"的框架；02 章建立了"每$bit \approx 6 dB$、范围利用率决定有效位宽"的误差理论。本章把这两个框架**套到真实的推理格式上**，并回答三个问题：

1. 每种格式的位布局长什么样，它的最大值/精度/适用范围怎么算？
2. 为什么推理工程给出的格式表里，FP8 是"质量/速度甜点"、FP4 是"激进选项"、NVFP4 是"FP4 里精度最好"？
3. 位宽到底如何变成毫秒和 token/s？（带宽模型）

---

## 2. 从 IEEE 754 到推理格式：一张家族谱系

所有推理格式都可以放进下面这张谱系（按"数据如何编码"分类）：

```
                  表示一个实数
                       │
        ┌──────────────┼──────────────────┐
     整数/定点         浮点（指数+尾数）      混合/块缩放
   INT8/INT4         FP16/BF16/FP8/FP4    MXFP8/MXFP4/NVFP4
   等距网格           每元素独立指数          块共享指数缩放
   动态范围差          动态范围好            动态范围好 + 省位
```

关键认知：

- **整数/定点**：网格等距，动态范围完全由位宽决定。outlier 一出现就爆（02 章）。
- **浮点**：每个值自带指数，能跨数量级表示数值。**这就是 Inference Engineering 说的"exponent gives higher dynamic range than integers, better representing outliers"**。
- **块缩放（block scaling / microscaling）**：为了在低位数里保留动态范围又不为每个值付指数位的代价，让**一组元素共享一个缩放因子**。这是 MXFP8/MXFP4/NVFP4 的设计哲学，也是 Blackwell 时代的核心创新。

---

## 3. 16-bit 的两种哲学：FP16 与 BF16

### 3.1 位布局与数值表

| 格式 | S | E | M（存储） | bias | 最大正常值 | 最小正常值 |$\varepsilon$|
|---|---|---|---|---|---|---|---|
| FP16 | 1 | 5 | 10 | 15 | 65,504 |$2^{-14} \approx 6.1e-5$|$2^{-10} \approx 9.8e-4$|
| BF16 | 1 | 8 | 7 | 127 |≈ $3.39e38$（同 FP32） |$2^{-126}$（同 FP32） |$2^{-7} \approx 7.8e-3$|

推导示例（FP16 最大值）：

FP16 指数域$e \in [1, 30]$，最大正常指数真值= $30 - 15 = 15$
最大尾数= $1.1111111111_{2} = 2 - 2^{-10}$
最大值= $(2 - 2^{-10}) \times 2^{15} = 1.9990234375 \times 32768 = 65,504$

### 3.2 为什么训练用 BF16、推理默认 FP16

训练（尤其预训练）：
  梯度跨多个数量级（$10^{-6} \sim 10^{2}$），范围比精度重要$\to BF16$（指数$8 bit = FP32$的范围）
  代价：尾数 7 bit，相对精度只有$\sim 0.8\%$，但训练有 FP32 master weights 兜底

推理：
  权重/激活值域已知且稳定（通常$\pm$几十），范围够用$\to FP16$（尾数 10 bit，精度高）
  代价：范围小，遇到极端 outlier 会溢出（max 65,504）

**一个直觉记忆**：BF16 是"把 FP32 的尾数砍掉一半"；FP16 是"把 FP32 的指数砍掉一半"。推理要精度、训练要范围，所以两者分工。

> 顺带一提 TF32：Ampere 训练用的 19-bit 格式（8 位指数 + 10 位尾数），是 FP32 输入的"截断版"，用于加速矩阵乘。它不是推理格式，但体现了同一 tradeoff。

---

## 4. FP8 深入：E4M3 与 E5M2

### 4.1 位布局

FP8 是 NVIDIA/Arm/Intel 2022 年联合发布的白皮书格式（arXiv:2209.05433），两种变体：

E4M3：S(1) + E(4) + M(3)   指数偏置 7
E5M2：S(1) + E(5) + M(2)   指数偏置 15

名字里的 E×M× 就是"指数位数 × 尾数位数"——看到任何格式名，第一反应就是套 01 章的总纲：**指数位多$\to$范围大精度低；尾数位多$\to$精度高范围小**。

### 4.2 E4M3 数值推导（完整）

$bias = 7$，指数域$e \in [0, 15]$

规格化数（$1 \le e \le 14$）：
  最小正常值= $1.0 \times 2^{1-7} = 2^{-6} \approx 0.0156$
  常规最大正常值= $(2 - 2^{-3}) \times 2^{14-7} = 1.875 \times 128 = 240$

E4M3FN 的扩展（没有 Infinity）：
$e = 15$时，$m = 000\sim 110$仍作为有限数：
    最大值= $1.11_{2} \times 2^{15-7} = 1.75 \times 256 = 448$
$m = 111$保留给 NaN（FP8 无 ∞）

次正规数：最小次正规= $2^{-6} \times 2^{-3} = 2^{-9} \approx 1.95e-3$
相对精度：$2^{-3} = 12.5\%$

### 4.3 E5M2 数值推导

$bias = 15$，指数域$e \in [0, 31]$

规格化数（$1 \le e \le 30$）：
  最小正常值= $2^{1-15} = 2^{-14} \approx 6.1e-5$
  最大值= $(2 - 2^{-2}) \times 2^{30-15} = 1.75 \times 32768 = 57,344$

$$
e = 31：m = 0 \to \pm \infty ；m \ne 0 \to NaN
$$
相对精度：$2^{-2} = 25\%$

### 4.4 E4M3 vs E5M2 怎么选

| | E4M3 | E5M2 |
|---|---|---|
| 指数/尾数 | 4 / 3 | 5 / 2 |
| 最大值 | 448 | 57,344 |
| 最小正常值 |$2^{-6}$|$2^{-14}$|
| 相对精度 | 12.5% | 25% |
| 特殊值 | 无 ∞（NaN 唯一） |$\pm \infty$、NaN |
| 白皮书建议 | **权重与激活（前向）** | **梯度（反向）与需要大范围的场景** |

直觉：权重/激活的值集中在$\pm$几十，精度更重要$\to E4M3$；梯度的数量级波动大，范围更重要$\to E5M2$。

> FP8 与 INT8 的本质区别：FP8 的指数位让它在**不增加位宽的情况下**保留了动态范围。量化 LLM 时同样的 8 bit，FP8 对 outlier 的容忍度远高于 INT8——这是"FP8 甜点"的第一个论据。

---

## 5. FP4 与块缩放：NVFP4、MXFP4

### 5.1 裸 FP4（E2M1）：16 种取值

FP4 元素格式 E2M1（1 符号 + 2 指数 + 1 尾数，bias 1）的全部取值：

$\pm 0$，$\pm 0.5$，$\pm 1$，$\pm 1.5$，$\pm 2$，$\pm 3$，$\pm 4$，$\pm 6$（共 16 种，含符号）

推导：$e=0$次正规 0.5；$e=1 \to 1$、1.5；$e=2 \to 2$、3；$e=3$（无 ∞ 时扩展）$\to 4$、6。

**裸 FP4 的问题**：最大值只有 6，且层与层之间数值尺度差异巨大（不同层激活的典型幅度可以差几个数量级）。直接对整层用同一个 FP4 格式，绝大多数值要么溢出要么精度被浪费。

### 5.2 块缩放：让 FP4 能用

解决方案是**给一小块元素配一个缩放因子**：

存储：每个元素 4 bit（E2M1）
     每 16 或 32 个元素共享 1 个 block scale（8/16 bit）
解码：$\hat{r} = \text{element\_value} \times \text{block\_scale}$

这样既保留了浮点的动态范围（scale 覆盖数量级），又把元素压到 4 bit。

### 5.3 NVFP4：NVIDIA 的 Blackwell 4-bit 方案

元素：E2M1（4 bit）
块大小：16（比 MXFP4 的 32 更细）
块缩放：FP8 E4M3（8 bit，非 2 的幂$\to$更精细）
张量缩放：额外一个 FP32 scale
硬件：Blackwell 原生

NVIDIA 官方博客（2025-06）给出的动机：块从 32 缩到 16，缩放因子能更"贴身"地适配局部动态范围；块缩放用 E4M3 而不是纯 2 的幂，进一步减少量化误差。这就是 Inference Engineering 表格里"$NVFP4 = Best FP4 accuracy (block size 16)$"的依据。

### 5.4 MXFP4：开放标准版

元素：E2M1（4 bit）
块大小：32
块缩放：E8M0（8 bit 纯指数，2 的幂）
标准：OCP Microscaling Formats

MXFP4 的 scale 是 2 的幂（E8M0，见 6.2），实现更简单、更省（不用归一化尾数），但缩放粒度不如 NVFP4 的 E4M3 精细。

| | NVFP4 | MXFP4 |
|---|---|---|
| 元素 | E2M1 | E2M1 |
| 块大小 | 16 | 32 |
| 块缩放 | FP8 E4M3 | E8M0（2 的幂） |
| 额外缩放 | FP32 张量 scale | 无 |
| 标准 | NVIDIA 专有 | OCP 开放标准 |

---

## 6. Microscaling（MX）格式：共享指数缩放

### 6.1 设计动机

回顾浮点格式的本质：**每个值花 bit 存自己的指数**。位数越低，指数越贵（FP4 里 2 个指数位占了 50% 的预算）。

Microscaling 的思路：**指数不跟着每个值走，而是跟着一块值走**：

传统浮点：每个值 = 尾数 + 自己的指数
MX 格式：  每个值 = 尾数（低精度元素）+ 整块共享的指数（E8M0 scale）

### 6.2 E8M0：纯指数缩放

$E8M0 = 8 bit$，无符号、无尾数：$s = 2^{e - 127}$，$e \in [0, 255]$
只表达 2 的幂次缩放，不参与"值"本身

### 6.3 MXFP8 / MXFP6 / MXFP4

OCP MX 规范定义了三种主力格式：

| 格式 | 元素 | 块大小 | scale |
|---|---|---|---|
| MXFP8 E4M3 | E4M3 | 32 | E8M0 |
| MXFP8 E5M2 | E5M2 | 32 | E8M0 |
| MXFP6 | E3M2 / E2M3 | 32 | E8M0 |
| MXFP4 | E2M1 | 32 | E8M0 |

Blackwell（SM 100+）原生加速 MX 格式的点积。

### 6.4 MX 为什么精度更好

同一个张量里，不同区域的数量级可能差很远。MX 让**每 32 个元素独立适配自己的数量级**，等效于"per-block 的指数"：

per-tensor FP8：整层一个数量级$\to$局部小值被大值"吃掉精度"
MXFP8：每块一个$E8M0 \to$局部数量级自动对齐$\to$有效精度接近"每块独立 FP8"

代价：scale 要额外存储（每 32 元素$8 bit \to$每元素 0.25 bit 开销）和计算（解码时乘一次）。这是 Inference Engineering 里"$MXFP8 = FP8$的精度升级版"的技术含义。

---

## 7. 硬件：Tensor Core、FLOPS 与带宽模型

### 7.1 Tensor Core 简史：精度下降的驱动力

Pascal (2016)：FP16 Tensor Core 登场（推理默认 FP16 的硬件起点）
$$
Ampere (2020)：TF32 / BF16 / INT8
$$
Hopper (2022)：FP8（E4M3/E5M2），FLOPS 翻倍
Blackwell (2024)：FP4 / MXFP8 / MXFP4 原生

关键事实：**Tensor Core 的吞吐随位宽近似线性翻倍**：

$$
H100 SXM：FP16 \approx 989 TFLOPS（dense），FP8 \approx 1979 TFLOPS
$$
B200：$FP8 \approx 4.5 PFLOPS$，$FP4 \approx 9 PFLOPS$（约 2 倍 FP8）

这就是"prefill 阶段量化降一级、FLOPS 翻倍"的出处。

### 7.2 decode 带宽模型：位宽 = 毫秒

decode 阶段每个 token 都要把权重从 HBM 搬一遍。设模型参数量 N，权重位宽 B bit，带宽 BW：

每 token 权重搬运时间= $N \times B/8 / BW$

70B 模型、H100（$BW \approx 3.35 TB/s$）：
  FP16：$70e9 \times 2 / 3.35e12 \approx 41.8 ms/token \to$上限$\sim 24 tok/s$
  FP8 ：$70e9 \times 1 / 3.35e12 \approx 20.9 ms/token \to$上限$\sim 48 tok/s$
  FP4 ：$70e9 \times 0.5 / 3.35e12 \approx 10.5 ms/token \to$上限$\sim 96 tok/s$

三个结论：

1. **decode 是访存密集的**：算力根本不是瓶颈，HBM 带宽才是；量化权重 = 直接减搬运量。
2. **每降一级≈ $2$倍带宽**：这与 Inference Engineering 的"30–50% 性能提升/级"一致（实际还有 kernel 开销、激活和 KV 的带宽，所以是 30–50% 而不是 100%）。
3. **FP4 的收益在 decode 端最大**：这也是为什么权重量化（W4A16/W4A8）在服务端如此流行。

### 7.3 prefill 计算模型：位宽 = FLOPS

推理 forward 每 token 的$FLOPs \approx 2N$（N 为参数量；乘法 + 加法各一次）：

70B 模型、2048 token 的 prefill：
$$
\begin{aligned}
 FLOPs \approx 2 \times 70e9 \times 2048 \approx 287 TFLOP
 H100 FP16（989 TFLOPS）：\approx 0.29 s
 H100 FP8 （1979 TFLOPS）：\approx 0.145 s
\end{aligned}
$$

这就是"prefill 是计算密集、量化降一级 FLOPS 翻倍、TTFT 减半"的出处。

> **把 7.2 和 7.3 放在一起**，就得到推理工程的完整图景：量化对 prefill 的作用在计算单元（FLOPS 翻倍），对 decode 的作用在数据通路（带宽翻倍）。两者的共同分母都是"位宽"。

---

## 8. 格式选择决策框架：为什么 FP8 是甜点

综合 01–03 章，把候选格式放进"质量 × 速度 × 风险"坐标系：

| 格式 | 位宽 | 相对速度 | 相对精度 | 质量风险 | 适用 |
|---|---|---|---|---|---|
| FP16 | 16 | 1x | 高 | 无 | 默认推理基线 |
| BF16 | 16 |$\sim 1x$| 中（范围大） | 低 | 训练 / 高动态范围 |
| FP8（E4M3） | 8 |$\sim 1.5x$| 中 | 低 | **生产甜点** |
| MXFP8 | 8 |$\sim 1.5x$| 中高 | 低 | 精度敏感的生产场景 |
| FP4 | 4 |$\sim 2x$| 低 | 高 | 激进压缩 |
| NVFP4 | 4 |$\sim 2x$| 低中 | 中高 | Blackwell 上的 4-bit 首选 |

**"FP8 是甜点"的三个论据**：

1. **质量**：E4M3 有 4 位指数，天然抗 outlier（优于 INT8）；配合 SmoothQuant（07 章）的 W8A8 在主流 LLM 上几乎无损；DeepSeek-V3 671B 全程 FP8 是大规模实证。
2. **速度**：prefill 2 倍 FLOPS、decode 2 倍带宽（7.2/7.3），实测约 1.5x 端到端。
3. **风险可控**：权重 + 激活量到 FP8、KV cache 谨慎处理、attention 保持原精度——这是 Inference Engineering 的 Key Takeaway，也是当前生产共识。

FP4 的定位：**在 Blackwell 上、质量评测通过的前提下**做激进压缩（省 75% 权重显存）。NVFP4 优先于裸 FP4/MXFP4（块更细、scale 更精）。

---

## 9. 格式速查表

| 格式 | S | E | M | bias | 最大值 | 最小正常值 | 最小次正规 |$\varepsilon$| 可表示取值 |
|---|---|---|---|---|---|---|---|---|---|
| FP16 | 1 | 5 | 10 | 15 | 65,504 |$2^{-14}$|$2^{-24}$|$2^{-10}$| 65,536 |
| BF16 | 1 | 8 | 7 | 127 |$\sim 3.4e38$|$2^{-126}$|$2^{-133}$|$2^{-7}$| 65,536 |
| FP8 E4M3 | 1 | 4 | 3 | 7 | 448 |$2^{-6}$|$2^{-9}$|$2^{-3}$| 256 |
| FP8 E5M2 | 1 | 5 | 2 | 15 | 57,344 |$2^{-14}$|$2^{-16}$|$2^{-2}$| 256 |
| FP4 E2M1 | 1 | 2 | 1 | 1 | 6 | 1 | — |$2^{-1}$| 16 |
| MXFP8 | 1 | 4/5 | 3/2 | 块共享 | 同 FP8 × 块 scale | — | — | 同 FP8 | 256 × 块 |
| NVFP4 | 1 | 2 | 1 | 块共享 | 6 × 块 scale | — | — |$2^{-1}$| 16 |

> 提示：FP16/BF16 的"可表示取值 65,536"是位模式数（含$NaN/Inf/\pm 0$），不是有效实数个数；FP8/FP4 同理。表格来自 Inference Engineering，语义是"码本大小上限"。

---

## 10. 本章小结

1. **浮点 = 符号 + 偏置指数 + 尾数**；指数位买动态范围，尾数位买精度——所有格式差异都能从这推导。
2. **FP8 E4M3**（权重/激活）与 **E5M2**（梯度）是同一预算的两种分配；E4M3FN 最大值 448、无 ∞。
3. **FP4 必须配块缩放**：NVFP4（块 16 + E4M3 scale）比 MXFP4（块 32 + E8M0）精度更好。
4. **MX 格式**把指数从"每值"改成"每块"，是低位数保留动态范围的优雅方案（Blackwell 原生）。
5. **位宽 × 硬件 = 性能**：decode 带宽模型（$140GB/3.35TB/s \approx 42ms$）与 prefill 计算模型（2N FLOPs/token）是量化的两个收益来源。
6. **FP8/MXFP8 是生产甜点**：质量几乎无损、速度约 1.5x、风险可控；FP4/NVFP4 是 Blackwell 上的激进选项。

> 一句话记忆：**"8 bit 的 E4M3 用 4 个指数位买到了 INT8 给不了的动态范围，这就是 FP8 能当甜点、INT8 只能当配角的全部秘密。"**

---

## 11. 习题与解答

### 题 1（推导）：MXFP8 的块开销

MXFP8 块大小 32、scale 为 E8M0（8 bit）。计算每元素存储开销，以及相对 8-bit 数据的百分比开销。

<details>
<summary>题 1 解答</summary>

每元素 scale 开销= $8/32 = 0.25 bit$；相对数据本身（8 bit）：$0.25/8 = 3.125\%$。
结论：MX 格式用$\sim 3\%$的存储开销换来每块独立的动态范围，性价比很高。
</details>

### 题 2（计算）：H100 上的 decode 上限

H200 带宽 4.8 TB/s，跑 405B FP8 模型。decode 每 token 的最小搬运时间是多少？上限 TPS 是多少？

<details>
<summary>题 2 解答</summary>

$405e9 \times 1 byte / 4.8e12 \approx 84.4 ms/token \to$上限≈ $11.8 tok/s$。这是纯带宽下限，实际还有 KV cache 读写、激活、kernel 开销，只会更低。这也是为什么 400B 级模型必须上多卡张量并行（10/11 章）。
</details>

### 题 3（手算）：E4M3 的 min normal / max

不用查表，从位布局推导 E4M3 的：min normal、max、min subnormal。

<details>
<summary>题 3 解答</summary>

$bias=7$。$\min normal = 2^{1-7} = 2^{-6}$。max（E4M3FN，$e=15$，$m=110$）= $1.11_{2} \times 2^{15-7} = 1.75\times 256 = 448$。$\min subnormal = 2^{-6} \times 2^{-3} = 2^{-9}$。（注意$e=15$且$m=111$是 NaN；$m=000\sim 110$是有限数。）
</details>

### 题 4（思考）：为什么训练不直接用 FP8 E4M3

既然 FP8 推理这么好，训练（前向+反向）为什么主流还是 BF16/FP32？

<details>
<summary>题 4 解答</summary>

训练梯度跨多个数量级，E4M3 最大 448、精度 12.5%，梯度容易溢出/欠精度；反向的梯度用 E5M2 也仍然太粗，且训练需要 FP32 master weight 做优化器状态。推理只有前向、值域可控、可校准，所以 FP8 够用。FP8 训练（如 DeepSeek-V3）能跑，但需要大量工程（loss scaling、分块缩放）才稳定。
</details>

### 题 5（编程）：格式模拟

用 Python 实现一个通用 FP 格式模拟器（输入 E、M、bias、是否有 ∞），输出$\max/\min normal/\min subnormal/\varepsilon$，并用它复现 03 章速查表。

<details>
<summary>题 5 解答要点</summary>

核心公式：$\max = (2 - 2^{-M}) \times 2^{emax - bias}$；$\text{min\_normal} = 2^{1 - bias}$；$\text{min\_subnormal} = 2^{1-bias-M}$；$\varepsilon = 2^{-M}$。注意 E4M3FN 的$emax=15$特例（无 ∞），$E5M2 emax=30$。跑完应得到 448 / 57344 / 6.1e−5 等速查表数值。
</details>

---

## 12. 延伸阅读

1. [FP8 Formats for Deep Learning（NVIDIA/Arm/Intel，arXiv:2209.05433）](https://arxiv.org/abs/2209.05433)：E4M3/E5M2 规格与使用建议的权威来源
2. [OCP Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)：MXFP8/MXFP4 的正式规范
3. [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference（NVIDIA 官方博客）](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)：NVFP4 块 16 + E4M3 缩放的动机
4. [Inference Engineering Ch5](https://inferenceengineering.tech/chapters/techniques/)：格式总览表与"FP8 甜点"结论
5. 上一篇：[02 量化问题的形式化与均匀量化理论](./LLM量化精读笔记-02-量化问题形式化与均匀量化理论.md)；下一篇：**[04 量化粒度、校准与离群值](./LLM量化精读笔记-04-量化粒度校准与离群值.md)**——把"格式选好了"变成"每个张量的 scale 怎么选、outlier 怎么对付"。
