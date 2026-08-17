# LLM 量化精读笔记 · 06 权重量化 II：AWQ、SqueezeLLM、QuIP#

> 对应：AWQ（arXiv:2306.00978，MLSys 2024 Best Paper）；SqueezeLLM（arXiv:2306.07629，ICML 2024）；QuIP#（arXiv:2402.04396，ICML 2024）；MIT 6.5940 Lecture 5。
> 学完本章你应该能：① 复述 AWQ 的核心观察（1% 显著权重、按激活幅度而非权重幅度识别）并推导"缩放减误差"的公式；② 说清 AWQ 与 GPTQ 的取舍（无重建 vs 二阶补偿）；③ 讲出 SqueezeLLM 的两板斧（敏感度非均匀量化 + 稠密/稀疏分解）；④ 讲出 QuIP# 为什么能到 2-bit（Hadamard 非相干 + E8 格码本）。

---

## 目录（本章）

1. [本章目标](#1-本章目标)
2. [为什么还需要"第二种"权重量化](#2-为什么还需要第二种权重量化)
3. [AWQ：激活感知的权重量化](#3-awq激活感知的权重量化)
4. [SqueezeLLM：敏感度非均匀量化 + 稠密/稀疏分解](#4-squeezellm敏感度非均匀量化--稠密稀疏分解)
5. [QuIP#：先洗牌、再用格码本](#5-quip先洗牌再用格码本)
6. [三方法对比](#6-三方法对比)
7. [本章小结](#7-本章小结)
8. [习题与解答](#8-习题与解答)
9. [延伸阅读](#9-延伸阅读)

---

## 1. 本章目标

05 章的 GPTQ 证明了"误差可以补偿"，但它有三个短板：**要逐层重建（慢）、校准集可能过拟合、2-bit 依然崩**。本章三条路线分别针对这三个短板：

$AWQ \to $不重建，几分钟完成；用"激活幅度"代替 Hessian
$SqueezeLLM \to $用敏感度 + 非均匀量化 + outlier 分离，冲 3-bit
QuIP#$\to $用"非相干变换"把误差变成白噪声，冲 2-bit

---

## 2. 为什么还需要"第二种"权重量化

GPTQ 的问题（论文原话与后续实验）：

1. **重建开销**：逐层缓存激活 + 线性代数求解，175B 要 4 GPU 小时；对"快速试错"不友好。
2. **校准集过拟合**：GPTQ 在重建时最小化校准集上的损失，可能把权重"扭曲"成只对校准分布最优，在分布外（指令微调模型、多模态）掉点。AWQ 论文专门拿这个做对比（其 Figure 8）。
3. **2-bit 崩**：二次假设在大误差下失效（05 章 9.1）。

此外，AWQ 提出了一个更本质的问题：**权重重要性不相等，而且"谁重要"要问激活，而不是权重自己。**

---

## 3. AWQ：激活感知的权重量化

### 3.1 观察一：1% 的显著权重决定成败

AWQ 论文做了个实验：INT3（$group=128$）量化后，把**部分通道保留为 FP16**（混合精度），看保留谁最有效：

OPT-13B, INT3-g128, WikiText-2 PPL（FP16 基线 10.13）：
  全量化（RTN）$\to 46.04$
  保留 1% 通道为 FP16（按激活幅度选）$\to 10.51 \leftarrow $几乎无损
  保留 1% 通道为 FP16（按权重幅度选）$\to 48.96 \leftarrow $没用
  保留 1% 通道为 FP16（随机选）$\to 42.00 \leftarrow $没用

**结论**：显著通道只有$\sim 1\%$（甚至 0.1%），必须按**输入激活的幅度**识别——激活大的通道处理的是重要特征。

### 3.2 观察二：混合精度硬件不友好，但可以"数学等效"

混合精度（1% FP16 + 99% INT3）在硬件上很难做（内存布局、kernel 分支）。AWQ 的替代：**用 per-channel 缩放模拟"保护"的效果，而不真正保留 FP16**。

### 3.3 核心推导：缩放为什么能减误差

考虑一组权重 w 和激活 x。对称量化（组内$scale \Delta = \max|w|/2^{N-1}$）：

原始：$Q(w)\cdot x = \Delta \cdot \operatorname{Round}(w/\Delta )\cdot x$
缩放：$Q(w\cdot s)\cdot (x/s) = \Delta ′\cdot \operatorname{Round}(ws/\Delta ′)\cdot x\cdot (1/s)$

两个经验事实（论文 Table 2 背后的分析）：

1. $RoundErr(\cdot ) \approx 0.25$（均匀分布在$[0, 0.5]$），缩放不改变它
2. 缩放单个（少数）元素通常不改变组的$\max \to \Delta ′ \approx \Delta $

于是误差：

原始误差≈ $\Delta \cdot 0.25\cdot x$
缩放后误差≈ $\Delta ′\cdot 0.25\cdot x\cdot (1/s) \approx \Delta \cdot 0.25\cdot x/s$

**结论：把显著通道的权重放大 s 倍、激活除以 s，在组内 max 基本不变的前提下，显著通道的量化误差约缩小 1/s。** 这就是"等效保护"。

代价：s 太大时，被放大的通道会成为组的 max（$\Delta ′ > \Delta $），反而伤到同组其他通道。论文在 OPT-6.7B INT3-g128 上扫 s：

| s |$\Delta ′\ne \Delta $的比例 | 平均$\Delta ′/\Delta \cdot (1/s)$| Wiki-2 PPL |
|---|---|---|---|
| 1 | 0% | 1.0 | 23.54 |
| 1.25 | 2.8% | 0.804 | 12.87 |
| 1.5 | 4.4% | 0.676 | 12.48 |
| **2** | 8.2% | 0.519 | **11.92** |
| 4 | 21.2% | 0.303 | 12.36 |

甜点在$s \approx 2$：显著通道误差减半，同时只有$\sim 8\%$的组被"顶到新 max"。

### 3.4 算法：怎么选 s

1. 用校准集前向，记录每层输入激活 X
2. 对每个输入通道 j：$s_{j} = (\max|X_{j}|)^\alpha $
3. 网格搜索$\alpha \in \{0, 0.05, ..., 1\}$，选使量化后校准损失最小的$\alpha $*
4. 应用：$W \leftarrow W\cdot \operatorname{diag}(s)$，$X \leftarrow X\cdot \operatorname{diag}(s)^{-1}$（权重放大、激活缩小）
5. 对缩放后的 W 做常规 group 量化（$group=128$）

要点：

- **无梯度、无重建**：量化一个 70B 模型只要几分钟（GPU），远快于 GPTQ。
- **scale 折叠**：s 在推理时被折叠进 per-group 的 scale，几乎零开销（TinyChat 的 kernel 做了这件事）。
- **校准集过拟合风险低**：因为只用了激活的 max（一阶统计），不求解权重。

### 3.5 结果与落地

- 4-bit（W4A16）在 LLaMA/OPT 全系列追平或超过 GPTQ；3-bit 显著优于 RTN/GPTQ。
- 指令微调模型（Vicuna）与多模态模型（OpenFlamingo）表现好（泛化优势）。
- TinyChat 框架：桌面/移动 GPU 上比 HF FP16 快 3.2–3.3x；Llama-2-70B 跑在 Jetson Orin 上。
- 被 HuggingFace Transformers、TensorRT-LLM、vLLM、LMDeploy 等全面采用。

### 3.6 AWQ vs GPTQ

| | GPTQ | AWQ |
|---|---|---|
| 机制 | Hessian 二阶补偿（05 章） | 激活幅度 per-channel 缩放 |
| 重建 | 需要 | 不需要 |
| 量化时间 | 小时级（175B） | 分钟级 |
| 校准集过拟合风险 | 有 | 低 |
| 2-bit 表现 | 崩 | 崩 |
| 硬件友好性 | 需要特殊 kernel（重排） | scale 折叠，天然友好 |
| 核心风险 | 分布外退化 | 只保护"幅度"大的通道，忽略相关性 |

---

## 4. SqueezeLLM：敏感度非均匀量化 + 稠密/稀疏分解

### 4.1 动机

均匀量化的两个先天问题（02/04 章）：① 权重分布不均匀，等距网格浪费码字；② outlier 权重视觉上少、影响上大。SqueezeLLM 用**非均匀量化 + 稀疏分离**同时解决。

### 4.2 敏感度：谁值得更精细的网格

沿用 OBS/GPTQ 的二阶思想，对每个权重算敏感度：

$$
S_{\text{ij}} = w_{\text{ij}}^{2} / (2\cdot [H^{-1}]_ii)
$$

（H 是层输入的 Hessian 对角近似；公式来源与 05 章$\Delta L = \frac{1}{2}(w-\hat{w})^{2}/[H^{-1}]_{qq}$同源。）

敏感度高的权重：量化它的损失大$\to $值得"特殊照顾"。

### 4.3 敏感度感知的非均匀量化

不用等距网格，而是用 **k-means 聚类**在敏感度高的区域多放码字：

1. 计算每个权重的敏感度$S_{\text{ij}}$
2. 用敏感度加权的 k-means 找$2^b$个聚类中心（码本）
3. 每个权重用最近的码字表示，索引存储

效果：分布集中处（通常也是敏感处）码字密，尾部稀疏。

### 4.4 稠密-稀疏分解：outlier 的最终归宿

即使有非均匀码本，极少数 outlier 依然会拉坏一切。SqueezeLLM 把它们**物理分离**：

权重 = 稠密部分（99%+，低比特量化，如 3/4-bit）
     + 稀疏部分（$\sim 0.01\%-0.1\%$，直接存 FP16 原值）

稀疏部分用稀疏矩阵存储（索引 + 值），存储开销很小（0.1% × 16 bit vs 99.9% × 3 bit）。推理时两条路径分别算，再相加。

### 4.5 结果

- **3-bit 追平 FP16**（LLaMA-7B/13B/30B，WikiText-2 PPL），超过 GPTQ 3-bit。
- **4-bit 与 FP16 基本无差**，是"4-bit 权重的强基线"之一。
- 与 AWQ 思路互补：AWQ 保护"通道"，SqueezeLLM 保护"个体权重"。

---

## 5. QuIP#：先洗牌、再用格码本

### 5.1 动机：2-bit 为什么崩

2-bit 只有 4 个码字，均匀量化等于"一个阈值切 4 段"。误差不再是小扰动，而是结构性失真——尤其在权重存在离群时。**要降到 2-bit，得让权重分布"对量化友好"。**

### 5.2 非相干处理（Incoherence Processing）

QuIP 系列（Chee et al. 2023）的关键概念：量化误差取决于权重与单位向量的"内积集中度"。如果权重矩阵**相干**（少数大分量），量化误差就集中在少数方向；如果**非相干**（分量均匀），量化误差像白噪声一样分散，对输出的伤害最小。

做法：用**随机 Hadamard 变换（RHT）**给权重"洗牌"：

$$
W \leftarrow D_{1} H W H D_{2}
$$

（D 是随机对角符号，H 是 Hadamard 矩阵；变换可逆、计算 O(n log n)，且可以折叠到相邻层）

洗牌后：权重各分量幅度趋于均匀$\to $量化误差各向同性$\to 2-bit$从"结构性破坏"变成"噪声级扰动"。

### 5.3 E8 格码本：2-bit 的最优打包

单纯标量量化（每权重独立找最近点）在 2-bit 下浪费严重。QuIP# 用 **8 维格（lattice）码本**：

把 8 个权重打包成一个 8 维向量
用 E8 格（8 维空间里 packing 最优的格之一）的码字近似

E8 格的性质：码字间的最小距离在 8 维格中最大$\to $同样的位预算下失真最小（经典编码理论的结论）。推理时用查表（LUT）快速解码。

### 5.4 结果

- **2-bit 达到当时 SOTA**（LLaMA-2 70B 2-bit PPL 接近 3-bit GPTQ）。
- 3-bit/4-bit 也强，但优势主要体现在 2-bit 极限区间。
- 代价：实现复杂度高（Hadamard 变换、格解码、LUT kernel），生态采用晚于 GPTQ/AWQ。

---

## 6. 三方法对比

| | GPTQ | AWQ | SqueezeLLM | QuIP# |
|---|---|---|---|---|
| 补偿机制 | Hessian 二阶更新 | 激活幅度缩放 | 敏感度码本 + 稀疏分离 | 非相干变换 + 格码本 |
| 是否需要重建 | 是 | 否 | 是（k-means） | 是（优化） |
| 量化时间 | 小时级 | 分钟级 | 中等 | 中等 |
| 甜点位宽 | 3–4 bit | 4 bit | 3–4 bit | **2–3 bit** |
| 校准数据 | 需要 | 需要（只需激活统计） | 需要 | 需要 |
| 硬件实现 | 中 | 低（scale 折叠） | 中 | 高（LUT/Hadamard） |
| 生态 | vLLM/TensorRT/HF | vLLM/TensorRT/HF | 部分 | 研究为主 |

一句话选型：
  要快、要稳、要生态$\to AWQ$（W4A16 事实标准）
  要 3-bit 极限质量$\to SqueezeLLM / GPTQ$
  要 2-bit 探索$\to QuIP$#

---

## 7. 本章小结

1. **AWQ**：1% 显著权重（按激活幅度）决定成败；用 $s = \max|X|^\alpha $ 缩放，数学上等效"保护"，无重建、分钟级、硬件友好——W4A16 的事实标准。
2. **SqueezeLLM**：敏感度（Hessian 对角）驱动非均匀码本 + outlier 稀疏分离，3-bit 追平 FP16。
3. **QuIP#**：Hadamard 把权重洗成"非相干"（误差白噪声化）+ E8 格码本打包，2-bit 极限区间 SOTA。
4. **共同主线**：都是"不均匀对待权重"——要么按通道（AWQ）、要么按个体（SqueezeLLM）、要么先把分布变均匀（QuIP#）。

> 一句话记忆：**"AWQ 给重要通道开后门，SqueezeLLM 给重要权重单独加座，QuIP# 干脆先让大家长得一样再坐。"**

---

## 8. 习题与解答

### 题 1（推导）：AWQ 的误差公式

从$Q(w\cdot s)\cdot (x/s) = \Delta ′\cdot \operatorname{Round}(ws/\Delta ′)\cdot x/s$出发，写出误差表达式，并解释$\Delta ′\approx \Delta $时为什么误差约缩小 1/s。

<details>
<summary>题 1 解答</summary>

真实输出 w·x；量化输出$\Delta ′\cdot \operatorname{Round}(ws/\Delta ′)\cdot x/s$。误差= $|w - \Delta ′\cdot \operatorname{Round}(ws/\Delta ′)/s|\cdot x = (\Delta ′/s)\cdot RoundErr(ws/\Delta ′)\cdot x$。若$\Delta ′\approx \Delta $：误差≈ $\Delta \cdot RoundErr\cdot x/s$，即原始误差（$\Delta \cdot RoundErr\cdot x$）除以 s。
</details>

### 题 2（思考）：为什么 s 不能无限大

用 3.3 的表格解释：$s=4$时"平均误差更小（0.303）"，为什么 PPL 反而回升？

<details>
<summary>题 2 解答</summary>

$s=4$时 21.2% 的组被缩放后的权重顶出新 max（$\Delta ′>\Delta $），同组非显著通道的网格变粗，误差上升；显著通道省下的误差 < 非显著通道失去的误差，总损失变大。缩放是"转移误差预算"，不是"消灭误差"。
</details>

### 题 3（对比）：AWQ vs SqueezeLLM 的保护粒度

同样是"保护重要的东西"，AWQ 和 SqueezeLLM 的保护对象和粒度有什么不同？

<details>
<summary>题 3 解答</summary>

AWQ 按**输入通道**（激活幅度大 = 通道重要），整体缩放该通道；SqueezeLLM 按**单个权重**（Hessian 敏感度大 = 个体重要），非均匀码本 + 稀疏 FP16 分离。AWQ 粒度粗但零开销、硬件友好；SqueezeLLM 粒度细但需要稀疏 kernel。
</details>

### 题 4（推导）：QuIP# 为什么有效

用"误差方向"的语言解释：为什么把权重矩阵变得非相干，能降低量化对输出的破坏？

<details>
<summary>题 4 解答要点</summary>

输出= $Wx$，量化误差$\varepsilon $对输出的影响= $\varepsilon ^{T}x$。相干矩阵的$\varepsilon $集中在少数坐标（与 x 的少数分量强相关），伤害大且不可预测；非相干后$\varepsilon $的各分量独立、幅度均匀，与 x 的内积像随机噪声，期望影响小、可被后续层平均掉。这就是"把结构性误差变成白噪声"。
</details>

### 题 5（实践）：给一个 7B 模型选权重量化方案

场景：要在手机端跑 7B 聊天模型，量化时间预算 30 分钟，生态要成熟。选 AWQ 还是 GPTQ？为什么？如果要 2-bit 进一步省内存呢？

<details>
<summary>题 5 解答要点</summary>

选 AWQ：分钟级、无重建、TinyChat/llama.cpp 生态成熟、指令微调模型泛化好。2-bit 则 QuIP# 类方法更合适（质量最好），但要接受 kernel 不成熟；工程上通常仍留在 3–4-bit（AWQ/SqueezeLLM）。
</details>

---

## 9. 延伸阅读

1. [AWQ（arXiv:2306.00978）](https://arxiv.org/abs/2306.00978)：本章主文献；[官方代码](https://github.com/mit-han-lab/llm-awq)
2. [SqueezeLLM（arXiv:2306.07629）](https://arxiv.org/abs/2306.07629)
3. [QuIP#（arXiv:2402.04396）](https://arxiv.org/abs/2402.04396)；QuIP 原版（NeurIPS 2023）
4. [AWQ 深度解读（GeneralCompute）](https://www.generalcompute.com/blog/activation-aware-quantization-awq-deep-dive)：公式与实现的对照
5. 上一篇：[05 权重量化 I：RTN 与 GPTQ](./LLM量化精读笔记-05-权重量化I-RTN与GPTQ.md)；下一篇：**[07 激活量化：LLM.int8() 与 SmoothQuant]**——把战场从权重扩展到激活（W8A8）。

