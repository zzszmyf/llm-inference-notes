# LLM 量化精读笔记 · 09 QAT 与训练内量化：STE、QLoRA、BitNet b1.58

> 对应：STE（Bengio et al. 2013；直通估计器）；QLoRA（arXiv:2305.14314，NeurIPS 2023）；BitNet b1.58（arXiv:2402.17764）；MIT 6.5940 Lecture 6（Quantization Part II）。
> 学完本章你应该能：① 说明量化函数为什么不可导、STE 怎么绕过去，并手推一个反向传播例子；② 讲清 QLoRA 的三件套（NF4、双重量化、paged optimizer）和"量化基座 + LoRA 适配器"的架构；③ 说出 BitNet b1.58 的三值化设计（{-1,0,+1}、1.58 bit、无乘法）与性能结论；④ 对比 PTQ/QAT/QLoRA/BitNet 四条路线的适用场景。

---

## 目录（本章）

1. 本章目标
2. 为什么需要训练侧参与
3. STE：让量化可训练
4. QAT 的标准流程与代价
5. QLoRA：量化 × 微调
6. BitNet b1.58：为 1-bit 而生的架构
7. 四条路线对比
8. 本章小结
9. 习题与解答
10. 延伸阅读

---

## 1. 本章目标

04–08 章全是 **PTQ**（训练后量化）：模型训练完，再想办法把误差压下去。本章转向**训练侧**：让模型在训练时就"学会忍受"量化。这会回答一个 PTQ 答不了的问题：

> 如果量化到 4-bit 以下、甚至 1.58-bit，什么方法还能保住质量？

三条路线的分工：

QAT（STE）$\to$标准做法：前向模拟量化，反向用直通估计器
$QLoRA \to$量化基座 + 高精度 LoRA 适配器：4-bit 微调超大模型
$BitNet b1.58 \to$架构级：三值权重 + 无乘法矩阵乘，从头训练

---

## 2. 为什么需要训练侧参与

PTQ 的隐含假设：**量化只是给训练好的权重加噪声，模型必须"生受"**。位宽越低，这个假设越不成立：

4-bit：PTQ 还能靠 GPTQ/AWQ 补救
3-bit：只有少数方法勉强（SqueezeLLM）
2-bit：PTQ 基本失守（QuIP# 靠"洗牌"续命）
1.58-bit：必须从训练时就让模型适应

QAT 的核心思想：**在训练前向里"假量化"（$\operatorname{quantize} \to \operatorname{dequantize}$），让 loss 直接看到量化噪声，梯度因此学会往"量化后依然好"的方向更新。**

---

## 3. STE：让量化可训练

### 3.1 问题：量化函数几乎处处不可导

对称量化的前向：

$$
\hat{x} = \operatorname{clamp}(\operatorname{round}(x/s), q_{\min}, q_{\max}) \cdot s
$$

`round` 是分段常数函数：除了跳变点，导数处处为 0。如果直接反向传播，梯度全为零，训练根本动不了。

### 3.2 直通估计器（Straight-Through Estimator）

STE 的做法：**前向照常量化，反向把量化器当成恒等函数**：

前向：$\hat{x} = \operatorname{clamp}(\operatorname{round}(x/s), q_{\min}, q_{\max}) \cdot s$
反向：$\partial L/\partial x \approx \partial L/\partial \hat{x}$（在量化范围内）

直觉：量化误差（几十分之一）相对梯度噪声是小事，梯度"假装"没有量化，直接穿过；模型因此既能正常更新，又在前向里尝到量化的苦头。

### 3.3 手推一个反向传播例子

设$x = 2.3$，$s = 1$（简化），前向：

$$
\hat{x} = \operatorname{round}(2.3) = 2
$$

反向时 STE 令：

$$
\partial L/\partial x = \partial L/\partial \hat{x} \times 1 = \partial L/\partial \hat{x}
$$

对比精确导数（0）与 STE（1）：STE 保留了"x 增大应该影响输出"的信息，哪怕量化的跳变把局部导数掩盖了。

边界处理：x 超出 [qmin·s, qmax·s] 时，clamp 把梯度截到边界（或直接传 0），防止训练把权重推到饱和区。

### 3.4 进阶：可学习 scale（LSQ 思路）

标准 STE 的 s 是超参；LSQ（Learned Step Size）把 s 也变成可学习参数：

$\partial L/\partial s = \partial L/\partial \hat{x} \times (\partial \hat{x}/\partial s)$（用 STE 近似）

训练中学到的 s 比手调更贴合数据，QAT 质量进一步上升。

---

## 4. QAT 的标准流程与代价

### 4.1 流程

1. 取预训练模型
2. 在所有线性层/注意力层插入 fake-quant（quantize+dequantize）
3. 用任务数据微调（几百到几千步），前向带量化、反向 STE
4. 训练结束：把 fake-quant 替换成真量化权重部署

### 4.2 与 PTQ 的成本对比

| | PTQ | QAT |
|---|---|---|
| 训练数据 | 不需要（只需校准集） | 需要 |
| 算力 | 分钟$\sim$小时 | 小时$\sim$天 |
| 质量上限 | 低（模型生受） | 高（模型适应） |
| 位宽下限 |$\sim 3-bit$（极限 2-bit） | 2-bit 可行、1.58-bit 可行 |
| 工程复杂度 | 低 | 中（训练管线改造） |

**工程策略**：先 PTQ 上线；质量不达标时先换更好的 PTQ（AWQ/QuIP#）；最后才考虑局部 QAT。

---

## 5. QLoRA：量化 × 微调

### 5.1 动机

大模型微调最贵的是**优化器状态和梯度**（动辄几十 GB）。QLoRA 的思路：**基座模型用 4-bit 冻结存储，只训练注入的高精度 LoRA 适配器**——单卡就能微调 65B。

### 5.2 三件套

**① NF4：4-bit NormalFloat（信息论最优码本）**

普通 INT4/FP4 的网格对权重分布不是最优。NF4 用**标准正态分布的分位数**做码本：

16 个码字= $7$个负值 + 0 + 8 个正值（非对称）
码字位置由 N(0,1) 的分位数决定$\to$信息论上对近似正态的权重最优

相比均分网格，同样的 4-bit 在权重分布最密集的地方码字更密。

**② 双重量化（Double Quantization）**

NF4 每个 block（64 个元素）配一个 FP32 scale。这些 scale 本身也可以量化：

第一层：NF4 权重（4-bit） + FP32 block scale
第二层：把 64 个 FP32 scale 再量成 FP8（每组 256 个 scale 共享一个 FP32 二级 scale）
收益：平均每参数省约 0.37 bit

**③ Paged Optimizer**

优化器状态峰值（如长序列的梯度 checkpoint）会瞬间爆显存；用 GPU 统一内存的分页机制把峰值"换页"出去，避免 OOM。

### 5.3 架构：量化基座 + LoRA

基座：NF4 冻结（4-bit，前向时反量化到 BF16 计算）
LoRA：低秩适配器（A、B 两个小矩阵），BF16 可训练
输出：$Y = W_4bit\cdot X + (B\cdot A)\cdot X$

### 5.4 结果

- **65B 模型在单张 48GB GPU 上微调**（此前需要多卡/数百 GB）。
- Guanaco-65B 达到 ChatGPT 在 Vicuna benchmark 上 **99.3%** 的水平。
- 微调质量与 16-bit LoRA 相当（论文在多项任务上验证）。

> 意义：量化从"部署工具"变成"训练工具"——**QLoRA 证明低精度存储不损失可学性**，是开源社区微调的事实标准之一。

---

## 6. BitNet b1.58：为 1-bit 而生的架构

### 6.1 三值权重

BitNet b1.58 把**每个权重**限制为三个值：

$$
w \in \{-1, 0, +1\}
$$

每个权重 1.58 bit（= $\log_2(3)$）

与 BitNet b1（二值 {-1,+1}）相比，**多了 0**：

0 的价值：
1. 更接近真实权重分布（很多权重本来就接近 0）
2. 隐含稀疏性（部分权重"关闭"）
3. 相同模型大小下质量显著提升，达到与 FP16 相当

### 6.2 无乘法矩阵乘

三值权重与激活相乘：

$$
y = \sum w_{i} \cdot x_{i}, w_{i} \in \{-1, 0, +1\}
$$

$\to$只需要加法和减法，不需要乘法！

实现（BitNet.cpp 等）：激活先量化为 8-bit，再用三值权重的符号决定加/减。论文估算（7nm）：矩阵乘算术能耗约为 FP16 的 **1/71.4**。

### 6.3 训练：从头 QAT

BitNet 不能从 FP16 模型"转换"而来，必须**从零训练**：

1. 权重初始化为 FP16，前向时三值化（round 到 {-1,0,+1}）+ 缩放因子
2. 反向用 STE（第 3 节）
3. 激活 8-bit 量化（类似 SmoothQuant 的思路）
4. 训练稳定后，部署时只剩加/减

### 6.4 结果

质量：相同模型大小与训练 token 下，匹配 FP16 LLaMA 的 perplexity 与下游任务
速度：比同规模 FP16 快约 2.71x（论文估计）
内存：约省 3.55x
能耗：矩阵乘算术能耗约 1/71.4（7nm 估算）
规模效应：模型越大，收益越明显

> 意义：BitNet 说明"量化"不一定要当 PTQ 的补救手段，也可以是**架构的起点**——1-bit 时代的 LLM 设计会反过来重塑硬件（专用 1-bit GEMM 单元）。

---

## 7. 四条路线对比

| | PTQ | QAT | QLoRA | BitNet b1.58 |
|---|---|---|---|---|
| 是否需要训练 | 否 | 是（微调） | 是（LoRA） | 是（从头） |
| 数据需求 | 校准集 | 任务数据 | 任务数据 | 海量预训练数据 |
| 位宽 | 3–4-bit 起步 | 2-bit | 4-bit 基座 | 1.58-bit |
| 质量上限 | 低 | 高 | 高（受限于基座） | 高（重新训练） |
| 训练成本 | 分钟$\sim$小时 | 小时$\sim$天 | 单卡小时级 | 预训练级 |
| 典型用途 | 生产推理默认 | 低比特精度冲刺 | 单卡微调大模型 | 未来架构探索 |

选型：
  部署已有模型$\to PTQ$（AWQ/GPTQ）
  部署 + 质量不够$\to$局部 QAT
  要微调大模型$\to QLoRA$
  要 1-bit 极限$\to BitNet$（接受重训成本）

---

## 8. 本章小结

1. **STE**：前向量化、反向恒等——让不可导的量化变得可训练（配合 LSQ 可学 scale）。
2. **QAT**：模型在训练中"学会忍受"量化，质量上限高于 PTQ，代价是数据和算力。
3. **QLoRA**：NF4 分位数码本 + 双重量化 + paged optimizer，量化基座 + BF16 LoRA，单卡微调 65B。
4. **BitNet b1.58**：三值权重 {-1,0,+1}（1.58 bit）、无乘法矩阵乘、从头 QAT，质量匹敌 FP16、能耗降一个数量级。
5. **一条主线**：位宽越低，越需要"训练侧参与"——从 PTQ 的补救，到 QAT 的适应，再到 BitNet 的架构重构。

> 一句话记忆：**"PTQ 让模型硬扛噪声，QAT 让模型学会挨打，QLoRA 让噪声只在底座、精细活交给 LoRA，BitNet 干脆把噪声变成架构。"**

---

## 9. 习题与解答

### 题 1（手算）：STE 反向

$x = 3.7$，$s = 1$，前向$\hat{x} = \operatorname{round}(3.7) = 4$。若$\partial L/\partial \hat{x} = 0.5$，STE 下$\partial L/\partial x$是多少？真实导数是多少？为什么 STE 可用？

<details>
<summary>题 1 解答</summary>

STE：$\partial L/\partial x = 0.5$。真实导数：round 在$x=3.7$处导数为 0。STE 可用因为它保留"增大 x 会增大输出"的方向信息；量化跳变点的局部导数（0 或 ∞）对训练没有统计意义，且训练中大量样本的 STE 期望近似于真实梯度。
</details>

### 题 2（推导）：1.58 bit 怎么来的

证明三值权重每参数 1.58 bit，并解释为什么它比"1-bit + 稀疏位"的混合更省。

<details>
<summary>题 2 解答</summary>

三个等概率状态需要$\log_2(3) \approx 1.585 bit$。若用"1 bit 符号 + 1 bit 是否为零"则需要 2 bit（且 4 个码字浪费 1 个）。三值直接编码是最紧凑的表示。
</details>

### 题 3（对比）：NF4 vs INT4

为什么 NF4 的码本对 LLM 权重通常优于均匀 INT4？

<details>
<summary>题 3 解答</summary>

LLM 权重近似零均值正态分布：大量值集中在 0 附近。均匀网格在 0 附近只有少量码字（02 章"范围利用率"）；NF4 用标准正态分位数布点，0 附近码字密、尾部疏，同样的 16 个码字信息量更大。
</details>

### 题 4（设计）：给 BitNet 写部署要点

BitNet 推理时，矩阵乘的乘法器可以完全去掉吗？还有哪些地方仍是乘法/高精度？

<details>
<summary>题 4 解答要点</summary>

权重×激活的乘法可退化为加/减（按权重符号），但：① 激活仍要 8-bit 量化（需要 scale 乘法）；② 归一化、残差、输出投影仍可能是高精度；③ 注意力（softmax、QK）仍需乘法。所以"无乘法"主要指权重矩阵乘主体，不是全模型。
</details>

### 题 5（开放）：量化 + 微调的组合矩阵

画出"位宽 × 是否训练"的 2×3 矩阵（4-bit/2-bit/1.58-bit × PTQ/QAT/从头训练），各填一个代表方法与一句话判断。

<details>
<summary>题 5 解答要点</summary>

4-bit：AWQ（PTQ）✓ 生产可用；QAT 4-bit（更稳）；从头 4-bit（少见）。
2-bit：QuIP#（PTQ，极限）；QAT 2-bit（可行）；BitNet 类（更适合）。
1.58-bit：PTQ 无解；QAT 勉强；BitNet b1.58（唯一成熟路线）。
结论：位宽越低，横轴（训练参与度）必须越大。
</details>

---

## 10. 延伸阅读

1. [QLoRA（arXiv:2305.14314）](https://arxiv.org/abs/2305.14314)：NF4、双重量化、paged optimizer
2. [The Era of 1-bit LLMs: BitNet b1.58（arXiv:2402.17764）](https://arxiv.org/abs/2402.17764)
3. [BitNet.cpp](https://github.com/microsoft/BitNet)：三值权重的 CPU 推理实现（加/减内核）
4. Bengio et al., *Estimating or Propagating Gradients Through Stochastic Neurons*（2013）：STE 的源头
5. 上一篇：[08 KV Cache 量化](./LLM量化精读笔记-08-KV-Cache量化与KIVI.md)；下一篇：**[10 质量评估方法论]**——学完所有方法，怎么科学地判断"量化有没有搞砸"。
