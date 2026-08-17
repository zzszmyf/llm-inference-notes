# LLM 推理优化 · 量化（Quantization）系统学习笔记

> 整理日期：2026-08-16
> 素材来源：
> - Inference Engineering（Baseten 出品）Ch 5: Quantization & Speculative Decoding：https://inferenceengineering.tech/chapters/techniques/
> - 配套交互练习 Quantization Quality Estimator：https://inferenceengineering.tech/exercises/quantization-estimator/
> - Han Song（韩松）MIT 6.5940 TinyML and Efficient AI Computing（EfficientML.ai）课程
> - 业界论文：GPTQ / AWQ / SmoothQuant / LLM.int8() / SqueezeLLM / QuIP# / KIVI / QServe / QLoRA / BitNet b1.58 等

---

## 目录

1. [怎么用这份笔记](#1-怎么用这份笔记)
2. [为什么学量化：收益与风险](#2-为什么学量化收益与风险)
3. [量化为什么能加速：Prefill 与 Decode](#3-量化为什么能加速prefill-与-decode)
4. [数字格式全景](#4-数字格式全景)
5. [量化的基本数学与粒度](#5-量化的基本数学与粒度)
6. [量化对象与敏感性排序](#6-量化对象与敏感性排序)
7. [方法谱系：从 RTN 到 1-bit](#7-方法谱系从-rtn-到-1-bit)
8. [质量评估三板斧](#8-质量评估三板斧)
9. [交互练习实测：Quantization Quality Estimator](#9-交互练习实测quantization-quality-estimator)
10. [生产建议与决策清单](#10-生产建议与决策清单)
11. [Han Song MIT 6.5940 课程地图](#11-han-song-mit-65940-课程地图)
12. [论文阅读清单](#12-论文阅读清单)
13. [工具与生态](#13-工具与生态)
14. [术语速查](#14-术语速查)
15. [参考资料](#15-参考资料)

---

## 1. 怎么用这份笔记

这是一份"教材 + 课程 + 论文"三合一的量化学习路线：

1. **先读正文（第 2–8 节）**：建立量化为什么存在、怎么工作、什么时候会翻车的完整框架，来源是 Inference Engineering Ch 5。
2. **动手玩练习（第 9 节）**：打开 [Quantization Quality Estimator](https://inferenceengineering.tech/exercises/quantization-estimator/)，对照笔记里的实测表自己调参数。
3. **再看 MIT 课程（第 11 节）**：Han Song 的 6.5940 第 5/6 讲讲量化原理与 PTQ/QAT，第 13 讲讲 LLM 量化部署，视频和幻灯片全公开。
4. **按顺序读论文（第 12 节）**：先读方法谱系里"一环扣一环"的 5 篇核心论文，再按兴趣扩展。

---

## 2. 为什么学量化：收益与风险

一句话定位：**量化是 LLM 推理优化里收益最直接、风险也最"可见"的一招。**

- **收益**：同时改善 TTFT（首 Token 延迟）和 TPS（吞吐），提升吞吐，还为其他优化腾出显存/带宽余量。
- **风险**：一旦量化失当，会实质性地降低输出质量（perplexity 变差、推理/代码任务退化），而且错误是"静默"的——模型不报错，只是变笨。
- **关键事实**：实践中，精度每降低一级，LLM 性能大约提升 **30–50%**。

> 章节原话的 Key Takeaway：FP8/MXFP8 是生产环境的甜点区间。权重和激活用 FP8 量化，KV cache 谨慎量化，attention 保持原精度。如果完全不能承受质量风险，本章其余技术（推测解码、缓存等）都是无损的。

---

## 3. 量化为什么能加速：Prefill 与 Decode

后训练量化（PTQ）把权重从原生格式（通常 BF16/FP16）降到更低精度。加速的来源取决于处于推理的哪个阶段：

| 阶段 | 瓶颈 | 量化带来的收益 |
|---|---|---|
| **Prefill（预填充）** | 计算密集（compute-bound） | 在低精度 Tensor Core 上跑，FLOPS 翻倍（2x） |
| **Decode（逐 Token 生成）** | 访存密集（memory-bound） | 权重数据加载量减半，等效带宽翻倍（2x） |

所以量化对两类延迟都有帮助：TTFT 受益于更快的矩阵乘，TPS 受益于更少的显存搬运。这也是为什么 4-bit 权重 + 8-bit 激活（W4A8）这类组合在服务系统里成为主流方向。

---

## 4. 数字格式全景

### 4.1 格式总览（章节原表）

| 名称 | 位数 | 首个支持架构 | 用途 |
|---|---|---|---|
| FP16 | 16 | Pascal（2016） | 默认推理格式 |
| BF16 | 16 | Ampere（2020） | 训练与推理 |
| FP8（E4M3 / E5M2） | 8 | Hopper（2022） | 质量/速度的甜点 |
| MXFP8 | 8 | Blackwell（2024） | Microscaling，精度更好 |
| FP4 | 4 | Blackwell（2024） | 激进量化 |
| NVFP4 | 4 | Blackwell（2024） | FP4 里精度最好（block size 16） |

### 4.2 位布局

浮点格式 = 符号位（S）+ 指数位（E）+ 尾数位（M）。**指数位决定了动态范围**——这是浮点相对整数的核心优势：更能表达量化后仍然存在的离群值（outliers）。

| 格式 | S | E | M | 相对精度（约） | 动态范围特点 |
|---|---|---|---|---|---|
| FP16 | 1 | 5 | 10 | 高（约 2⁻¹⁰） | 与 FP32 相同的精度区间，范围小（max ~65504） |
| BF16 | 1 | 8 | 7 | 低（约 2⁻⁷） | 与 FP32 相同的动态范围，精度粗 |
| FP8 E4M3 | 1 | 4 | 3 | 中（约 2⁻³） | 范围中等（max 448），无 Infinity，单 NaN |
| FP8 E5M2 | 1 | 5 | 2 | 低（约 2⁻²） | 范围接近 FP16（max ~57344），精度粗 |
| FP4（E2M1） | 1 | 2 | 1 | 很低 | 只有 16 种取值（max 6），必须配合块缩放 |
| NVFP4 | 1 | 2 | 1 | 很低 + 块缩放 | block size 16，FP8 E4M3 块缩放 + FP32 张量缩放 |
| MXFP8 / MXFP4 | 1 | 4/2 | 3/1 | 中/低 + 块缩放 | block size 32，E8M0 共享指数缩放 |

### 4.3 可表示数值数量（章节原表）

| 格式 | 可表示的不同取值 |
|---|---|
| FP16 | 65,536 |
| BF16 | 65,536 |
| FP8（E4M3） | 256 |
| FP8（E5M2） | 256 |
| FP4 | 16 |
| NVFP4 | 16 |

位数越少越快，但风险越高——这是所有格式选择的底层 tradeoff。

### 4.4 FP8 两种变体怎么选（E4M3 vs E5M2）

- **E4M3**：4 位指数 + 3 位尾数，精度更高、范围更小（max 448）。适合**权重和激活**——它们的值通常集中在较小范围，精度更重要。NVIDIA 推荐 E4M3 用于前向（inference/训练 forward）。
- **E5M2**：5 位指数 + 2 位尾数，范围大（接近 FP16）但精度粗。适合**梯度/反向传播**，或值域跨度大的场景。

依据：NVIDIA/Arm/Intel 联合白皮书《FP8 Formats for Deep Learning》（arXiv:2209.05433）。

### 4.5 MXFP8 / MXFP4 与 NVFP4 的区别

- **MX 格式（OCP Microscaling Formats）**：一组元素（block size 32）共享一个 E8M0 指数缩放（纯 2 的幂），元素本身仍是指数+尾数格式。用"一组一个缩放"换更细的适配，Blackwell 原生加速。
- **NVFP4**：NVIDIA 在 Blackwell 上的 4-bit 方案。元素是 E2M1（16 种取值），但 block size 缩小到 **16**，块缩放用 FP8 E4M3（比 E8M0 更细），再加一个 FP32 张量缩放。粒度更细 → 目前 FP4 里精度最好。

一句话：**FP8 = 平衡；MXFP8 = FP8 的精度升级；FP4/NVFP4 = 激进压缩，省 75% 权重显存，但必须接受质量验证。**

---

## 5. 量化的基本数学与粒度

### 5.1 均匀量化公式

对称量化（最常用）：

```
q = clamp(round(x / s), q_min, q_max)
s = max|x| / 2^(b-1)
反量化：x ≈ q * s
```

非对称量化额外有一个 zero-point z：`q = clamp(round(x / s) + z, ...)`。

### 5.2 粒度（Granularity）决定风险与开销

| 粒度 | 一个 scale 覆盖范围 | 质量风险 | 开销 |
|---|---|---|---|
| Tensor-level（整层一个 scale） | 整个权重矩阵 | 高 | 最小 |
| Channel-level（按输出通道） | 每行/每列一个 scale | 中 | 低 |
| Block/Group-level（按块） | 每 16/32/128 个元素一个 scale | 低 | 更高（scale 也要存储） |

**粒度越细，质量损失越小，但 scale 本身的存储和计算开销越大。** GPTQ/AWQ/QuIP# 的 4-bit 能逼近 FP16 质量，很大程度靠的就是细粒度分组 + 精心的 scale 选择。

---

## 6. 量化对象与敏感性排序

Transformer 各组件对量化的敏感度从低到高（章节原表）：

1. **权重（线性层）** —— 最不敏感
2. **激活** —— 有些敏感
3. **KV cache** —— 中等敏感（误差会逐 Token 累积）
4. **Attention（尤其是 softmax）** —— 高度敏感

原因直觉：
- 权重是"静态"的，可以用校准数据精调（GPTQ/AWQ），且 outlier 可以单独处理；
- 激活是"动态"的，每层输入分布未知，LLM 激活还有著名的 outlier 问题（少数维度值巨大），直接量化会毁掉精度（→ SmoothQuant / LLM.int8() 的动机）；
- KV cache 误差会随着序列变长反复参与计算，错误像滚雪球一样累积；
- softmax 的指数运算对数值范围极其敏感，所以生产上通常让 attention 留在高精度。

---

## 7. 方法谱系：从 RTN 到 1-bit

按"要量化什么、怎么克服误差"这条主线，业界方法可以排成一条清晰的谱系：

### 7.1 基线：RTN（Round-to-Nearest）+ 均匀量化

把权重四舍五入到最近的量化网格。实现最简单，但 4-bit 以下质量崩得快，因为忽略了权重的重要性和分布。

### 7.2 权重低比特（Weight-only）

- **GPTQ**（ICLR 2023）：逐层量化 + 用二阶 Hessian 信息做误差补偿（近似最优的逐列更新）。一次性把 OPT-175B/BLOOM-176B 量化到 3–4 bit 只需约 4 GPU 小时，perplexity 几乎不涨。→ **"数学补偿派"**
- **AWQ**（MLSys 2024，Han Lab）：不动模型，只根据激活分布找出约 1% 的显著权重，用 per-channel 缩放保护它们。比 GPTQ 更省事、硬件更友好（无逐列重建）。→ **"保护重要权重派"**
- **SqueezeLLM**（ICML 2024）：基于敏感度的非均匀量化 + 稠密/稀疏分解，把 outlier 权重单独用高精度存。→ **"outlier 单独处理派"**
- **QuIP#**（ICML 2024）：用随机 Hadamard 变换把权重"打散"成对量化友好的分布（incoherence），配合 E8 格码本，2-bit 权重达到当时 SOTA。→ **"先洗牌再量化派"**

### 7.3 权重 + 激活（W8A8）

- **LLM.int8()**（NeurIPS 2022）：INT8 矩阵乘 + 混合精度分解——把激活里有 outlier 的列挑出来留在 FP16 算，其余走 INT8。首个无精度损失的 175B 模型 INT8 推理方案。
- **SmoothQuant**（ICML 2023，Han Lab）：激活难量化、权重好量化，那就通过数学上等价的缩放把"量化难度"从激活迁移到权重（`s = max|X|^α / max|W|^(1−α)`），实现 W8A8 且精度几乎无损。已被 TensorRT-LLM 采用。

### 7.4 KV Cache 量化

- **KIVI**（ICML 2024）：首个免调参的 2-bit KV cache 量化。洞察：Key 的分布模式稳定（per-channel 量化），Value 的分布易变（per-token 量化）。KV 省 4 倍显存，长上下文场景吞吐大增。

### 7.5 系统协同（Quantization × Serving System）

- **QServe / QoQ**（MLSys 2025，Han Lab）：W4A8KV4——4-bit 权重、8-bit 激活、4-bit KV cache，配合专门设计的 GEMM 调度（int4 权重打包、避免 dequant 开销），是"算法 + 系统"一起设计的代表作。

### 7.6 训练侧：QAT 与 1-bit

- **QAT（Quantization-Aware Training）/ STE**：训练时模拟量化误差（直通估计器），模型学会"容忍"低精度。质量上限最高，但需要数据和训练算力。
- **QLoRA**（NeurIPS 2023）：NF4（4-bit NormalFloat）+ 双重量化 + paged optimizer，让 65B 模型能在单张 48GB GPU 上微调，且不损失 16-bit 微调效果。→ 量化 + 参数高效微调的经典组合。
- **BitNet b1.58**（2024）：参数只有 {-1, 0, +1} 的三值权重（1.58 bit），从头训练，同等规模/训练 Token 下匹配 FP16 Transformer 的困惑度与下游表现，推理能耗大幅下降。→ "从架构上拥抱量化"的终极形态。

### 7.7 前沿：Attention 量化与大规模实践

- **FlashAttention-3**（2024）：Hopper 上支持 FP8 的 attention（低精度 + 异步拷贝）。
- **SageAttention**（2024）：FP8 量化 attention 且保持精度，图像/视频生成加速。
- **DeepSeek-V3**（2024）：671B MoE 全程 FP8 训练/推理（DeepGEMM），证明 FP8 在超大模型上是可行的。

---

## 8. 质量评估三板斧

量化有没有搞砸，用三种方法检查（章节原表），每种都在找"与噪声不可区分"的差异：

| 方法 | 成本 | 灵敏度 | 说明 |
|---|---|---|---|
| **Perplexity**（WikiText-2 / C4） | 最低 | 最粗 | 最常用的快速筛查，量化后涨幅应可忽略 |
| **智能基准**（MMLU、SWE-bench 等） | 中 | 中 | 覆盖知识和真实编程任务，更能暴露应用层退化 |
| **自定义评测（Custom Evals）** | 最高 | 最准 | 用你真实业务的 prompt/任务集评测，**最终以它为准** |

要点：只跑 perplexity 不够；只跑几个 MMLU 题也不够。生产上线前，至少要在自定义评测上把量化模型和原模型做并排对比，确认差异落在噪声范围内。

---

## 9. 交互练习实测：Quantization Quality Estimator

我把练习页 [Quantization Quality Estimator](https://inferenceengineering.tech/exercises/quantization-estimator/) 实际跑了一遍（70B 模型、FP16 原始精度），实测结果如下：

| 量化组件 | 目标精度 | 量化后大小 | 内存节省 | 估算加速 | 估算质量风险 |
|---|---|---|---|---|---|
| 仅权重 | FP8 | 70 GB | 50% | ~1.4x | Medium |
| 权重 + KV | FP8 | 70 GB | 50% | ~1.6x | High |
| 权重 + 激活 | FP8 | 70 GB | 50% | ~1.5x | High |
| 全部 | FP8 | 70 GB | 50% | ~1.5x | High |
| 仅权重 | FP4 | 35 GB | 75% | ~2.2x | High |
| 权重 + KV | FP4 | 35 GB | 75% | ~2.5x | High |
| 权重 + 激活 | FP4 | 35 GB | 75% | ~2.5x | High |
| 全部 | FP4 | 35 GB | 75% | ~2.8x | High |

几个值得注意的点：

1. **内存节省只看目标精度**：FP16→FP8 省 50%，FP16→FP4 省 75%（模型 70B = 140 GB / 70 GB / 35 GB）。勾选哪些组件**不影响**内存数字——估算器只按权重算模型大小，KV/激活的显存与上下文长度相关，没被建模。
2. **组件勾选影响加速比和风险**：加量化 KV/激活，加速比略升（1.4x → 1.6x；FP4 从 2.2x → 2.8x），但风险迅速拉高。
3. **估算器比章节结论保守**：章节说 FP8 是甜点、值得把权重+激活量到 FP8；但估算器对 70B 模型把"FP8 全组件"标成了 High risk。两者不矛盾——估算器是启发式"提示你小心"，章节的立场是"小心但值得做，用评测验证"。

> 结论：这是一个给数量级直觉的小工具，不是精度预言机。真实部署必须跑第 8 节的评测流程。

---

## 10. 生产建议与决策清单

### 10.1 默认配方（从章节和业界实践综合）

1. **首选 FP8 / MXFP8（W8A8）**：权重 + 激活都量化，Hopper/Blackwell 上有原生 Tensor Core 支持，收益 1.5x 左右，质量风险最低。
2. **KV cache 单独谨慎处理**：先跑长上下文场景的评测再决定是否量化（KIVI 类方法可到 4-bit/2-bit）。
3. **Attention 留在原精度**：除非用 FlashAttention-3/SageAttention 这类专门方案。
4. **4-bit 权重（W4A16/W4A8）是激进选项**：用 GPTQ/AWQ/QuIP# 生成，配合 vLLM/TensorRT-LLM 的 kernel，收益 2x+，但必须过自定义评测。
5. **能用无损优化就先无损**：推测解码、prefix caching、KV 复用等全部无损，优先级应该在"激进量化"之前。

### 10.2 上线检查清单

- [ ] 用校准集（几百条代表性 prompt）跑 perplexity，与 FP16 基线对比
- [ ] 跑 MMLU/SWE-bench 等基准，量化前后差异 < 噪声
- [ ] 用真实业务 prompt 集做自定义评测（尤其代码、数学、长上下文）
- [ ] 检查 outlier 场景：长 prompt、多轮对话、流式输出
- [ ] 实测 TTFT / TPS / 吞吐 / 显存占用，确认收益达到预期
- [ ] 保留 A/B 开关，可随时回退到高精度版本

---

## 11. Han Song MIT 6.5940 课程地图

韩松（Song Han）是 MIT EECS 副教授，HAN Lab 负责人，也是 SmoothQuant、AWQ、QServe 等方法的作者。他的公开课 **6.5940 TinyML and Efficient AI Computing（EfficientML.ai）** 是学习量化的最佳课程入口。

### 11.1 课程主页

- 2026 Fall（当前）：https://hanlab.mit.edu/courses/2026-fall-65940
- 2024 Fall：https://hanlab.mit.edu/courses/2024-fall-65940
- 2023 Fall：https://hanlab.mit.edu/courses/2023-fall-65940
- 课程视频/讲义聚合：https://efficientml.ai

### 11.2 与量化直接相关的讲座

| 讲座 | 内容 | 视频入口 |
|---|---|---|
| Lecture 5: Quantization (Part I) | 线性量化的数学基础：bitwidth、均匀/非均匀量化、PTQ（RTN、GPTQ、LLM.int8、SmoothQuant） | [Class Central（Fall 2024）](https://www.classcentral.com/course/youtube-efficientml-ai-lecture-5-quantization-part-i-mit-6-5940-fall-2024-340161) |
| Lecture 6: Quantization (Part II) | QAT 与 STE、蒸馏、低比特/二值网络、量化与部署的配合 | [Class Central（Fall 2024）](https://www.classcentral.com/course/youtube-efficientml-ai-lecture-6-quantization-part-ii-mit-6-5940-fall-2024-zoom-recording-340157) |
| Lecture 13: LLM Quantization and Deployment（2026 版）/ LLM Deployment Techniques（2024 版） | 把量化放进真实 LLM 服务系统：KV cache、并行、serving | 课程主页 Schedule 里找 [Slides]/[Video] |

> 提示：B 站有该课程的搬运与笔记（如 [EfficientML.ai Lecture 6 笔记](https://www.bilibili.com/opus/918609044111884338)），中文学习可配合使用。课程中文介绍可看[澎湃新闻的报道](https://www.thepaper.cn/newsDetail_forward_24735283)。

### 11.3 课程里的配套 Lab

6.5940 有动手 Lab（实现剪枝/量化并部署 Llama 到笔记本/移动端）。建议至少做量化相关的 lab：亲手实现 uniform quantization → 对比 RTN vs GPTQ vs AWQ 的质量差异，比只看论文有效得多。

---

## 12. 论文阅读清单

按推荐阅读顺序排列（核心 5 篇在前，扩展在后）：

| # | 论文 | 会议/年份 | 一句话要点 |
|---|---|---|---|
| 1 | [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) | NeurIPS 2022 | outlier 列单独高精度，首个无损 INT8 大模型推理 |
| 2 | [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) | ICLR 2023 | 二阶 Hessian 误差补偿，一次性 3–4 bit 量化 175B |
| 3 | [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) | ICML 2023 | 激活→权重迁移量化难度，W8A8 无损 |
| 4 | [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) | MLSys 2024 | 按激活重要性保护 1% 权重，硬件友好的 4-bit |
| 5 | [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | NeurIPS 2023 | NF4 + 双重量化，量化模型上直接微调 |
| 6 | [SqueezeLLM: Dense-and-Sparse Quantization](https://arxiv.org/abs/2306.07629) | ICML 2024 | 敏感度非均匀量化 + outlier 分离 |
| 7 | [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396) | ICML 2024 | Hadamard 洗牌 + E8 格码本，2-bit SOTA |
| 8 | [KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) | ICML 2024 | Key 按通道、Value 按 Token 量化，KV 省 4 倍 |
| 9 | [QServe: W4A8KV4 Quantization and System Co-design](https://arxiv.org/abs/2405.04532) | MLSys 2025 | 算法与 GEMM 调度协同，4-8-4 组合落地 |
| 10 | [The Era of 1-bit LLMs: BitNet b1.58](https://arxiv.org/abs/2402.17764) | 2024 | 三值权重 {-1,0,1}，匹配 FP16 质量、大幅降耗 |
| 11 | [FP8 Formats for Deep Learning（NVIDIA/Arm/Intel）](https://arxiv.org/abs/2209.05433) | 2022 | E4M3 / E5M2 规格与使用建议 |
| 12 | [OCP Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) | OCP 2023 | MXFP8/MXFP4 的 block-32 + E8M0 标准 |
| 13 | [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08691) | 2024 | Hopper 上的 FP8 attention |
| 14 | [SageAttention: Accurate 8-Bit Attention](https://arxiv.org/abs/2410.02367) | 2024 | FP8 量化 attention 保持精度 |
| 15 | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 2024 | 671B MoE 全程 FP8 训练/推理实践 |
| 16 | [NVFP4: NVIDIA 官方介绍](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) | 2025 | Blackwell 4-bit：block 16 + E4M3 缩放 |

阅读策略：先把 1–5 读完，你就能回答"量化为什么难、怎么解决"；6–10 是进阶（极低比特 + KV cache + 系统协同）；11–16 是格式规范与生产案例，按需查阅。

---

## 13. 工具与生态

| 工具 | 说明 |
|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | 最主流的开源推理引擎，支持 AWQ/GPTQ/FP8/MXFP8/FP4、KV cache 量化（[量化文档](https://docs.vllm.ai/en/latest/features/quantization/)） |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA 高性能引擎，深度支持 FP8/INT4/FP4 与 SmoothQuant/AWQ |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF 生态，Q2_K–Q8_0 各档量化，本地/边缘部署首选 |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | NF4/INT8，HuggingFace 生态 + QLoRA 微调 |
| [SGLang](https://github.com/sgl-project/sglang) | 高性能推理框架，量化支持同样丰富 |
| [HAN Lab omniserve](https://github.com/mit-han-lab/omniserve) | QServe/LServe 官方代码，学 W4A8KV4 系统实现 |
| [HuggingFace 量化文档](https://huggingface.co/docs/transformers/quantization) | GPTQ/AWQ/FP8/bitsandbytes 的 API 与教程 |

---

## 14. 术语速查

- **PTQ（Post-Training Quantization）**：训练后量化，不需要重训，用少量校准数据即可。推理优化默认指这个。
- **QAT（Quantization-Aware Training）**：训练时模拟量化误差，质量上限更高、成本更高。
- **RTN**：Round-to-Nearest，最朴素的四舍五入量化。
- **W8A8 / W4A16 / W4A8KV4**：权重位数-激活位数（-KV位数）的缩写，如 W4A8KV4 = 4-bit 权重 + 8-bit 激活 + 4-bit KV cache。
- **Scale / Zero-point**：量化缩放系数与零点偏移，负责把实数区间映射到整数网格。
- **Block size / Group size**：一组元素共享一个 scale 的元素个数（16/32/128）。
- **Outlier（离群值）**：远大于其他值的少数维度，是低比特量化的主要敌人。
- **KV cache**：解码时缓存的历史 Key/Value 向量，长上下文下占显存大头。
- **TTFT / TPS**：Time To First Token（首 Token 延迟）/ Tokens Per Second（生成吞吐）。

---

## 15. 参考资料

1. Inference Engineering, Ch 5: Quantization & Speculative Decoding — https://inferenceengineering.tech/chapters/techniques/
2. Inference Engineering, Recommended Reading — https://inferenceengineering.tech/reading/
3. Inference Engineering, Quantization Quality Estimator — https://inferenceengineering.tech/exercises/quantization-estimator/
4. MIT 6.5940 EfficientML.ai（Han Song）— https://hanlab.mit.edu/courses/2026-fall-65940
5. HAN Lab 论文主页（SmoothQuant / AWQ / QServe / BitNet）— https://hanlab.mit.edu
6. 本笔记对应的论文（见第 12 节表格，均为 arXiv 原文链接）

