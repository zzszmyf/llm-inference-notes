# LLM 推理优化精读笔记

一套 **MIT lecture note 级别**的 LLM 推理优化中文精读笔记，两大主线：

- **量化（有损换速度）**：从信息论与数值编码的地基，到均匀量化理论、数值格式、粒度与离群值、主流 PTQ/QAT 方法（GPTQ / AWQ / SmoothQuant / KIVI / QLoRA / BitNet 等）、质量评估与生产部署。
- **推测解码（无损换步数）**：从接受率数学与拒绝采样，到原始推测解码、Medusa 多头解码、EAGLE 特征空间草稿、n-gram/检索式无模型路线，以及系统集成与生产验收。

全部公式使用 **Markdown + LaTeX**（行内 `$...$`、独立 `$$...$$`），本站通过 MathJax 渲染。

## 量化精读笔记（00-11）

| 章节 | 内容 |
|---|---|
| [速览笔记](./LLM推理优化-Quantization量化学习笔记.md) | 概览版，适合快速复习 |
| [00 总览与学习地图](./LLM量化精读笔记-00-总览与学习地图.md) | 章节结构、符号约定、阅读路线、配套资源 |
| [01 数值编码与计算机表示基础](./LLM量化精读笔记-01-数值编码与计算机表示基础.md) | 信息论、整数/定点编码、IEEE 754、舍入与截断、存储层次 |
| [02 量化问题形式化与均匀量化理论](./LLM量化精读笔记-02-量化问题形式化与均匀量化理论.md) | 量化的一般形式、均匀量化数学、误差与 SNR 理论（6 dB/bit 推导） |
| [03 数值格式与硬件](./LLM量化精读笔记-03-数值格式与硬件.md) | FP16/BF16/FP8/FP4/MXFP8/NVFP4、Tensor Core、带宽模型 |
| [04 量化粒度、校准与离群值](./LLM量化精读笔记-04-量化粒度校准与离群值.md) | per-tensor/channel/group、有效位宽、校准设计、outlier 问题 |
| [05 权重量化 I（RTN 与 GPTQ）](./LLM量化精读笔记-05-权重量化I-RTN与GPTQ.md) | OBS/OBQ 二阶误差补偿推导、GPTQ 工程化 |
| [06 权重量化 II（AWQ、SqueezeLLM、QuIP#）](./LLM量化精读笔记-06-权重量化II-AWQ-SqueezeLLM-QuIP.md) | 激活感知缩放、敏感度非均匀量化、Hadamard 非相干 + 格码本 |
| [07 激活量化（LLM.int8 与 SmoothQuant）](./LLM量化精读笔记-07-激活量化-LLM-int8与SmoothQuant.md) | 混合精度分解、迁移公式、W8A8 与 scale 折叠 |
| [08 KV Cache 量化（KIVI）](./LLM量化精读笔记-08-KV-Cache量化与KIVI.md) | KV 显存账本、误差累积、K 按通道 / V 按 token |
| [09 QAT 与训练内量化](./LLM量化精读笔记-09-QAT与训练内量化-STE-QLoRA-BitNet.md) | STE、QLoRA（NF4）、BitNet b1.58 |
| [10 质量评估方法论](./LLM量化精读笔记-10-质量评估方法论.md) | perplexity、智能基准、自定义评测、统计显著性与验收关卡 |
| [11 系统协同与部署](./LLM量化精读笔记-11-系统协同与部署.md) | QServe W4A8KV4、FP8 Attention、引擎选型与部署决策树 |

## 推测解码精读笔记（00-06）

| 章节 | 内容 |
|---|---|
| [00 总览与学习地图](./LLM推测解码精读笔记-00-总览与学习地图.md) | 章节结构、符号约定、与量化系列的关系 |
| [01 问题形式化与接受率数学](./LLM推测解码精读笔记-01-问题形式化与接受率数学.md) | 自回归为什么慢、接受率 α、E[N] 推导、墙钟收益模型 |
| [02 原始推测解码（草稿模型与拒绝采样）](./LLM推测解码精读笔记-02-原始推测解码-草稿模型与拒绝采样.md) | 完整算法、无损性定理、最优 K、草稿规模权衡 |
| [03 Medusa（多头解码）](./LLM推测解码精读笔记-03-Medusa-多头解码.md) | 多头并行预测、树注意力、典型验收、Medusa-1/2 |
| [04 EAGLE（特征空间草稿）](./LLM推测解码精读笔记-04-EAGLE-特征空间草稿.md) | 特征级自回归、shifted-token、EAGLE-2 动态树 |
| [05 n-gram/检索式与无模型路线](./LLM推测解码精读笔记-05-n-gram检索式与无模型路线.md) | Prompt Lookup、Lookahead Decoding、REST |
| [06 系统集成与生产验收](./LLM推测解码精读笔记-06-系统集成与生产验收.md) | 量化 × 推测组合模型、TTFT/TPS、验收协议、决策树 |

## 阅读顺序

```
01 数值编码（地基）→ 02 均匀量化理论（数学）→ 03 数值格式与硬件（格式）
→ 04 粒度、校准与离群值（难点）→ 05-06 权重量化 → 07 激活量化
→ 08 KV Cache → 09 训练侧 → 10 质量评估 → 11 系统与部署
```

两套系列每章结构统一：形式化定义 → 数学推导 → 伪代码/算法 → 数值算例 → 直觉解释 → 习题（含答案）→ 延伸阅读。

## 素材来源

- [Inference Engineering, Ch5](https://inferenceengineering.tech/chapters/techniques/)（Baseten 出品）
- [MIT 6.5940 EfficientML.ai](https://hanlab.mit.edu/courses/2026-fall-65940)（Song Han / HAN Lab）
- 论文：LLM.int8() / GPTQ / SmoothQuant / AWQ / SqueezeLLM / QuIP# / KIVI / QServe / QLoRA / BitNet b1.58 / FP8 白皮书 / OCP MX 规范（各章"延伸阅读"附 arXiv 链接）
- 推测解码论文：Leviathan et al. 2023 / Chen et al. 2023 / Medusa / EAGLE / EAGLE-2 / Lookahead Decoding / REST（各章"延伸阅读"附 arXiv 链接）
