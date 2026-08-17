# LLM 量化精读笔记 · 00 总览与学习地图

> 系列定位：在 [LLM推理优化-Quantization量化学习笔记.md](./LLM推理优化-Quantization量化学习笔记.md)（概览版）的基础上，按 **MIT lecture note 水准**逐章精读 LLM 推理量化。每章独立成文件，包含：形式化定义、完整数学推导、伪代码、数值算例、直觉解释、习题与答案、延伸阅读。
> 撰写方式：一章一章写，写完一章再进入下一章。
> ✅ **全系列 11 章已完成（2026-08-17）**，按$01 \to 11$顺序阅读即可；每章"延伸阅读"末尾有上一篇/下一篇跳转。

---

## 1. 系列结构与来源映射

| 章节 | 文件 | 核心内容 | 对应来源 |
|---|---|---|---|
| 00 | 本文件 | 学习地图、符号约定、阅读顺序 | — |
| 01 | [01-数值编码与计算机表示基础](./LLM量化精读笔记-01-数值编码与计算机表示基础.md) | 信息论（bit/熵/率失真）、整数/定点编码、IEEE 754 浮点、舍入与截断、存储层次 | MIT L5 开场；IEEE 754；Goldberg；Horowitz |
| 02 | [02-量化问题形式化与均匀量化理论](./LLM量化精读笔记-02-量化问题形式化与均匀量化理论.md) | 量化的一般形式、均匀量化数学、误差与 SNR 理论 | MIT 6.5940 L5（前半）；Inference Engineering Ch5 |
| 03 | [03-数值格式与硬件](./LLM量化精读笔记-03-数值格式与硬件.md) | FP16/BF16/FP8(E4M3/E5M2)/FP4/MXFP8/NVFP4 的位布局、动态范围、Tensor Core 与带宽 | Inference Engineering Ch5；FP8 白皮书；OCP MX 规范 |
| 04 | [04-量化粒度校准与离群值](./LLM量化精读笔记-04-量化粒度校准与离群值.md) | per-tensor/channel/block、校准数据、outlier 问题 | MIT L5；LLM.int8()；Inference Engineering Ch5 |
| 05 | [05-权重量化 I](./LLM量化精读笔记-05-权重量化I-RTN与GPTQ.md) |$RTN \to GPTQ$（二阶误差补偿，含逐层推导） | GPTQ 论文；MIT L5 |
| 06 | [06-权重量化 II](./LLM量化精读笔记-06-权重量化II-AWQ-SqueezeLLM-QuIP.md) | AWQ（激活感知缩放）、SqueezeLLM（稠密/稀疏）、QuIP#（Hadamard + 格码本） | AWQ / SqueezeLLM / QuIP# 论文 |
| 07 | [07-激活量化](./LLM量化精读笔记-07-激活量化-LLM-int8与SmoothQuant.md) | LLM.int8()（outlier 分解）、SmoothQuant（迁移公式推导）、W8A8 | LLM.int8() / SmoothQuant 论文 |
| 08 | [08-KV Cache 量化](./LLM量化精读笔记-08-KV-Cache量化与KIVI.md) | 为什么 KV 是瓶颈、KIVI（per-channel key / per-token value）、误差累积 | KIVI 论文；Inference Engineering Ch5 |
| 09 | [09-QAT 与训练内量化](./LLM量化精读笔记-09-QAT与训练内量化-STE-QLoRA-BitNet.md) | STE、QLoRA(NF4)、BitNet b1.58（三值化） | MIT L6；QLoRA / BitNet 论文 |
| 10 | [10-质量评估方法论](./LLM量化精读笔记-10-质量评估方法论.md) | perplexity、MMLU/SWE-bench、自定义评测、统计显著性、校准集设计 | Inference Engineering Ch5；SWE-bench |
| 11 | [11-系统协同与部署](./LLM量化精读笔记-11-系统协同与部署.md) | QServe(W4A8KV4)、FP8 Attention、vLLM/TensorRT-LLM/llama.cpp 工程 | QServe / FlashAttention-3 / SageAttention |

## 2. 两条学习主线

### 主线 A：按"问题"走（推荐给第一次系统学习）

1. **打地基**（01）：信息论、整数/浮点编码、舍入与截断——回答"一个数在计算机里到底怎么存、误差从哪来"。
2. **为什么能省**（02–03）：量化 = 有损压缩；位宽每减 1，噪声音量减半、SNR 涨$\sim 6 dB$；浮点格式用指数位换动态范围。
3. **难在哪**（04）：outlier、分布利用不充分、粒度选择——所有方法都在绕这三个坑。
4. **怎么解决**（05–08）：权重（GPTQ/AWQ/QuIP#）$\to$激活（LLM.int8/SmoothQuant）$\to KV cache$（KIVI），从"最不敏感"到"最敏感"逐层攻克。
5. **训练侧怎么做**（09）：QAT、QLoRA、BitNet。
6. **怎么验收**（10）＋**怎么部署**（11）。

### 主线 B：按"论文时间线"走（适合已经读过概览）

$LLM.int8 (2022) \to GPTQ (2022) \to SmoothQuant (2023) \to AWQ (2023) \to QLoRA (2023) \to KIVI (2024) \to QuIP$#$(2024) \to QServe (2025)$。这条线能看出：**先解决权重，再解决激活，再解决 KV cache，最后做算法-系统协同设计**。

## 3. 配套资源

| 资源 | 用途 |
|---|---|
| [MIT 6.5940 Fall 2024 Lecture 5（Class Central）](https://www.classcentral.com/course/youtube-efficientml-ai-lecture-5-quantization-part-i-mit-6-5940-fall-2024-340161) | 量化 Part I：线性量化、bitwidth、PTQ（RTN/GPTQ/LLM.int8/SmoothQuant） |
| [MIT 6.5940 Fall 2024 Lecture 6（Class Central）](https://www.classcentral.com/course/youtube-efficientml-ai-lecture-6-quantization-part-ii-mit-6-5940-fall-2024-zoom-recording-340157) | 量化 Part II：QAT、STE、蒸馏、低比特/二值 |
| [6.5940 课程主页（2026 Fall）](https://hanlab.mit.edu/courses/2026-fall-65940) | 讲义/视频入口（Lecture 5/6/13） |
| [6.5940 Lab 2：Quantization（GitHub）](https://github.com/CalebDu/MIT6.5940-EfficientML/blob/master/Lab2-quantization/Lab2.ipynb) | 动手实现 linear quantize / k-means 量化，配套每章习题 |
| [Inference Engineering Ch5](https://inferenceengineering.tech/chapters/techniques/) | 教材正文（本系列的"骨架"） |
| 各章引用的 arXiv 论文 | 见每章末尾"延伸阅读" |

## 4. 全局符号约定（全系列通用）

| 符号 | 含义 |
|---|---|
| b | 位宽（bitwidth） |
| r | 原始实数（权重/激活/KV 值） |
| $\hat{r}$ | 反量化后的近似实数 |
| q | 量化后的整数值 |
| qmin, qmax | 量化整数值域（如$INT8: -128 \sim 127$） |
| s | 缩放因子 scale（实数） |
| z | 零点 zero-point（整数） |
|$\Delta$| 步长 step size（相邻量化层间距） |
| [rmin, rmax] | 量化覆盖的实数范围 |
|$\varepsilon$| 量化误差（$\varepsilon = r - \hat{r}$） |
|$\sigma ^{2}$| 方差（信号$\sigma ^{2}_s$，噪声$\sigma ^{2}_e$） |
| SNR / SQNR | 信噪比 / 量化信噪比（dB） |
| W8A8 / W4A16 / W4A8KV4 | 权重位数-激活位数（-KV 位数）记法 |
| PTQ / QAT | 训练后量化 / 量化感知训练 |
| RTN | Round-to-Nearest（就近舍入） |
| clamp(x, lo, hi) | min(max(x, lo), hi) |
| round(·) | 四舍五入到最近整数 |

## 5. 使用建议

- 每章先读"本章目标"，再读主体，最后做"习题"；习题答案在本章末尾，先自己算再看。
- 涉及代码的习题建议直接在 [Lab 2 notebook](https://github.com/CalebDu/MIT6.5940-EfficientML/blob/master/Lab2-quantization/Lab2.ipynb) 的环境里跑，公式与本系列一致。
- 数学符号统一用上表；遇到跨章引用会标注章节号。
