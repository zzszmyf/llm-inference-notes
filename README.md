# LLM 推理优化精读笔记（LLM Inference Optimization Study Notes）

> 一套 MIT lecture note 级别的 **LLM 推理优化**中文精读笔记，两条主线：
>
> - **量化（有损换速度）**：信息论与数值编码 → 均匀量化理论 → 数值格式与硬件 → 粒度/校准/离群值 → PTQ/QAT（GPTQ / AWQ / SmoothQuant / KIVI / QLoRA / BitNet 等）→ 质量评估 → 生产部署，共 11 章 + 总览。
> - **推测解码（无损换步数）**：接受率数学 → 原始推测解码 → Medusa 多头解码 → EAGLE 特征空间草稿 → n-gram/检索式无模型路线 → 系统集成与生产验收，共 6 章 + 总览。
> - **注意力与计算内核**：注意力机制与复杂度 → FlashAttention → MQA/GQA/MLA → 稀疏/线性注意力 → PagedAttention → 内核优化 → 系统集成，共 7 章 + 总览。
>
> 全部公式使用 **Markdown + LaTeX**（行内 `$...$`、独立 `$$...$$`），可直接在支持 MathJax/KaTeX 的 Markdown 阅读器中渲染。

## 在线浏览

📖 静态网站（GitHub Pages，带侧边栏导航、中文搜索、MathJax 公式渲染）：

<https://zzszmyf.github.io/llm-inference-notes/>

源码目录 `docs/` 即网站源文件；本地重新构建网站：

```bash
./build_site.sh        # 需要项目内 .venv（mkdocs-material）
```

---

## 章节列表

| 章节 | 内容 | 状态 |
|---|---|---|
| [00 总览与学习地图](./docs/LLM量化精读笔记-00-总览与学习地图.md) | 章节结构、符号约定、阅读路线、配套资源 | ✅ |
| [01 数值编码与计算机表示基础](./docs/LLM量化精读笔记-01-数值编码与计算机表示基础.md) | 信息论（bit/熵/率失真）、整数/定点编码、IEEE 754、舍入与截断、存储层次 | ✅ |
| [02 量化问题形式化与均匀量化理论](./docs/LLM量化精读笔记-02-量化问题形式化与均匀量化理论.md) | 量化的一般形式、均匀量化数学、误差与 SNR 理论（6 dB/bit 推导） | ✅ |
| [03 数值格式与硬件](./docs/LLM量化精读笔记-03-数值格式与硬件.md) | FP16/BF16/FP8(E4M3/E5M2)/FP4/MXFP8/NVFP4 位布局、Tensor Core、带宽模型 | ✅ |
| [04 量化粒度、校准与离群值](./docs/LLM量化精读笔记-04-量化粒度校准与离群值.md) | per-tensor/channel/group、有效位宽、校准设计、outlier 问题 | ✅ |
| [05 权重量化 I：RTN 与 GPTQ](./docs/LLM量化精读笔记-05-权重量化I-RTN与GPTQ.md) | OBS/OBQ 二阶误差补偿的完整推导、GPTQ 工程化 | ✅ |
| [06 权重量化 II：AWQ、SqueezeLLM、QuIP#](./docs/LLM量化精读笔记-06-权重量化II-AWQ-SqueezeLLM-QuIP.md) | 激活感知缩放、敏感度非均匀量化、Hadamard 非相干 + 格码本 | ✅ |
| [07 激活量化：LLM.int8 与 SmoothQuant](./docs/LLM量化精读笔记-07-激活量化-LLM-int8与SmoothQuant.md) | 混合精度分解、迁移公式、W8A8 与 scale 折叠 | ✅ |
| [08 KV Cache 量化与 KIVI](./docs/LLM量化精读笔记-08-KV-Cache量化与KIVI.md) | KV 显存账本、误差累积、K 按通道 / V 按 token | ✅ |
| [09 QAT 与训练内量化](./docs/LLM量化精读笔记-09-QAT与训练内量化-STE-QLoRA-BitNet.md) | STE、QLoRA（NF4）、BitNet b1.58 | ✅ |
| [10 质量评估方法论](./docs/LLM量化精读笔记-10-质量评估方法论.md) | perplexity、智能基准、自定义评测、统计显著性与验收关卡 | ✅ |
| [11 系统协同与部署](./docs/LLM量化精读笔记-11-系统协同与部署.md) | QServe W4A8KV4、FP8 Attention、引擎选型与部署决策树 | ✅ |

另附：[量化学习速览笔记](./docs/LLM推理优化-Quantization量化学习笔记.md)（概览版，适合快速复习）。

## 推测解码精读笔记（00-06）

| 章节 | 内容 | 状态 |
|---|---|---|
| [00 总览与学习地图](./docs/LLM推测解码精读笔记-00-总览与学习地图.md) | 章节结构、符号约定、与量化系列的关系 | ✅ |
| [01 问题形式化与接受率数学](./docs/LLM推测解码精读笔记-01-问题形式化与接受率数学.md) | 自回归为什么慢、接受率 α、E[N] 推导、墙钟收益模型 | ✅ |
| [02 原始推测解码（草稿模型与拒绝采样）](./docs/LLM推测解码精读笔记-02-原始推测解码-草稿模型与拒绝采样.md) | 完整算法、无损性定理、最优 K、草稿规模权衡 | ✅ |
| [03 Medusa（多头解码）](./docs/LLM推测解码精读笔记-03-Medusa-多头解码.md) | 多头并行预测、树注意力、典型验收、Medusa-1/2 | ✅ |
| [04 EAGLE（特征空间草稿）](./docs/LLM推测解码精读笔记-04-EAGLE-特征空间草稿.md) | 特征级自回归、shifted-token、EAGLE-2 动态树 | ✅ |
| [05 n-gram/检索式与无模型路线](./docs/LLM推测解码精读笔记-05-n-gram检索式与无模型路线.md) | Prompt Lookup、Lookahead Decoding、REST | ✅ |
| [06 系统集成与生产验收](./docs/LLM推测解码精读笔记-06-系统集成与生产验收.md) | 量化 × 推测组合模型、TTFT/TPS、验收协议、决策树 | ✅ |

## 注意力与计算内核精读笔记（00-07）

| 章节 | 内容 | 状态 |
|---|---|---|
| [00 总览与学习地图](./docs/LLM注意力内核精读笔记-00-总览与学习地图.md) | 章节结构、符号约定、与量化/推测解码系列的关系 | ✅ |
| [01 注意力机制基础与复杂度分析](./docs/LLM注意力内核精读笔记-01-注意力机制基础与复杂度分析.md) | softmax attention 定义、O(L²) 复杂度、KV cache 角色、prefill/decode 形态 | ✅ |
| [02 FlashAttention（IO 感知的精确注意力）](./docs/LLM注意力内核精读笔记-02-FlashAttention-IO感知的精确注意力.md) | IO 复杂度、tiling、online softmax、重计算、FA2/FA3 | ✅ |
| [03 注意力头变体（MQA/GQA/MLA）](./docs/LLM注意力内核精读笔记-03-注意力头变体-MQA-GQA-MLA.md) | KV 头共享、低秩压缩、DeepSeek MLA | ✅ |
| [04 稀疏、滑动窗口与线性注意力](./docs/LLM注意力内核精读笔记-04-稀疏滑动窗口与线性注意力.md) | StreamingLLM、滑动窗口、H2O、Mamba | ✅ |
| [05 PagedAttention 与 KV 显存管理](./docs/LLM注意力内核精读笔记-05-PagedAttention与KV显存管理.md) | 分页 KV、vLLM 块管理、与批处理组合 | ✅ |
| [06 内核优化与算子融合](./docs/LLM注意力内核精读笔记-06-内核优化与算子融合.md) | 访存-计算模型、Tensor Core、FP8 注意力、编译优化 | ✅ |
| [07 系统集成与生产验收](./docs/LLM注意力内核精读笔记-07-系统集成与生产验收.md) | 与量化/推测解码组合、注意力精度验收 | ✅ |

## 阅读顺序

```
01 数值编码（地基）
  → 02 均匀量化理论（数学）
  → 03 数值格式与硬件（格式）
  → 04 粒度、校准与离群值（难点）
  → 05-06 权重量化（GPTQ/AWQ/SqueezeLLM/QuIP#）
  → 07 激活量化（LLM.int8/SmoothQuant）
  → 08 KV Cache（KIVI）
  → 09 训练侧（QAT/QLoRA/BitNet）
  → 10 质量评估（怎么验收）
  → 11 系统与部署（怎么上线）
```

每章结构统一：形式化定义 → 数学推导 → 伪代码/算法 → 数值算例 → 直觉解释 → 习题（含答案）→ 延伸阅读。

## 公式渲染

**GitHub 原生支持**：Markdown 与 README 中的 LaTeX 数学公式可直接渲染（行内 `$...$`、独立 `$$...$$`，引擎为 MathJax）。本仓库已按 GitHub 的语法规则处理（`$` 与内容之间不留空格），打开任意章节即可看到公式。

本地阅读器（支持 MathJax/KaTeX 的 Markdown + LaTeX）：

- VS Code + [Markdown Preview Enhanced](https://marketplace.visualstudio.com/items?itemName=shd101wyy.markdown-preview-enhanced)（或 VS Code 内置 Markdown 预览）
- [Typora](https://typora.io/)
- [Obsidian](https://obsidian.md/)

## 素材来源

- [Inference Engineering, Ch5](https://inferenceengineering.tech/chapters/techniques/)（Baseten 出品）
- [MIT 6.5940 EfficientML.ai](https://hanlab.mit.edu/courses/2026-fall-65940)（Song Han / HAN Lab）
- 论文：LLM.int8() / GPTQ / SmoothQuant / AWQ / SqueezeLLM / QuIP# / KIVI / QServe / QLoRA / BitNet b1.58 / FP8 白皮书 / OCP MX 规范（各章"延伸阅读"附 arXiv 链接）

## License

[MIT](./LICENSE)
