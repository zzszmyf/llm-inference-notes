# LLM 注意力与计算内核精读笔记 · 00 总览与学习地图

> 系列定位：继 [量化精读笔记](./LLM量化精读笔记-00-总览与学习地图.md)（每一步的数值/带宽）与 [推测解码精读笔记](./LLM推测解码精读笔记-00-总览与学习地图.md)（需要的步数）之后的第三部 **MIT lecture note 级别**推理优化精读。本系列进入"单步前向内部"：**注意力机制与底层计算内核**。
> 格式与之前一致：形式化定义 → 数学推导 → 伪代码/算法 → 数值算例 → 直觉解释 → 习题（含答案）→ 延伸阅读；公式使用 Markdown + LaTeX（`$...$` / `$$...$$`）。

---

## 1. 系列结构与来源映射

| 章节 | 文件 | 核心内容 | 对应来源 |
|---|---|---|---|
| 00 | 本文件 | 学习地图、符号约定、与另两个系列的关系 | — |
| 01 | 01-注意力机制基础与复杂度分析 | softmax attention 定义、O(L²) 复杂度、因果掩码、KV cache 角色、prefill/decode 形态 | Vaswani et al. 2017；Inference Engineering Ch5 |
| 02 | 02-FlashAttention：IO 感知的精确注意力 | online softmax、tiling、重计算、FA2/FA3、为什么"算更多反而更快" | FlashAttention（arXiv:2205.14135；2307.08691；2407.08608） |
| 03 | 03-注意力头变体：MQA/GQA/MLA | KV 头共享、低秩压缩、DeepSeek MLA、显存与质量权衡 | GQA（arXiv:2305.13245）；DeepSeek-V2（arXiv:2405.04434） |
| 04 | 04-稀疏、滑动窗口与线性注意力 | StreamingLLM / Attention Sink、滑动窗口、H2O、线性注意力、SSM/Mamba | StreamingLLM（arXiv:2309.17453）；Mamba（arXiv:2312.00752） |
| 05 | 05-PagedAttention 与 KV 显存管理 | 分页 KV、vLLM 块管理、与连续批处理/前缀缓存组合 | PagedAttention / vLLM（arXiv:2309.06180） |
| 06 | 06-内核优化与算子融合 | 访存-计算模型、算子融合、Tensor Core、FA 的 kernel 细节、FP8 注意力、编译优化 | FlashAttention 系列；工程实践 |
| 07 | 07-系统集成与生产验收 | 与量化/推测解码/调度组合、注意力精度验收、决策树 | vLLM/SGLang/TensorRT-LLM 实践 |

## 2. 三部系列的关系：一张总表

```
                    LLM 推理优化
       ┌──────────────┼──────────────────┐
   每步成本        需要的步数           单步内部
       │               │                  │
   量化（已写完）   推测解码（已写完）   注意力与内核（本系列）
   数值/带宽         步数/草稿           计算路径/显存
```

互补性：

```
量化：让"一次搬移/一次运算"更便宜（带宽减半、FLOPS 翻倍）
推测：让"需要的步数"更少（一次验证多步）
注意力内核：让"单步前向里最贵的部分"更高效（O(L²) 注意力与 KV 显存）
三者乘法叠加：每步成本 × 步数 × 每步内部效率 = 端到端收益
```

决策框架（衔接之前系列）：

```
1. 先看注意力：长上下文时它是最贵的（01 章 L/(2d) 判据）
2. 再看步数：推测解码无损提步数（已写完）
3. 最后看数值：量化兜底（已写完）
```

## 3. 符号约定（全系列通用，新增）

| 符号 | 含义 |
|---|---|
| $L$ | 序列长度 |
| $d$ / $d_{\text{model}}$ | 隐藏维度 |
| $h$ | 注意力头数 |
| $d_{\text{head}}$ | 每头维度（$d_{\text{head}} = d/h$） |
| $n_{\text{kv}}$ | KV 头数（MQA 为 1，GQA 取中间值） |
| $d_{\text{kv}}$ | KV 总维度（$n_{\text{kv}} \times d_{\text{head}}$） |
| $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ | 查询/键/值矩阵（$L \times d_{\text{head}}$） |
| $M$ | 片上 SRAM 容量（FlashAttention 的预算） |
| $B_c, B_r$ | FlashAttention 的块大小（列/行） |
| $\text{KV bytes}$ | KV cache 每 token 字节数 |

沿用：$T_p$（目标每 token 时间）、$\alpha$（接受率）、$c$（速度比）、W4A8KV4 等记法。

## 4. 阅读顺序

```
01 注意力基础与复杂度（为什么 O(L²)、KV cache 从哪来）
  → 02 FlashAttention（prefill 侧：怎么把 L² 算得快且省显存）
  → 03 MQA/GQA/MLA（decode 侧：怎么把 KV 显存砍下来）
  → 04 稀疏/线性注意力（长上下文：怎么跳过 L²）
  → 05 PagedAttention（系统侧：KV 显存怎么分页管理）
  → 06 内核优化（硬件侧：算子融合与 Tensor Core）
  → 07 系统集成与验收（怎么组合、怎么验收）
```

## 5. 配套资源

| 资源 | 用途 |
|---|---|
| [Attention Is All You Need（arXiv:1706.03762）](https://arxiv.org/abs/1706.03762) | 缩放点积注意力的原始定义 |
| [FlashAttention（arXiv:2205.14135）](https://arxiv.org/abs/2205.14135) / [FA2（arXiv:2307.08691）](https://arxiv.org/abs/2307.08691) / [FA3（arXiv:2407.08608）](https://arxiv.org/abs/2407.08608) | IO 感知精确注意力 |
| [GQA（arXiv:2305.13245）](https://arxiv.org/abs/2305.13245) / [DeepSeek-V2 MLA（arXiv:2405.04434）](https://arxiv.org/abs/2405.04434) | KV 头共享与压缩 |
| [StreamingLLM（arXiv:2309.17453）](https://arxiv.org/abs/2309.17453) / [Mamba（arXiv:2312.00752）](https://arxiv.org/abs/2312.00752) | 长上下文与次二次方法 |
| [PagedAttention / vLLM（arXiv:2309.06180）](https://arxiv.org/abs/2309.06180) | KV 分页与系统集成 |
| [Inference Engineering Ch5](https://inferenceengineering.tech/chapters/techniques/) | 教材正文（Attention 与系统部分） |

