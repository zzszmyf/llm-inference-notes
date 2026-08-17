# LLM 推测解码精读笔记 · 00 总览与学习地图

> 系列定位：继 [LLM 量化精读笔记](./LLM量化精读笔记-00-总览与学习地图.md) 之后的第二部 **MIT lecture note 级别** 推理优化精读。量化是"有损换速度"，推测解码是**无损提速度**——两者是 LLM 推理优化里互补的两大主线，一起读完才能建立完整的优化决策框架。
> 格式与量化系列一致：形式化定义 → 数学推导 → 伪代码 → 数值算例 → 直觉解释 → 习题（含答案）→ 延伸阅读；公式使用 Markdown + LaTeX（`$...$` / `$$...$$`）。

---

## 1. 系列结构与来源映射

| 章节 | 文件 | 核心内容 | 对应来源 |
|---|---|---|---|
| 00 | 本文件 | 学习地图、符号约定、与量化系列的关系 | — |
| 01 | [01-问题形式化与接受率数学](./LLM推测解码精读笔记-01-问题形式化与接受率数学.md) | 自回归为什么慢、草稿+验证范式、接受率 α、期望 token 数 E[N] 推导、墙钟收益模型 | Leviathan et al. 2023；Inference Engineering Ch5 |
| 02 | [02-原始推测解码：草稿模型与拒绝采样](./LLM推测解码精读笔记-02-原始推测解码-草稿模型与拒绝采样.md) | 完整算法、无损性定理、数值算例、草稿速度比与最优 K | Leviathan et al. 2023；Chen et al. 2023 |
| 03 | [03-Medusa：多头解码](./LLM推测解码精读笔记-03-Medusa-多头解码.md) | 多头并行预测、树注意力、Medusa-1/2、草稿成本分析 | Medusa（arXiv:2401.10774） |
| 04 | [04-EAGLE：特征空间草稿](./LLM推测解码精读笔记-04-EAGLE-特征空间草稿.md) | 特征空间自回归、为什么接受率更高、EAGLE-2 | EAGLE（arXiv:2401.15077）；EAGLE-2（arXiv:2406.16858） |
| 05 | [05-n-gram/检索式与无模型路线](./LLM推测解码精读笔记-05-n-gram检索式与无模型路线.md) | Lookahead Decoding、REST、Prompt Lookup | Lookahead（arXiv:2307.09991）；REST（arXiv:2311.08252） |
| 06 | [06-系统集成与生产验收](./LLM推测解码精读笔记-06-系统集成与生产验收.md) | 与量化/KV 缓存/批处理组合、吞吐与延迟权衡、验收协议 | vLLM/TensorRT-LLM 实践；量化系列 10 章方法论 |

## 2. 与量化系列的关系：一张总表

```
                    LLM 推理优化
                    /            \
              有损路线            无损路线
                 │                  │
       量化（已写完 00-11）      推测解码（本系列）
       权重/激活/KV 降精度       用草稿 + 验证换并行度
       省显存、省带宽、省 FLOPS  不改变输出分布，只省墙钟时间
```

互补性：

```
量化：让"每一步更快"（带宽减半、FLOPS 翻倍）
推测：让"需要的步数更少"（一步验证 K 个候选 token）
两者乘法叠加：量化后的每步成本 × 推测后的步数减少 = 端到端收益
```

决策框架（量化系列 10 章的延伸）：

```
1. 无损优先：推测解码、前缀缓存、KV 复用——先上
2. 有损兜底：量化（W8A8 → W4A8KV4）——质量评测通过再上
3. 组合验收：推测 × 量化的联合收益要单独测（06 章）
```

## 3. 符号约定（全系列通用，沿用量化系列并新增）

| 符号 | 含义 |
|---|---|
| p | 目标模型（大模型）的输出分布 |
| q | 草稿模型（小模型）的输出分布 |
| α | 每 token 接受概率 $\alpha = \sum_x \min(p(x), q(x))$ |
| β | β = 1 − α（拒绝概率，也等于 TV 距离） |
| K | 每轮草稿 token 数 |
| N | 每轮实际产出 token 数（随机变量） |
| T_p / T_q | 目标/草稿每 token 解码时间 |
| c | 速度比 c = T_p / T_q |
| TV(p, q) | 全变差距离 $\mathrm{TV}(p,q) = \frac{1}{2}\sum_x |p(x) - q(x)|$ |
| W4A8KV4 等 | 沿用量化系列记法 |

## 4. 阅读顺序

```
01 形式化与接受率数学（本章公式是所有后续推导的地基）
  → 02 原始推测解码（算法 + 无损性定理）
  → 03 Medusa（草稿从哪来：模型自带多头）
  → 04 EAGLE（草稿从哪来：特征空间）
  → 05 无模型路线（n-gram / 检索）
  → 06 系统集成（怎么组合、怎么验收）
```

## 5. 配套资源

| 资源 | 用途 |
|---|---|
| [Fast Inference from Transformers via Speculative Decoding（Leviathan et al., arXiv:2211.17192）](https://arxiv.org/abs/2211.17192) | 原始论文：接受率公式、无损性、最优草稿规模 |
| [Accelerating LLM Inference with Staged Speculative Decoding（Chen et al., arXiv:2302.01318）](https://arxiv.org/abs/2302.01318) | 同期独立工作 |
| [Medusa（arXiv:2401.10774）](https://arxiv.org/abs/2401.10774) | 多头解码 |
| [EAGLE（arXiv:2401.15077）](https://arxiv.org/abs/2401.15077) | 特征空间草稿 |
| [EAGLE-2（arXiv:2406.16858）](https://arxiv.org/abs/2406.16858) | 动态草稿树 |
| [Lookahead Decoding（arXiv:2307.09991）](https://arxiv.org/abs/2307.09991) / [REST（arXiv:2311.08252）](https://arxiv.org/abs/2311.08252) | 无模型路线 |
| [Inference Engineering Ch5](https://inferenceengineering.tech/chapters/techniques/) | 教材正文（Speculative Decoding 一节） |
