# LLM 量化精读笔记 · 11 系统协同与部署：QServe、FP8 Attention 与工程工具

> 对应：QServe / QoQ（arXiv:2405.04532，MLSys 2025）；FlashAttention-3（arXiv:2407.08691）；SageAttention（arXiv:2410.02367）；vLLM / TensorRT-LLM / llama.cpp。
> 学完本章你应该能：① 解释"量化算法好 ≠ 端到端快"的系统瓶颈（dequant、内存布局、kernel）；② 讲清 QServe 的 W4A8KV4 协同设计（为什么 W4A8 而不是 W4A4、KV4 为什么重要）；③ 说出主流引擎的量化支持地图；④ 给出"无损优先"的组合优化栈与部署决策树。

---

## 目录（本章）

1. 本章目标
2. 从算法到系统：三个隐形瓶颈
3. QServe：W4A8KV4 的算法-系统协同
4. 注意力量化（前沿）
5. 推理引擎的量化支持地图
6. 组合优化栈：无损优先
7. 部署决策树（含验收闭环）
8. 本章小结
9. 习题与解答
10. 延伸阅读

---

## 1. 本章目标

前面各章证明了"某种量化**算法**质量好"。本章回答最后一个问题：

> 同一套量化方案，为什么在不同引擎里速度差几倍？怎么把量化装进生产系统，和其他优化组合？

核心转变：从"量化权重"到"**为量化重新设计系统**"。

---

## 2. 从算法到系统：三个隐形瓶颈

### 2.1 反量化（dequantization）开销

低比特权重在 GPU 上**不能直接算矩阵乘**（Tensor Core 没有 INT4 指令，FP4 只在 Blackwell 有），通常要：

1. 读入打包的 4-bit 权重
2. 解包成 INT8/FP16 格式
3. 乘 scale（dequant）
4. 才能喂给 Tensor Core

如果第 2–3 步在寄存器/内存里反复做，省下的带宽可能被计算开销吃回去。**"4-bit 权重= $2$倍速度"只在 kernel 把 dequant 做进流水线时才成立。**

### 2.2 内存布局与打包

4-bit 不是字节对齐的（2 个$4-bit = 1$字节）。怎么把 4-bit 值塞进寄存器、怎么按 group 排列 scale，决定 kernel 效率。差的布局会让访存模式碎裂。

### 2.3 kernel 覆盖

量化收益只在"有对应 kernel 的层"兑现：

$$
W8A8 \to INT8 GEMM kernel（Ampere+）
$$
$W4A16 \to INT4 weight-only kernel$（各家自定义）
$W4A8KV4 \to$专门的 4-bit GEMM + KV 4-bit attention（QServe）

没有 kernel 的层退回 FP16，收益就漏掉一块。

---

## 3. QServe：W4A8KV4 的算法-系统协同

### 3.1 QoQ：W4A8KV4 量化方案

QoQ（Quattuor-Octo-Quattuor，拉丁语 4-8-4）：

权重：4-bit（group 128 或 per-channel），配合激活感知缩放（AWQ 式）做"渐进量化"
激活：8-bit（INT8/FP8）
$$
KV cache：4-bit
$$
质量：WikiText-2 PPL 与 FP16 几乎无差（近无损）

"渐进量化"（progressive quantization）是关键算法设计：让 W4A8 GEMM 能在 **INT8 Tensor Core** 上执行（把 4-bit 权重解包成两个 INT8 或直接按 INT8 对齐布局），避免 Blackwell 之前的硬件不支持。

### 3.2 两个"为什么"

**为什么 W4A8，而不是 W4A4？**

W4A4 的 GEMM 主循环里，权重和激活都要解包/去量化，每步开销更大
W4A8：激活 8-bit 可直接进 INT8 Tensor Core，只有权重需要解包$\to$主循环更干净
$\to$端到端反而更快（QServe 论文专门分析了这个 tradeoff）

**为什么 KV4？**

大 batch 下 attention 占运行时 50%+（$batch=64$时）
KV 4-bit 让 attention 的峰值性能翻倍（相对 KV8）
$\to W4A8KV4$是"权重省带宽 + 激活保计算 + KV 救 attention"的组合

### 3.3 系统技术

1. compute-aware weight reordering：按 kernel 的访存模式重排权重布局
2. register-level parallelism：把 dequant 摊进寄存器流水，隐藏开销
3. 专用 attention kernel：4-bit KV 读取 + 反量化 + FP16 计算

### 3.4 性能

相比当时工业界最强基线 TensorRT-LLM：
  一般模型吞吐高 1.2–1.4x
  Qwen1.5-72B 高 2.4–3.5x
  L40S（低端卡）跑 QServe 的吞吐超过 A100 上 TensorRT-LLM（8 个模型里 6 个）

**结论**：量化方案的收益上限由"算法 × kernel × 调度"共同决定——这是"协同设计"的价值。

---

## 4. 注意力量化（前沿）

Inference Engineering 的默认建议：**attention 留在原精度**（softmax 太敏感）。但前沿工作在突破：

| 方法 | 做法 | 状态 |
|---|---|---|
| FlashAttention-3（2024） | Hopper 上 FP8 attention + 异步拷贝 | 生产化中 |
| SageAttention（2024） | 8-bit 量化 Q/K/V，精度保持，图像/视频生成加速 | 研究+落地中 |

工程建议：

默认：QK/softmax 用 FP16/FP32
FP8 attention 只在：① 有专用 kernel ② 长序列 attention 占比高 ③ 评测通过

---

## 5. 推理引擎的量化支持地图

| 引擎 | 支持的量化 | 特点 |
|---|---|---|
| [vLLM](https://docs.vllm.ai/en/latest/features/quantization/) | AWQ、GPTQ、FP8(E4M3/E5M2)、MXFP8、FP4(Blackwell)、KV cache 量化 | 生态最广；PagedAttention；生产首选之一 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | FP8、INT4/FP4、SmoothQuant W8A8、AWQ、KV 量化 | NVIDIA 深度优化；部署灵活 |
| [SGLang](https://github.com/sgl-project/sglang) | 同上（复用多种后端） | 高性能 + 结构化生成 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF：$Q2_K\sim Q8_0$、IQ 系列、MXFP4 | 本地/边缘；格式事实标准 |
| [bitsandbytes / HF Transformers](https://huggingface.co/docs/transformers/quantization) | NF4/INT8（bitsandbytes）、GPTQ、AWQ、FP8 | 研究/微调（QLoRA） |
| [MLX（Apple）](https://github.com/ml-explore/mlx) | 4-bit 量化（MLX 生态） | Apple Silicon 本地部署 |

选引擎的检查清单：

1. 我要的量化格式有没有 kernel（不是"支持"而是"实测速度"）
2. 激活是否也量化（W8A8 还是只 W4A16）
3. KV cache 量化是否可开关
4. 长上下文、GQA、MoE 是否适配
5. 是否支持我要的硬件（Hopper / Blackwell / 本地）

---

## 6. 组合优化栈：无损优先

Inference Engineering Ch5 的关键提示：**除了量化，本章其他技术都是无损的**。所以优化顺序：

第一优先（无损）：
  1. 前缀缓存 / KV 复用（相同前缀不算第二次）
  2. 推测解码（draft 模型多 token 猜测）
  3. PagedAttention 类显存管理

第二优先（有损但收益大）：
  4. W8A8 FP8（权重+激活，01 章带宽/FLOPS 账本）
  5. KV cache 4-bit/FP8（长上下文显存）
  6. W4A8KV4（激进，Blackwell/QServe 类系统）

最后（按需）：
  7. 模型并行 / 预填充-解码解耦（另一套工程）

> 记忆：**"先白嫖（无损），再冒险（量化），最后上重型工程（并行/解耦）。"**

---

## 7. 部署决策树（含验收闭环）

① 有 GPU 与模型$\to$先跑 FP16 基线（速度/质量双基线）
② 质量必须无损$\to$只用无损优化（缓存/推测解码）
③ 可以接受"验证过的"风险$\to$
   默认配方：W8A8 FP8 + KV 4-bit/FP8，attention 留 FP16
   激进：W4A8KV4（QServe）/ FP4（Blackwell）
④ 每档都走 10 章的 Gate 1–4 验收
⑤ 不达标$\to$降档：更细粒度$\to$更高位宽$\to$换方法$\to$局部 QAT
⑥ 达标$\to$灰度$A/B \to$上线$\to$监控长尾/长上下文

---

## 8. 本章小结

1. **算法好 ≠ 系统快**：dequant、内存布局、kernel 覆盖是三个隐形瓶颈。
2. **QServe 示范了协同设计**：W4A8KV4（渐进量化让 4-bit 权重跑在 INT8 Tensor Core）+ 权重重排 + 寄存器级 dequant + KV4 attention；vs TensorRT-LLM 高 1.2–1.4x（72B 上 2.4–3.5x）。
3. **attention 默认不动**，FP8 attention 是前沿选项（FA3/SageAttention）。
4. **引擎选择看 kernel 实测**，不是看支持列表。
5. **无损优化优先**：缓存/推测解码先上，量化后上，并行/解耦最后。
6. **部署 = 量化 × 系统 × 验收闭环**（接 10 章）。

> 一句话记忆：**"量化解决'数据太多'，系统解决'解包太慢'，验收解决'到底行不行'——三件事一起做，才是推理工程的量化。"**

---

## 9. 习题与解答

### 题 1（思考）：为什么 4-bit 不一定快

一个 naive 的 W4A16 kernel 比 W8A8 慢，可能的原因有哪些？

<details>
<summary>题 1 解答</summary>

① 4-bit 解包/去量化开销大（无原生 INT4 Tensor Core）；② 内存布局碎片化导致访存效率低；③ 激活 16-bit 让计算量不变，只是省了权重带宽，而带宽收益被 kernel 开销抵消；④ scale 折叠没做好，多了一轮乘法。
</details>

### 题 2（设计）：给 70B 部署选型

场景：H200（单卡 141GB）、70B 模型、32K 上下文、要求 PPL 无损 + 尽量快。给出方案组合与理由。

<details>
<summary>题 2 解答要点</summary>

无损优先：前缀缓存 + PagedAttention。量化：W8A8 FP8（权重+激活近无损、FLOPS 翻倍）+ KV 4-bit（32K 上下文显存大头，08 章算过≈$10.7GB\to 2.7GB$）。attention 留 FP16。引擎：vLLM 或 TensorRT-LLM（FP8 kernel 成熟）。验收：10 章 Gate 1–4。
</details>

### 题 3（对比）：W4A16 vs W4A8KV4

两者权重都是 4-bit，为什么端到端收益不同？

<details>
<summary>题 3 解答</summary>

W4A16 只省权重带宽（decode 受益，prefill 计算仍是 16-bit）；W4A8KV4 的激活 8-bit 让 prefill 也用上低精度 Tensor Core（2x FLOPS），KV4 让 attention 带宽减半——三段（权重、激活、KV）都受益，端到端收益更高，但系统复杂度也更高。
</details>

### 题 4（排错）：量化后速度没提升

W8A8 量化部署后 TPS 没变，排查思路（至少 4 条）。

<details>
<summary>题 4 解答要点</summary>

① kernel 是否真的生效（profile 看算子的数据类型）；② 是否只有权重量化、激活还是 FP16（退化成 W8A16）；③ 是否被注意力/非量化算子占比掩盖（短序列时 attention 占比高）；④ 是否 batch 太小（带宽没打满）；⑤ 是否 KV/前缀缓存把瓶颈移到别处。
</details>

### 题 5（开放）：量化 × 推测解码

推测解码（无损）和量化（有损）叠加时，验收要注意什么？

<details>
<summary>题 5 解答要点</summary>

两者独立验收后再做联合验收：① draft 模型与 target 量化是否引入接受率下降（量化误差影响 draft 一致性）；② 联合后的端到端质量仍要走 Gate 1–4；③ 速度收益要分别归因（量化省带宽、推测省步数），避免把噪声当收益。
</details>

---

## 10. 延伸阅读

1. [QServe / QoQ（arXiv:2405.04532）](https://arxiv.org/abs/2405.04532)；[代码](https://github.com/mit-han-lab/qserve)
2. [FlashAttention-3（arXiv:2407.08691）](https://arxiv.org/abs/2407.08691)
3. [SageAttention（arXiv:2410.02367）](https://arxiv.org/abs/2410.02367)
4. [vLLM 量化文档](https://docs.vllm.ai/en/latest/features/quantization/)；[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)；[llama.cpp](https://github.com/ggml-org/llama.cpp)
5. 上一篇：[10 质量评估方法论](./LLM量化精读笔记-10-质量评估方法论.md)；本系列完结，回到 [00 总览](./LLM量化精读笔记-00-总览与学习地图.md)。

