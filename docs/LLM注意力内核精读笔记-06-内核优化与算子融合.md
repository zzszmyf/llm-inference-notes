# LLM 注意力与计算内核精读笔记 · 06 内核优化与算子融合

> 对应：Williams et al., *Roofline: An Insightful Visual Performance Model for Multicore Architectures*（2009）；FlashAttention 系列（02 章已核实数据）；vLLM / TensorRT-LLM / Triton 工程实践。
> 前置：01 章（FLOPs 与 KV 账本）、02 章（IO 感知）、03/05 章（KV 优化）、量化系列 03 章（带宽模型）。学完本章你应该能：① 用算术强度（FLOPs/byte）判断一个 kernel 是计算受限还是带宽受限；② 推导 decode 与 prefill 分别落在 Roofline 的哪一侧；③ 说出算子融合的三种层次（elementwise、reduction、整体融合）与 FlashAttention 融合了什么；④ 解释 Tensor Core 与 FP8 张量核对注意力的意义；⑤ 对比 CUDA Graph、Triton、torch.compile 的适用场景。

---

## 目录（本章）

1. 本章目标
2. 硬件性能模型：Roofline 与算术强度
3. decode 带宽受限、prefill 计算受限
4. 算子融合：三种层次
5. Tensor Core 与矩阵乘组织
6. FP8 注意力与低精度内核
7. CUDA Graph、Triton 与编译优化
8. 注意力内核的效率曲线（FA1→FA3）
9. 数值算例：Roofline 判定
10. 本章小结
11. 习题与解答
12. 延伸阅读

---

## 2. 硬件性能模型：Roofline 与算术强度

一个 kernel 快不快，先问一句：**它缺的是算力还是带宽？** 定义算术强度：

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{HBM bytes}}
$$

Roofline 模型给出两个天花板：

$$
\text{峰值} \le \min\left(\text{峰值 FLOPs/s},\ \text{峰值带宽} \times \text{算术强度}\right)
$$

```
算术强度低于"山脊"（ridge point）→ 带宽受限：加算力没用，减搬移才有用
算术强度高于山脊 → 计算受限：加算力/减 FLOPs 才有用
```

A100 的量级：约 312 TFLOPS（BF16）/ 2 TB/s ≈ 山脊在 150 FLOP/byte 附近。

---

## 3. decode 带宽受限、prefill 计算受限

### 3.1 decode（单 token）

每步 FLOPs ≈ 全部层的注意力 + MLP：

$$
\text{FLOPs} \approx n_{\text{layers}} \times \left(4Ld + 8d^2\right)
$$

每步搬移的字节 ≈ 权重（$2 \times$ 参数量）+ KV（02/03 章）：

$$
\text{bytes} \approx \text{weights} + 2 L\, d_{\text{kv}} n_{\text{layers}} b
$$

7B 模型、$L=4K$：约 6.4 GFLOP vs 15 GB 搬移 → 算术强度 $\approx 0.4$，远低于山脊 → **带宽受限**。

### 3.2 prefill（长输入）

注意力单层 FLOPs $= 4L^2d$，输入字节约 $3Ld$：

$$
\text{AI}_{\text{attn}} \approx \frac{4L^2d}{3Ld} = \frac{4L}{3}
$$

$L = 4096$ 时约 5461 FLOP/byte，远高于山脊 → **计算受限**。

### 3.3 推论

```
decode 优化 → 减搬移（量化权重、GQA/MLA 减 KV、分页防浪费）
prefill 优化 → 减 FLOPs 或提高利用率（FlashAttention、稀疏/线性、FP8 张量核）
```

这解释了整个系列的路线选择：02 章 FlashAttention 主攻 prefill，03/05 章主攻 decode。

---

## 4. 算子融合：三种层次

"融合"= 把多个 kernel 合并成一个，减少两件事：**kernel 启动开销**和**中间张量在 HBM 的读写**。

### 4.1 Elementwise 融合

逐元素算子（激活、残差、缩放）读写比极低，最该融合：

```
例：LayerNorm + residual + dropout + 激活 → 一个 kernel
收益：省 4 次中间张量落盘 + 4 次 kernel 启动
```

### 4.2 Reduction 融合

含归约的算子（softmax、normalization）与相邻计算融合，避免中间结果落盘：

```
例：FlashAttention 的整个注意力 = 一个融合 kernel
   QK^T（GEMM）→ 掩码 → rowmax → exp → rowsum → rescale → PV（GEMM）
   全在 SRAM 里完成，S/P 矩阵永不落 HBM（02 章）
```

### 4.3 整体 kernel 融合（层级）

把一层的前向合成一个或少数几个 kernel（如 QKV 投影融合、MLP 两段融合、MQA/GQA 的 KV 头广播融合）。现代引擎（TensorRT-LLM、vLLM 的 CUDA 核）大量采用。

---

## 5. Tensor Core 与矩阵乘组织

注意力最贵的两次 GEMM（QKᵀ、PV）在 Tensor Core 上执行。关键事实：

```
优化过的 GEMM 能到 80–90% 理论峰值（FA2 论文实测）
而 FA1 注意力只到 30–50%（前向）、25–35%（反向）
→ 差距全部来自"注意力里非 GEMM 的部分"和 IO 组织
```

FA2/FA3 就是把这个差距补上：

```
FA2：warp 间划分 KV 维度，减少共享内存读写 → 73% 峰值
FA3：wgmma（warpgroup 矩阵乘）+ TMA 异步搬运 + 软max 藏在 GEMM 下
     → 75% 峰值（FP16 740 TFLOPS；FP8 约 1.2 PFLOPs on H100）
```

---

## 6. FP8 注意力与低精度内核

Hopper/Blackwell 的 FP8 张量核吞吐是 FP16 的约 **2 倍**（FA3 原文：WGMMA 在 FP8 下每 SM 吞吐翻倍）。注意力用 FP8 有两个难点：

```
① 数值：注意力分数/softmax 对精度敏感（01 章结论）
   → FA3 用块级量化（per-block scale）+ incoherent processing 打散离群值
   → 误差比基线 FP8 注意力低 2.6 倍
② 布局：WGMMA 对 FP8 操作数的内存布局有严格要求，需要专门的转换
```

注意：**FP8 注意力是"近似"**（低精度），与 FlashAttention 的"精确"是两回事——前者省的是数值位宽，后者省的是 IO。生产上两者可叠加（如 QServe 的 FP8 Attention，量化系列 11 章）。

---

## 7. CUDA Graph、Triton 与编译优化

```
CUDA Graph：把一串 kernel launch 固化成图，一次提交、GPU 端自动调度
  → 消除 CPU 端 launch 开销（小 batch/短序列时收益显著）
  → 推理引擎（vLLM 等）普遍使用

Triton：类 Python 的 kernel 语言，自动做 tile 划分与共享内存分配
  → 让研究者写出接近手写 CUDA 性能的 kernel（FlashAttention 有 Triton 版）

torch.compile：把 PyTorch 计算图做算子融合 + 代码生成
  → 训练/研究场景的开箱即用优化
```

选择逻辑：追求极致性能且算力充足 → 手写/FA 库；需要快速迭代 → Triton / torch.compile；部署固定形状 → CUDA Graph + 预编译引擎。

---

## 8. 注意力内核的效率曲线（FA1→FA3）

| 实现 | 峰值利用率 | 说明 |
|---|---:|---|
| 朴素 PyTorch 注意力 | ~低 | 中间矩阵多次落盘 |
| FlashAttention-1 | 30–50%（前向） | 提出 IO 感知框架 |
| FlashAttention-2 | 73%（230 TFLOPS on A100） | 并行与工作划分 |
| FlashAttention-3 | 75%（740 TFLOPS on H100）；FP8 ≈1.2 PFLOPs | 异步 + 低精度 |
| 优化 GEMM | 80–90% | 注意力的"天花板参照" |

从 30% 到 75% 的每一步，都是在**逼近 GEMM 的效率**——注意力的内核优化史，就是"把非 GEMM 的部分藏进 GEMM"的历史。

---

## 9. 数值算例：Roofline 判定

### 9.1 decode：7B、$L=4K$

$$
\text{FLOPs} \approx 32 \times (4\times4096\times4096 + 8\times4096^2) = 32 \times 2.01\times10^8 \approx 6.4\ \text{GFLOP}
$$

$$
\text{bytes} \approx 13\ \text{GB（权重）} + 2\ \text{GB（KV）} = 15\ \text{GB}
$$

$$
\text{AI} = 6.4/15000 \approx 0.43\ \text{FLOP/byte} \ll 150 \Rightarrow \textbf{带宽受限}
$$

### 9.2 prefill：$L=4K$ 单层注意力

$$
\text{AI} = \frac{4L^2d}{3Ld} = \frac{4L}{3} = 5461 \gg 150 \Rightarrow \textbf{计算受限}
$$

### 9.3 优化方向验证

```
decode：量化权重（带宽减半）→ 每步时间接近减半 ✓（量化系列 03 章）
prefill：FP8 张量核（FLOPs/s 翻倍）→ 长 prefill 明显提速 ✓（FA3）
```

---

## 10. 本章小结

1. **Roofline 是决策框架**：先算算术强度，再决定"减搬移"还是"减 FLOPs"。
2. **decode 带宽受限、prefill 计算受限**——整个系列的优化路线由此展开。
3. **融合三层次**：elementwise → reduction → 整体；FlashAttention 是 reduction 融合的典范。
4. **Tensor Core + FP8** 是算力侧的两大杠杆；FP8 注意力需要块级量化与 incoherent processing。
5. **工程手段**：CUDA Graph 消启动、Triton/torch.compile 提效率。

> 一句话记忆：**"先问 Roofline 缺什么：decode 缺带宽就给内存做减法（量化/GQA/分页），prefill 缺算力就给计算做减法（FlashAttention/FP8/稀疏）——所有内核优化都是把'瓶颈环节'藏到'不瓶颈的硬件'里。"**

---

## 11. 习题与解答

### 题 1（推导）：decode 的算术强度

推导 decode 每步 FLOPs 与搬移字节的表达式，并说明"为什么加算力救不了 decode"。

<details>
<summary>题 1 解答</summary>

FLOPs $\approx n_{\text{layers}}(4Ld + 8d^2)$，bytes $\approx$ 权重 + $2L d_{\text{kv}} n_{\text{layers}} b$。7B/4K 时 AI ≈ 0.4，远低于 A100 山脊（~150），处于带宽受限区——此时 FLOPs/s 再高也白搭，只能减搬移（量化权重、GQA/MLA、分页）。
</details>

### 题 2（计算）：Roofline 判定

70B 模型 decode（$L=4K$、权重 130GB、GQA-8 的 KV 每 token 320KB）：算 FLOPs、bytes、AI，并判断瓶颈。

<details>
<summary>题 2 解答</summary>

FLOPs ≈ $80\times(4\times4096\times8192 + 8\times8192^2) = 80\times(1.34\times10^8 + 5.37\times10^8) \approx 54$ GFLOP；bytes ≈ $130 + 320\text{KB}\times4096 = 130 + 1.25 \approx 131$ GB；AI ≈ 0.41 → 带宽受限。KV 只占 1%：**70B 的 decode 瓶颈几乎全在权重搬移**（这也是量化对超大模型收益大的原因）。
</details>

### 题 3（思考）：为什么 FA 是"融合"不是"稀疏"

说明 FlashAttention 与稀疏/线性注意力的本质区别，以及为什么前者在"精确性"上零代价。

<details>
<summary>题 3 解答要点</summary>

FA 不改变注意力数学，只改变计算组织（分块 + 在线 softmax + 重算），结果逐位等价；稀疏/线性改变"看哪些 key"或"用什么核"，是近似。融合省的是 IO 和启动开销，近似省的是 FLOPs 本身——两者不冲突，可叠加。
</details>

### 题 4（设计）：给注意力列融合清单

列出从"朴素三步注意力"到 FlashAttention 之间融合/消除的 kernel 与中间张量，指出每一处省的是什么（启动？HBM 读写？）。

<details>
<summary>题 4 解答要点</summary>

朴素：QKᵀ kernel → 写 S（HBM）→ softmax kernel → 读 S 写 P（HBM）→ PV kernel → 读 P。融合后：单个 kernel 内完成 GEMM→mask→rowmax→exp→rowsum→rescale→GEMM，S/P 只在 SRAM；省 4 次 HBM 读写 + 3 次 kernel 启动。反向的 S/P 重算省掉 O(N²) 存盘。
</details>

### 题 5（对比）：FP8 注意力 vs FlashAttention 的"快"

两者都叫"加速注意力"，省的东西各是什么？生产上如何叠加，风险在哪？

<details>
<summary>题 5 解答要点</summary>

FlashAttention 省 HBM 访问（IO），结果精确；FP8 省每字节位数与张量核吞吐翻倍，结果是近似（需块级量化 + incoherent processing 保精度）。叠加 = 融合 kernel 内用 FP8 张量核；风险是数值误差复合，需要 07 章的精度验收。
</details>

### 题 6（编程）：kernel 启动开销测量

写一个 PyTorch 实验：对同一计算（如 LayerNorm+ReLU+残差），比较"三个 kernel"与"融合实现（torch.compile 或手写 CUDA）"的耗时，记录 GPU kernel 数量（可用 torch.profiler），并解释小输入时融合收益为什么更大。

<details>
<summary>题 6 解答要点</summary>

torch.profiler 会显示 kernel 个数从 3+ 降到 1；小输入时每个 kernel 的启动/调度开销占比高，融合收益最大；大输入时带宽占主导，融合主要省中间张量读写。这正好复现第 2 节的"IO 稀缺"观点。
</details>

---

## 12. 延伸阅读

1. [Roofline: An Insightful Visual Performance Model（Williams et al., 2009）](https://people.eecs.berkeley.edu/~kubitron/courses/cs267-S12/handouts/roofline.pdf)：本章性能模型的原始出处。
2. [FlashAttention 三篇论文](https://arxiv.org/abs/2205.14135)（02 章）：效率曲线的全部数据来源。
3. [Triton 官方文档](https://triton-lang.org/)：kernel 编写与自动优化。
4. [量化系列 03/11 章](./LLM量化精读笔记-03-数值格式与硬件.md)：带宽模型与 FP8 Attention 的部署案例。
5. 上一篇：[05 PagedAttention 与 KV 显存管理](./LLM注意力内核精读笔记-05-PagedAttention与KV显存管理.md)；下一篇：**07 系统集成与生产验收**——把本系列所有方法放进一份部署决策与验收协议（本系列完结章）。
