# LLM 注意力与计算内核精读笔记 · 05 PagedAttention 与 KV 显存管理

> 对应：Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*（arXiv:2309.06180，SOSP 2023），vLLM 论文。
> 前置：01 章（KV cache 大小公式）、02 章（FlashAttention 的分块）、03 章（$d_{\text{kv}}$ 压缩）。学完本章你应该能：① 说出 KV 显存管理的三种浪费（预留、内部碎片、外部碎片）与论文实测的有效利用率（20.4%–38.2%）；② 解释 PagedAttention 的"逻辑块→物理块"分页思想与 block table；③ 说明 PagedAttention 与 FlashAttention 的关系与区别；④ 解释 copy-on-write 如何让并行采样/beam search 共享 KV；⑤ 手算 KV 每 token 字节数与分块账本。

---

## 目录（本章）

1. 本章目标
2. 问题：KV 显存管理为什么低效
3. 核心思想：像操作系统一样分页
4. PagedAttention：分块注意力计算
5. vLLM：块管理、按需分配与抢占调度
6. 共享：Copy-on-Write 与解码算法
7. 块大小权衡
8. KV 优化全景：维度、内容、布局
9. 数值算例
10. 实验数据
11. 本章小结
12. 习题与解答
13. 延伸阅读

---

## 2. 问题：KV 显存管理为什么低效

01 章算过 KV 账本：7B 模型 512KB/token。vLLM 论文给了一个更直观的例子——**OPT-13B**：

$$
2 \times 5120 \times 40 \times 2 = 800\ \text{KB/token}
$$

一条最多 2048 token 的请求，KV cache 高达 **1.6GB**。更糟的是，现有 serving 系统里这些显存**大部分是浪费的**：

```
浪费 1（预留 reserved）：为"可能达到的最大长度"提前占位，
                         占着茅坑不拉屎，别的请求进不来
浪费 2（内部碎片）：请求实际长度 << 最大长度，预分配的槽位用不满
浪费 3（外部碎片）：buddy allocator 等分配器产生的不连续空洞
```

论文实测（Fig. 2）：现有系统的 KV 显存**只有 20.4%–38.2% 真正存了 token**——约六到八成被浪费。同时 GPU 趋势是"算力翻倍、显存不涨"（A100→H100：FLOPS 翻倍，80GB 不变），显存会越来越是瓶颈。

---

## 3. 核心思想：像操作系统一样分页

操作系统解决"内存碎片 + 共享"的经典方案是**虚拟内存分页**。PagedAttention 把它搬进 GPU：

```
把每个请求的 KV cache 切成固定大小的"逻辑块"（默认 16 token/块）
物理块：GPU DRAM 里预先划好的等大内存片
block table：记录"逻辑块 → 物理块"的映射 + 每块已填充位置数
```

为什么这样能治病：

```
按需分配（要多少块给多少块）→ 消除预留与内部碎片
所有物理块等大、可任意排列 → 消除外部碎片
块是共享/复用的最小单位 → 支持跨请求共享（第 6 节）
```

---

## 4. PagedAttention：分块注意力计算

KV 不再连续存放，注意力计算也要改成**块级**。设块大小 $B$，第 $j$ 个 key 块：

$$
K_j = (k_{(j-1)B+1}, \dots, k_{jB}), \qquad
V_j = (v_{(j-1)B+1}, \dots, v_{jB})
$$

注意力输出按块累加：

$$
\mathbf{O}_i = \sum_j \mathbf{P}_{ij} \mathbf{V}_j, \qquad
\mathbf{S}_{ij} = \mathbf{Q}_i \mathbf{K}_j^\top
$$

softmax 的归一化跨块进行——**用的正是 02 章 online softmax 的机制**（运行 max/ℓ 逐块合并）。所以：

```
FlashAttention 解决"怎么在 SRAM 里高效算"
PagedAttention 解决"KV 在 HBM 里怎么放"（非连续、可共享）
两者正交，工程上同时使用
```

PagedAttention 的额外挑战：一个 batch 里各请求长度不同、块位置不同，注意力 kernel 要按 block table **动态取块**并处理变长序列。

---

## 5. vLLM：块管理、按需分配与抢占调度

vLLM 把 PagedAttention 做成完整 serving 引擎：

```
① 块引擎：GPU DRAM 划成物理块池；每个请求一张 block table
② 按需分配：token 只进已填满块之后的下一个空位，满块才申请新物理块
   → 请求级浪费被压到"一个块以内"
③ 抢占调度：显存不足时挂起请求、释放其块（或换到 CPU RAM），
   再按 block table 恢复——类似 OS 的换页
④ 连续批处理：每次迭代把"能算的请求"打包，与块管理协同
```

结果：**KV cache 接近零浪费**，batch 可以显著加大，吞吐随之提升。

---

## 6. 共享：Copy-on-Write 与解码算法

### 6.1 并行采样

一个 prompt 生成多个输出时，**prompt 部分的 KV 完全相同**，可以让多条输出共享同一批物理块。只有各自新生成的 token 需要独占空间；如果新 token 要写进一个被共享的旧块，就用 **copy-on-write（COW）**：复制该块、改引用计数，不污染其他输出。

### 6.2 Beam Search

beam search 的多个候选不仅共享 prompt，还会在解码过程中动态分叉/合并——共享模式像 OS 的进程树。vLLM 按块管理引用计数：

```
候选被淘汰 → 引用计数减一 → 归零的物理块释放复用
新候选加入 → 分配新物理块
论文报告：beam search 场景下共享可省最多 55% 的 KV 显存
```

对比：旧系统要在候选之间**拷贝 KV**（代价高）；分页共享让"复制"退化为"多一行 block table 指向同一物理块"。

---

## 7. 块大小权衡

$$
\text{块越大：} \begin{cases}
\text{好：一次 kernel 处理更多 token，并行度高、延迟低} \\
\text{坏：最后一个块可能用不满 → 内部碎片变大}
\end{cases}
$$

$$
\text{块越小：} \text{碎片少，但 kernel 开销/块表开销大}
$$

默认 16 token/块是实践折中；论文 §7.2 专门做了块大小敏感性实验。**碎片与并行度的平衡点取决于模型与负载**。

---

## 8. KV 优化全景：维度、内容、布局

把 03/04/05 三章放在一张图里：

```
KV cache 优化
├── 维度（03 章）：MQA/GQA/MLA —— 每 token 存多少
├── 内容（04 章）：滑窗/驱逐/稀疏 —— 存哪些 token
└── 布局（05 章）：PagedAttention —— 怎么存放、怎么共享
三者可叠加：MLA 减少每 token 字节 → H2O 决定留谁 → 分页管理剩余
```

---

## 9. 数值算例

### 9.1 OPT-13B 账本（论文原例）

$$
\text{每 token： } 2 \times 5120 \times 40 \times 2 = 800\ \text{KB}
$$

$$
\text{单请求上限 2048 token： } 800\ \text{KB} \times 2048 = 1.6\ \text{GB}
$$

### 9.2 分页示例

7B 模型（512KB/token）、块大小 16 token：一条 40 token 的请求需要 $40/16 \to 3$ 个逻辑块（16+16+8），映射到物理块（如 5, 2, 9 号）——物理上不连续、按需分配：

```
逻辑块 1 → 物理块 5（满）
逻辑块 2 → 物理块 2（满）
逻辑块 3 → 物理块 9（8/16 填充）
```

若三个请求共享同一 prompt（并行采样），逻辑块 1 可以指向**同一个物理块 5**，只各配一块新的 COW 块。

---

## 10. 实验数据

| 指标 | 数据 | 出处 |
|---|---|---|
| 现有系统 KV 有效利用率 | 仅 20.4%–38.2% | vLLM Fig. 2 |
| OPT-13B KV 账本 | 800KB/token；2048 token = 1.6GB | vLLM §3.2 |
| vLLM 吞吐 | 比 FasterTransformer/Orca 高 **2–4x**（同等延迟，不影响精度） | vLLM 摘要 |
| beam search 共享 | 最多省 55% KV 显存 | vLLM §6.3 |
| 块大小 | 默认 16 token/块；§7.2 做敏感性分析 | vLLM §4.1/§7.2 |

---

## 11. 本章小结

1. **问题**：KV 显存浪费来自预留、内部碎片、外部碎片，实测有效利用率只有 20.4%–38.2%。
2. **方案**：分页（固定大小逻辑块 + block table + 按需分配），把碎片压到"一个块以内"。
3. **计算**：PagedAttention 用 02 章的 online softmax 逐块合并；与 FlashAttention 正交叠加。
4. **共享**：COW + 引用计数让并行采样、beam search 共享 KV（最多省 55%）。
5. **全景**：维度（03）→ 内容（04）→ 布局（05），三层可叠加。

> 一句话记忆：**"KV 显存管理就是给 GPU 装一个操作系统——把整块地皮切成统一大小的页，谁要谁领、不用就还，多个进程（beam/采样）还能共享只读页。"**

---

## 12. 习题与解答

### 题 1（计算）：KV 账本与分块

LLaMA-2 70B（80 层、GQA-8、$d=8192$、$d_{\text{head}}=128$、BF16）：① 每 token KV 字节数；② 若块大小 16，一条 1000 token 请求需要多少逻辑块；③ 最后一块的填充率。

<details>
<summary>题 1 解答</summary>

① $2 \times 80 \times 8 \times 128 \times 2 = 320\ \text{KB/token}$。② $1000/16 = 62.5 \to 63$ 个逻辑块。③ 前 62 块满（$62\times16=992$），第 63 块填充 $8/16 = 50\%$——整条请求的浪费被压到"不到一个块"。
</details>

### 题 2（推导）：为什么"等大块 + 按需分配"同时消两种碎片

解释内部碎片与外部碎片各自的成因，以及分页方案分别怎么消除。

<details>
<summary>题 2 解答</summary>

内部碎片来自"预分配最大长度"（预留超过实际所需）；按需分配让每条请求只占用已产生的 token 的块，浪费被限制在最后一个不满块内。外部碎片来自"不同请求预分配不同大小"造成的分配器空洞；所有物理块等大、逻辑到物理自由映射，空洞不再存在。
</details>

### 题 3（思考）：PagedAttention vs FlashAttention

两者都"分块"，解决的是同一个问题吗？工程上如何叠加？

<details>
<summary>题 3 解答要点</summary>

不是。FlashAttention 解决"计算时中间矩阵不落 HBM"（IO 优化，KV 仍是连续存储）；PagedAttention 解决"KV 在 HBM 里的布局与共享"（内存管理）。FlashAttention kernel 处理连续块，PagedAttention kernel 按 block table 取非连续块；vLLM 等引擎两者同时启用。
</details>

### 题 4（设计）：COW 引用计数

并行采样 4 条输出共享 prompt 的物理块 7。① 第 3 条输出要写入块 7 的新位置，流程是什么？② 块 7 的引用计数如何变化？

<details>
<summary>题 4 解答</summary>

① 检测到块 7 被 4 条输出共享 → 复制块 7 为新物理块 7'，把第 3 条输出的 block table 指向 7'，其余 3 条仍指向 7；写入 7' 的新 token。② 7 的引用计数从 4 降到 3；7' 引用计数为 1。任何输出结束后引用计数减一，归零即释放。
</details>

### 题 5（对比）：三种"省 KV"的层次

用一句话分别说明 MLA（03）、H2O/StreamingLLM（04）、PagedAttention（05）在 KV 上动的是什么，并给一个三者叠加的部署组合。

<details>
<summary>题 5 解答要点</summary>

MLA：每 token 存更少（维度）；H2O/StreamingLLM：存更少的 token（内容）；PagedAttention：存得更紧凑、可共享（布局）。叠加示例：MLA 压缩维度 + H2O 驱逐低频 token + 分页管理剩余 KV 并让并行采样共享。
</details>

### 题 6（编程）：分块分配模拟器

实现一个简单模拟器：给定请求到达/离开序列与块大小，① 按需分配物理块（block table）；② 统计三种方案的碎片率：预分配最大长度 / 按需分配 / 统一分页；③ 验证"按需分页"碎片率趋近 0。

<details>
<summary>题 6 解答要点</summary>

预分配：每条请求占"最大长度/块"块，请求提前结束时大量空闲；按需：请求结束时只剩最后一个不满块；统一分页 + 引用计数：请求间可复用、释放即时。统计总占用/总分配即可复现 vLLM Fig. 2 的结论。
</details>

---

## 13. 延伸阅读

1. [Efficient Memory Management for LLM Serving with PagedAttention（arXiv:2309.06180）](https://arxiv.org/abs/2309.06180)：本章全部内容的出处（§3 浪费分析、§4 算法、§6 共享、§7 块大小）。
2. [vLLM 文档](https://docs.vllm.ai/)：PagedAttention 的生产实现、前缀缓存、并行采样。
3. [RadixAttention / SGLang（arXiv:2312.07104）](https://arxiv.org/abs/2312.07104)：前缀缓存的树形管理，PagedAttention 思想的延伸。
4. 上一篇：[04 稀疏、滑动窗口与线性注意力](./LLM注意力内核精读笔记-04-稀疏滑动窗口与线性注意力.md)；下一篇：**06 内核优化与算子融合**——从算法回到硬件：FLOPs/带宽模型、算子融合、Tensor Core、FP8 注意力与编译优化。
