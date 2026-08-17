# LLM 推测解码精读笔记 · 04 EAGLE：特征空间草稿

> 对应：Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*（arXiv:2401.15077，2024）；Li et al., *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees*（arXiv:2406.16858，2024）。
> 前置：01–03 章（接受率数学、拒绝采样、树验证）。学完本章你应该能：① 说清"token 层自回归"与"特征层自回归"各自的不确定性来源；② 解释 EAGLE 的 shifted-token 输入为什么能消除特征分支歧义；③ 证明 EAGLE 每轮第一个草稿 token 的接受概率恒为 1；④ 写出 EAGLE 草稿模型的结构、训练损失与推理循环；⑤ 解释为什么特征误差小 ⟹ 接受率高（共享 LM head 的光滑性）；⑥ 描述 EAGLE-2 动态树的扩展/重排算法及其理论基础。

---

## 目录（本章）

1. 本章目标
2. 从 03 到 04：Medusa 的瓶颈在哪
3. 核心洞察：token 不确定性与特征不确定性
4. 架构：EAGLE 草稿模型
5. 训练：损失、噪声增强与成本
6. 推理循环：特征级草稿 + 树验证
7. 为什么接受率更高：三个机制
8. 数值算例：完整一轮
9. 实验结果
10. EAGLE-2：上下文感知的动态草稿树
11. 与 02/03 章对照
12. 实现细节与坑
13. 本章小结
14. 习题与解答
15. 延伸阅读

---

## 2. 从 03 到 04：Medusa 的瓶颈在哪

03 章 Medusa 用"多头 + 树"把每层接受从 top-1 命中率升级为 top-$s$ 覆盖率，实测接受率约 **0.6**。瓶颈在哪？看它的 head 公式：

$$
p_t^{(k)} = \mathrm{softmax}\!\left(W_2^{(k)}\left(\mathrm{SiLU}(W_1^{(k)} h_t) + h_t\right)\right)
$$

head $k$ 只依赖**当前** $h_t$，平行地猜第 $k$ 个未来 token——它不知道中间采样出了什么。这就是论文里说的"不确定性"：

```
前缀 "I"（特征 f_I）后面既可能接 "am"（特征 f_am），也可能接 "always"（特征 f_always）。
仅凭 f_I，谁也说不清下一个特征应该是 f_am 还是 f_always。
Medusa 的 head 只能"平均"两种可能，接受率天花板被按在 ~0.6。
```

EAGLE 的洞察：**不确定性不是 token 独有的，特征也有；但特征的不确定性可以被"一个时间步之前的 token 序列"消除**。先看两个观察，再看架构。

---

## 3. 核心洞察：token 不确定性与特征不确定性

### 3.1 记号

沿用论文记号：token $t$、token 嵌入 $e(t)$、特征 $f$（**第二顶层隐藏状态**，即 LM head 之前的表示）、分布 $p$。目标模型的计算链：

$$
T_{1:j} \;\to\; E_{1:j} \;\to\; f_j \;\to\; p_{j+1} = \mathrm{LM\_Head}(f_j) \;\to\; t_{j+1}
$$

"第二顶层"这个位置很关键：再往上就是 LM head 的线性映射，特征到分布的距离由 LM head 的平滑性决定（第 7 节展开）。

### 3.2 观察一：token 层自回归难，特征层自回归容易

token 是**离散**的，词表 $V$ 个类，分布熵高：一句话里"下一个词"的可能选择成百上千，分类误差大。特征是**连续**的、高维的：相邻 token 的特征通常也相邻，回归目标平滑。论文实验（Vicuna-7B，MT-bench，温度 0）：

```
token 层草稿：1.5x
特征层草稿：1.9x     ← 只把"猜 token"换成"猜特征"，就快了 27%
```

### 3.3 观察二：特征层自回归也有不确定性

但"猜特征"有个坑：下一个特征**取决于采样结果**。给定 $f_I$，"am" 和 "always" 都可能是下一个 token，它们的特征 $f_{am}$、$f_{always}$ 差别很大——模型不知道该往哪个方向回归。这就是论文标题里的 **feature uncertainty**。

### 3.4 EAGLE 的解法：shifted-token 输入

既然不确定来自"不知道采样出了什么"，那就**把采样结果告诉模型**：草稿模型输入序列里，token 序列整体**提前一个时间步**——

$$
\text{predict } f_{always} \text{ from } (f_I,\ t_{always}); \qquad
\text{predict } f_{am} \text{ from } (f_I,\ t_{am})
$$

一旦草稿模型已经采样出 $t_{always}$，它要预测的就是"这个 token 的特征"——这是一个**确定的回归目标**，不再有歧义。论文实验：加上 shifted-token 后，Vicuna-7B 的加速从 1.9x 涨到 **2.8x**。

---

## 4. 架构：EAGLE 草稿模型

### 4.1 组件

EAGLE 的草稿模型由三部分组成：

```
① Embedding 层：直接用目标模型的参数（冻结，不训练）
② LM Head：直接用目标模型的参数（冻结，不训练）
③ Autoregression Head：可训练（FC 投影 + 轻量 decoder）
```

输入是"特征序列 + 提前一步的 token 序列"的融合：

$$
\text{input}_j = \left[\, f_j \,\|\, e(t_{j+1}) \,\right] \in \mathbb{R}^{2d}
$$

（$f_j$ 与 $t_{j+1}$ 的嵌入拼接，$d$ 为隐藏维度。）Autoregression Head 先用 FC 把 $2d$ 压回 $d$，再过 decoder 层，输出下一个特征的预测：

$$
\hat{f}_{j+1} = M_{\text{draft}}(f_j,\ e(t_{j+1}))
$$

然后**用目标模型自己的 LM Head** 得到草稿分布并采样：

$$
\hat{p}_{j+2} = \mathrm{LM\_Head}(\hat{f}_{j+1}), \qquad t_{j+2} \sim \hat{p}_{j+2}
$$

采出的 token 与预测特征一起拼进下一轮输入，继续自回归。

### 4.2 为什么共享 Embedding 和 LM Head

```
共享 Embedding：token 语义空间一致，草稿与目标"说同一种语言"
共享 LM Head：草稿分布 q = softmax(G(ĥ)) 与目标分布 p = softmax(G(h))
            共享同一个映射 G；ĥ ≈ h 时 q ≈ p（第 7 节）
```

对比 Medusa：Medusa 的 head 也有各自的输出投影（$W_2^{(k)} \in \mathbb{R}^{d\times V}$），虽然初始化为 LM head，但训练后会漂移；EAGLE 的 LM head **彻底冻结共享**，草稿分布永远贴着目标分布走。

### 4.3 训练参数规模

| 目标模型 | EAGLE 可训练参数 | 占比（约） |
|---|---:|---:|
| LLaMA2-Chat 7B | 0.24B | 3.4% |
| LLaMA2-Chat 13B | 0.37B | 2.8% |
| Vicuna-33B | 0.56B | 1.7% |
| LLaMA2-Chat 70B | 0.99B | 1.4% |
| Mixtral 8x7B（MoE） | 0.28B | — |

训练成本：70B 模型用不超过 7 万条 ShareGPT 对话，4×A100（40G）训 1–2 天；7B–33B 在 RTX 3090 单机 1–2 天即可。一次训练永久复用，均摊成本随请求数趋近于 0。

---

## 5. 训练：损失、噪声增强与成本

预测特征是一个**回归**任务，预测 token 是**分类**任务，EAGLE 两个目标一起优化：

$$
\mathcal{L} = \underbrace{\mathcal{L}_{\text{smooth-L1}}(\hat{f}_{j+1},\ f_{j+1})}_{\text{特征回归}} + \underbrace{\mathcal{L}_{\text{CE}}(\hat{p}_{j+2},\ t_{j+2})}_{\text{token 分类}}
$$

（$\hat{f}$ 是草稿预测特征，$f$ 是目标模型真实特征；Smooth L1 比 L2 对离群特征更稳。）

**噪声增强**：推理时特征误差会沿自回归链累积（预测的 $\hat{f}$ 又被喂回去预测下一个 $\hat{f}$）。训练时对目标特征加均匀噪声：

$$
f \leftarrow f + u, \qquad u \sim \mathcal{U}(-0.1, 0.1)
$$

让草稿模型学会"在有点脏的特征上也能正确回归"，显著缓解误差累积。

---

## 6. 推理循环：特征级草稿 + 树验证

### 6.1 草稿阶段（特征级自回归）

```
输入：当前已接受前缀的最后一个真实特征 f_j

1. 草稿第 1 个 token：
     q_{j+1} = LM_Head(f_j)          # 与目标分布完全一致
     t̂_{j+1} ~ q_{j+1}
2. 草稿第 2 个 token：
     f̂_{j+1} = M_draft(f_j, e(t̂_{j+1}))
     q_{j+2} = LM_Head(f̂_{j+1})
     t̂_{j+2} ~ q_{j+2}
3. 草稿第 3 个 token：
     f̂_{j+2} = M_draft(f̂_{j+1}, e(t̂_{j+2}))
     q_{j+3} = LM_Head(f̂_{j+2})
     t̂_{j+3} ~ q_{j+3}
……重复到 K 个草稿 token
```

注意第 1 步的输入是**真实特征**，第 2 步起输入是**预测特征**——误差从第 2 步开始累积，这是接受率随位置下降的根源。

### 6.2 验证阶段（树注意力 + 递归拒绝采样）

与 03 章相同：多个候选分支构成树，目标一次前向（树注意力）算出所有节点概率，然后**递归地**逐节点做拒绝采样：

$$
\text{accept } \hat{t} \iff r \le \min\left(1,\ \frac{p(\hat{t})}{\hat{q}(\hat{t})}\right), \qquad
\text{否则从 } \mathrm{norm}(\max(0, p - \hat{q})) \text{ 重采样}
$$

这保证了分布严格等于目标模型（贪心与采样设置都成立）。验证结束后，**接受路径每个 token 的真实特征**被保存，作为下一轮草稿的起点——不需要再跑一遍目标前向。

---

## 7. 为什么接受率更高：三个机制

### 7.1 机制一：第一个草稿 token 恒被接受

草稿第 1 步 $q_{j+1} = \mathrm{LM\_Head}(f_j)$，而目标分布 $p_{j+1} = \mathrm{LM\_Head}(f_j)$——**同一个特征、同一个映射**，所以 $q_{j+1} \equiv p_{j+1}$。由 02 章引理：

$$
\text{accept prob} = \sum_x q(x)\min\!\left(1,\frac{p(x)}{q(x)}\right) = \sum_x \min(p,q) = 1 \quad (q \equiv p)
$$

**结论：EAGLE 每轮第一个草稿 token 就是目标分布的一个样本，接受概率恒为 1。** 接受率下降完全来自后续预测特征的误差累积。

### 7.2 机制二：共享 LM head 把"特征误差"平滑地传导为"分布误差"

设 $G = \mathrm{LM\_Head}$ 是**光滑**映射。01 章已知 $\alpha = 1 - \mathrm{TV}(p, q)$，而

$$
\mathrm{TV}\!\left(\mathrm{softmax}(G(\hat{f})),\ \mathrm{softmax}(G(f))\right) \le L \cdot \|\hat{f} - f\|
$$

（$L$ 与 softmax 雅可比和 $G$ 的 Lipschitz 常数有关）。**特征预测误差按常数倍传导为分布距离，进而线性地压低接受率**。特征回归误差是连续、可控的；而 token 分类一旦选错类，分布距离直接是 1——这就是"特征级草稿比 token 级草稿接受率高"的量化根源。

### 7.3 机制三：shifted-token 把"多目标回归"变成"单目标回归"

没有 shifted-token 时（Medusa 式），给定 $f_I$ 的回归目标是 $f_{am}$ 与 $f_{always}$ 的混合——模型被训练去"平均"两个不相容的目标，误差下限高。有 shifted-token 时，输入已经包含采样的 $t$，回归目标唯一。论文数据：

```
接受率（top-1，链式草稿）：
  Medusa：≈ 0.6
  Lookahead：更低
  EAGLE：≈ 0.8
```

三个机制合起来：**第一 token 免费 + 平滑传导 + 目标唯一**，把草稿接受率从 0.6 抬到 0.8。

---

## 8. 数值算例：完整一轮

场景：前缀 "I"，目标真实特征 $f_1$，草稿长度 $K = 3$。

**草稿阶段**（特征级自回归）：

```
第 1 步：q_2 = LM_Head(f_1)          ← 与目标分布相同，恒接受
         采样 t̂_2 = "am"
第 2 步：f̂_2 = M_draft(f_1, e("am"))
         q_3 = LM_Head(f̂_2)，采样 t̂_3 = "a"
第 3 步：f̂_3 = M_draft(f̂_2, e("a"))
         q_4 = LM_Head(f̂_3)，采样 t̂_4 = "student"
```

**验证阶段**：目标一次前向计算 "I am a student" 的分布 $p_2, p_3, p_4$，逐位拒绝采样。假设接受率剖面：

$$
\alpha_1 = 1.0 \quad(\text{真实特征，} q_2 \equiv p_2), \qquad
\alpha_2 = 0.9, \qquad \alpha_3 = 0.7
$$

期望产出：

$$
E[N] = 1 + \alpha_1 + \alpha_1\alpha_2 + \alpha_1\alpha_2\alpha_3
     = 1 + 1 + 0.9 + 0.63 = 3.53
$$

对比：Medusa 式均匀 $\alpha = 0.6$、$K=3$ 时 $E[N] = 1 + 0.6 + 0.36 + 0.216 = 2.176$。EAGLE 多出约 60% 的每轮产出，主要来自"第一个草稿免费 + 后续衰减慢"。

若一次目标前向的墙钟为 $T_p$、3 步轻量草稿共约 $0.4\,T_p$：

$$
\text{speedup} \approx \frac{3.53}{1.4} \approx 2.5\text{x}
$$

（示意值；论文 70B 实测 2.7–3.5x，见第 9 节。）

---

## 9. 实验结果

论文（EAGLE-1）与 EAGLE-2 的已核实数据：

| 实验 | 结果 |
|---|---|
| 草稿方式消融（Vicuna-7B，MT-bench，温度 0） | token 层 1.5x → 特征层 1.9x → +shifted-token 2.8x |
| 草稿接受率（top-1 链式） | EAGLE ≈ 0.8；Medusa ≈ 0.6；Lookahead 更低 |
| **LLaMA2-Chat 70B** | **延迟加速 2.7x–3.5x，吞吐翻倍，分布保持** |
| 覆盖任务 | 对话（MT-bench）、代码（HumanEval）、数学（GSM8K）、指令（Alpaca） |
| EAGLE-2（动态树） | 加速 3.05x–4.26x，比 EAGLE-1 快 20%–40% |
| EAGLE-2 全任务范围 | 2.5x–5x（六个任务、三个模型系列） |
| EAGLE-2 每轮接受长度 | ≈ 4–5.5 token，约为标准推测解码和 Medusa 的 2 倍 |
| 无损性 | EAGLE 与 EAGLE-2 都用严格拒绝采样，贪心与非贪心均保持分布 |

值得注意：EAGLE-2 在代码生成任务上最高 5x（代码模板规律性强，特征预测准）；EAGLE-2 在 MT-bench 上约比 Medusa 快 2x、比 Lookahead 快 2.3x。

---

## 10. EAGLE-2：上下文感知的动态草稿树

### 10.1 静态树的隐含假设与反例

EAGLE-1、Medusa 都用**固定形状**的树：每层加 $k$ 个候选，隐含假设"接受率只取决于位置"。EAGLE-2 指出这个假设不成立：

```
查询 "10+2="：下一个 token 几乎必然是 "1"，一个候选就够
查询 "10+2"（还没打完）：下一个 token 很难猜，需要多个候选
```

论文实验（Vicuna-7B，Alpaca）：接受率既随位置变化（左上 P1 最高、右下 P6 最低），**也随上下文大幅波动**（同一位置不同查询差异显著）。

### 10.2 置信度 ≈ 接受率（校准性）

动态调树需要"不跑目标模型就能估计接受率"。论文发现 EAGLE 草稿模型**校准良好**：置信度 $c$（草稿分布给出的概率）与接受率强正相关——

```
置信度 c < 0.05 → 平均接受率 ≈ 0.04
置信度 c > 0.95 → 平均接受率 ≈ 0.98
```

于是可以用 $c$ 近似接受率，零额外开销。

### 10.3 扩展 + 重排

定义节点 $t_i$ 的**全局接受概率**（路径上所有节点接受率的乘积，用置信度近似）：

$$
\mathrm{value}(t_i) = \prod_{t_j \in \mathrm{Path}(\mathrm{root}, t_i)} c_j
$$

（一个节点要被接受，它的所有祖先都得先被接受——所以是连乘。）

```
扩展阶段：从当前最后一层里选 value 最高的 top-k 节点，喂给草稿模型展开下一层
         （避免整层指数级展开，控制草稿前向开销）
重排阶段：在所有节点里选 value 最高的 top-m，展平成 1D 序列送去验证
         （浅层未扩展的高价值节点不会被深层的低价值节点挤掉）
```

两个性质保证正确性：

1. **$\mathrm{value}(\text{child}) \le \mathrm{value}(\text{parent})$**：因为置信度 $c \le 1$，连乘沿路径单调不增；
2. 由于父节点 value 不小于子节点，重排后选出的 top-$m$ 集合**仍是连通的树**（若子节点入选，父节点必然更早入选；平局优先选浅层）。

EAGLE-2 不改草稿模型、不改验证，**零额外训练**，严格无损。

---

## 11. 与 02/03 章对照

| 维度 | 02 原始推测解码 | 03 Medusa | 04 EAGLE |
|---|---|---|---|
| 草稿形式 | 独立小模型，token 级自回归 | 多头 MLP，平行猜未来 token | 轻量 decoder，**特征级自回归** |
| 草稿对主干的依赖 | 无（完全独立） | 用主干 $h_t$ | 用主干特征 + 共享 Embedding/LM Head |
| 第一草稿 token | 需按拒绝采样判定 | 目标贪心直接收（典型验收） | **与目标分布相同，恒接受** |
| 典型接受率 | 视草稿模型 | ≈ 0.6 | ≈ 0.8 |
| 每轮草稿成本 | $K$ 步小模型前向 | head 前向（近免费） | $K$ 步轻量 decoder 前向 |
| 树 | 链（可推广树） | 静态树 | 静态树（EAGLE-1）/ 动态树（EAGLE-2） |
| 无损性 | 严格 | 严格（拒绝采样）/ 近似（典型验收） | **严格（两个版本都是）** |
| 训练 | 无 | 训 $K$ 个头 | 训轻量 decoder（1–2 天） |
| 70B 实测 | — | — | 2.7–3.5x |

演进主线：02 换"谁来猜"（外部模型）→ 03 换"怎么猜"（多头 + 树）→ 04 换"猜什么"（**token 换成特征**）。猜的东西越贴近主干的真实计算，接受率越高。

---

## 12. 实现细节与坑

1. **特征层必须选对**：取 LM head 之前的第二顶层隐藏状态。取错层（如顶层 logits）会破坏共享 LM head 的平滑性假设，$q$ 与 $p$ 失去对应关系。
2. **噪声增强不能省**：不加 $\mathcal{U}(-0.1, 0.1)$ 噪声，预测特征误差沿链累积会让草稿快速发散。
3. **必须保存接受路径的真实特征**：下一轮草稿的起点是验证阶段算出的真实特征；丢弃它们就得重跑目标前向，白付一次验证。
4. **共享 LM head 意味着同词表**：EAGLE 天然满足（Embedding 都是目标的）；若换词表则整个机制失效。
5. **树验证细节**：与 03 章相同（掩码只看祖先、RoPE 位置按层共享）；拒绝采样逐节点递归执行，保证无损。
6. **第一草稿免费的性质只对 EAGLE 成立**：它来自"同一特征 + 同一 LM head"；Medusa 的 head 有自己的投影，不享受此性质。
7. **EAGLE-2 的动态树别和静态树混淆**：静态树靠位置定形状；动态树按置信度连乘定形状，零额外训练。
8. **量化与批处理**：草稿 decoder 很小，可与 W4A16 等量化叠加；vLLM 已集成 EAGLE（含 EAGLE-2 风格的动态树）。

---

## 13. 本章小结

1. **两个观察**：特征层自回归比 token 层容易（1.9x vs 1.5x）；但特征也有分支不确定性（$f_I \to f_{am}$ 或 $f_{always}$）。
2. **解法**：shifted-token 输入——把"刚采样出的 token"告诉模型，让回归目标唯一（加速从 1.9x 到 2.8x）。
3. **架构**：共享目标 Embedding 与 LM Head + 可训练轻量 decoder；$q = \mathrm{LM\_Head}(\hat{f})$。
4. **接受率高的三个机制**：第一草稿恒接受（$q_1 \equiv p_1$）、光滑传导（$\mathrm{TV} \le L\|\hat{f}-f\|$）、目标唯一。
5. **数据**：LLaMA2-Chat 70B 2.7–3.5x、吞吐翻倍；EAGLE-2 动态树 3.05–4.26x。
6. **无损性**：EAGLE 两个版本都用严格拒绝采样，贪心与非贪心都保持目标分布。

> 一句话记忆：**"猜 token 是在赌单词，猜特征是在画轨迹——先把已经抽到的单词告诉模型（shifted token），再让它画出这个单词的轨迹（特征），用同一把尺子（共享 LM head）量出下一个词。"**

---

## 14. 习题与解答

### 题 1（推导）：第一草稿恒接受

严格证明：EAGLE 每轮第一个草稿 token 的接受概率为 1，且输出分布等于目标分布 $p_{j+1}$。

<details>
<summary>题 1 解答</summary>

第一个草稿位置的输入是目标真实特征 $f_j$，草稿分布 $q_{j+1} = \mathrm{LM\_Head}(f_j)$；目标验证时算的也是 $p_{j+1} = \mathrm{LM\_Head}(f_j)$。故 $q \equiv p$，接受概率 $\sum_x \min(p(x),q(x)) = 1$。输出 token 按 $q = p$ 采样，边际分布即 $p_{j+1}$，无损。
</details>

### 题 2（计算）：接受率剖面

EAGLE 链式草稿 $K=4$，接受率剖面 $\alpha = (1.0, 0.9, 0.7, 0.6)$；Medusa 链式 $\alpha = 0.6$ 均匀。分别算 $E[N]$，并解释差距来源。

<details>
<summary>题 2 解答</summary>

EAGLE：$E[N] = 1 + 1 + 0.9 + 0.63 + 0.378 = 3.908$。Medusa：$1 + 0.6 + 0.36 + 0.216 + 0.1296 = 2.306$。差距主要来自两项：第一项免费（+0.4）和早期接受率高（0.9/0.7 vs 0.6）带来的乘积放大。
</details>

### 题 3（推导）：TV 与特征误差

设 $p = \mathrm{softmax}(G(f))$、$q = \mathrm{softmax}(G(\hat{f}))$，$G$ 是 $L$-Lipschitz。说明为什么 $\alpha \ge 1 - L\|\hat{f} - f\|$ 形式的界成立，并解释"回归误差连续传导"的含义。

<details>
<summary>题 3 解答要点</summary>

$\alpha = 1 - \mathrm{TV}(p,q)$（01 章）。softmax∘G 复合映射在分布空间的 Lipschitz 常数有限，故 $\mathrm{TV}(p,q) \le L\|\hat{f}-f\|$，接受率随特征误差线性下降。含义：只要回归误差小（连续量），分布就接近；而 token 分类选错类时 $\mathrm{TV}=1$，没有"半对"的中间状态。
</details>

### 题 4（思考）：分支不确定性

画出示意图：前缀 "I" 之后 "am" 与 "always" 两个分支。解释为什么没有 shifted-token 时草稿模型只能"平均"两个目标，以及这对 Medusa 接受率 ~0.6 的贡献。

<details>
<summary>题 4 解答要点</summary>

$f_I$ 之后有两条合法路径：$(t_{am}, f_{am})$ 与 $(t_{always}, f_{always})$。若输入只有 $f_I$，回归目标在 $f_{am}$ 与 $f_{always}$ 之间摇摆，模型学到的输出是两者的混合特征，映射到分布后与两个真实分支都不一致，接受率被压低。shifted-token 输入把"路径选择"交给采样（不确定性被 token 显式承担），回归目标唯一。
</details>

### 题 5（设计）：EAGLE-2 的 value 与连通性

证明 $\mathrm{value}(\text{child}) \le \mathrm{value}(\text{parent})$，并说明为什么重排后选出的 top-$m$ 节点仍是连通树。

<details>
<summary>题 5 解答</summary>

$\mathrm{value}(t_i) = \prod_{t_j \in \mathrm{Path}(\mathrm{root},t_i)} c_j$。子节点的路径 = 父节点路径 + 子节点自身，多乘一个 $c \in [0,1]$，故 value 不增。若某节点入选 top-$m$，其父节点 value 更大（或相等），必然也在 top-$m$ 里（平局优先浅层）——所以选出集合的每个节点都有祖先入选，构成连通树。这保证展平后可以用树掩码验证。
</details>

### 题 6（编程）：toy 特征级草稿

实现一个 toy 模拟：给定目标特征映射 $f_j$、真实特征序列生成器与一个"误差随步数增长"的预测器，跑 5 万轮 EAGLE 式草稿-验证，统计每轮 token 数与 $E[N] = 1 + \sum_{i=1}^{K}\prod_{j\le i}\alpha_j$ 的偏差；再对比"无 shifted-token"版本（回归目标混合）的接受率。

<details>
<summary>题 6 解答要点</summary>

① 第一草稿用 $p$ 本身采样（恒接受）；② 后续草稿用带噪声的特征映射 $f \to \hat{f} = f + \varepsilon_i$ 生成分布，接受率 $\alpha_i$ 由 $\varepsilon_i$ 决定；③ 统计均值应接近公式；④ 无 shifted-token 版本把两个分支特征平均，$\alpha$ 明显下降——复现论文 Fig. 4 的趋势。
</details>

---

## 15. 延伸阅读

1. [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty（arXiv:2401.15077）](https://arxiv.org/abs/2401.15077)：本章全部内容出处（观察与解法 §1、架构 §2、训练 §3、实验 §4）。
2. [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees（arXiv:2406.16858）](https://arxiv.org/abs/2406.16858)：动态树（扩展/重排、置信度校准）出处。
3. [SpecInfer（arXiv:2305.09781）](https://arxiv.org/abs/2305.09781)：EAGLE 树验证所用的递归拒绝采样框架。
4. [NVIDIA 技术博客：An Introduction to Speculative Decoding](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)：工程视角的对照阅读。
5. 上一篇：[03 Medusa：多头解码](./LLM推测解码精读笔记-03-Medusa-多头解码.md)；下一篇：**05 n-gram / 检索式与无模型路线**——Lookahead Decoding、REST、Prompt Lookup，回答"不训练任何东西能不能猜"。
