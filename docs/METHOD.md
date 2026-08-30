# COGA 方法说明

## 1. 要解决的问题

长程 Terminal Agent 的轨迹只有终局 success/failure，直接把整条成功轨迹都当成等价监督，会把任务难度、
长度、工具风格和 policy 习惯混进数据选择。COGA 不再给每个 turn 硬造 dense reward，而是问一个更容易
被模型自身回答的问题：同一道题、同一个 policy 下，成功与失败轨迹对当前 Qwen3-8B 参数提出的更新方向，
差在哪里？

## 2. 原子定义

对同一 `(family, prompt, policy)` cohort 的成功轨迹 `tau+` 与失败轨迹 `tau-`，只在各自最后一个
assistant target 上计算 causal loss：

```text
g+ = grad_phi L_target(tau+; theta, phi)
g- = grad_phi L_target(tau-; theta, phi)
d  = Sketch(g+ - g-)
```

`theta` 是冻结的 4-bit Qwen3-8B base，`phi` 是 q/k/v/o projection 上的 LoRA 参数。正式抽取前先用 128
个真实 success final-turn 做短 warm-up，避免只观察零初始化 LoRA 的退化切面；抽取时切换 `eval()`，关闭
dropout 的随机扰动。

Dense LoRA gradient 仍然很大，因此用固定 hash/sign 的 8,192 维 CountSketch：它不需要把全量 gradient
写盘，仍可近似保留 L2 与 cosine geometry。每个 pair 只落盘约几十 KB 的 embedding。

## 3. 跨折原型与 COGA score

以 `family + prompt` 做确定性 5-fold。对第 `k` 折样本，原型只由其余四折的 real contrast 构成：

```text
mu(-k) = normalize(mean(normalize(d_j))), j not in fold k and control=real
COGA_i = cosine(normalize(d_i), mu(-k))
```

这个 score 衡量某个同任务 success-minus-failure 更新方向，是否与其它任务上反复出现的“成功方向”一致。
它不是训练 loss、不是 reward，也不是把自己的 gradient 与自己比较；cross-fit 去掉了最直接的自相似偏置。

## 4. 七个证伪控制

- `reward_shuffle`：cohort 内打乱 outcome label。
- `turn_shuffle`：破坏轨迹时间顺序。
- `cross_cohort_pairing`：同 operator、不同 task 的 failure。
- `length_matched_swap`：跨 task 只匹配轨迹长度。
- `bag_of_actions`：保留 action 集合、打乱 action 顺序。
- `random_label`：完全随机分配正负角色。
- `self_pair_identity`：同一 success 与自身相减，应接近零。

全部控制都使用同一个由 real rows 构造的 cross-fitted prototype。主 gate 固定为：`real - strongest control`
的 10,000 次 paired cohort bootstrap 95% CI 下界大于 0。这样“方法成立”至少意味着顺序、同任务配对与
真实 outcome 同时有贡献，而不是长度或工具风格在冒充梯度信用。

## 5. 从 score 到训练数据

若 gate 通过，按 cross-fitted COGA score 选择 Top 50% cohort，并收集这些 cohort 内所有 reward=1 的真实
trajectory。每一个 assistant turn 都用当时可见的 `instruction + history + tool observation` 渲染；
solution、tests、verifier、reward、future observation 与 COGA score 都不进入 student input。

对照臂在相同 cohort 数量下做确定性 Rejection-Success 抽样。两个臂都按 cohort round-robin 消费数据，
最后一条 target 只开放剩余的 label positions，因此各自精确等于 1,000,000 target loss tokens。

训练固定为 Qwen3-8B 4-bit NF4 / BF16、LoRA r=16 alpha=32 dropout=0.05、q/k/v/o、max length 4096、
target-only loss、no packing、1 epoch、PagedAdamW8bit。两个 adapter 从同一 base 独立初始化。

## 6. 双 Benchmark

- Terminal-Bench 2.1：Harbor + Terminus-2 + Docker verifier，对本地完整 89-task export 逐一评测，测长程
  终端规划、状态操作和闭环完成率。
- BFCL V4 full scoring：官方 `bfcl-eval --test-category all_scoring`，覆盖 non-live、live、multi-turn 与
  agentic memory/web-search 全部计分类别；测 function schema grounding、argument construction、多轮 Tool
  Calling 和 agentic tool use。运行前需配置官方要求的 agentic backends/credentials。

Terminal-Bench 主统计量是 `COGA - Rejection` paired task difference 与 bootstrap CI；BFCL 保留官方
category/overall accuracy。两边都同时报告 base、rejection、COGA，避免把 base 波动误写成选样收益。

## 7. 这条故事真正新在哪里

不是“又训练了一个 LoRA”，而是从终局标量 outcome 到模型更新几何的完整 data-centric 闭环：

```text
Outcome Pairing -> Target-only Gradient -> Cross-fitted Geometry
-> Falsification Gate -> Exact-token Curation -> QLoRA -> Agent/Tool Eval
```

LESS 提供 gradient-influence data selection 的起点；GIST 提醒要看 coupled optimization geometry；本项目把
这套思想改造成 long-horizon terminal trajectories 上的同任务成败对比，并补上顺序、长度、跨任务与 identity
控制。最终 claim 以真实运行结果为准，当前代码状态为
`FULL_TRAIN_EVAL_CODE_READY_NOT_RUN`；没有正式训练或评测结果。
