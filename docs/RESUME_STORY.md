# 简历项目经历｜COGA：长程 Terminal Agent 梯度对比数据引擎

## 可直接放简历的版本（指标待工作站运行后回填）

**COGA 长程 Agent Data Engine｜模块级 Owner｜从 0→1 打通 Gradient Curation → QLoRA → Agent Eval**

- **背景：** 针对 32.7 万条 RST 长程 Terminal Agent trajectories 只有 terminal outcome、缺少 step-level
  credit，且历史 Process/Style-Aware 轨迹筛选在固定 1M loss-token 的 Qwen3-8B 对照中未建立增益，重构
  数据策展粒度：从启发式动作风格转向同任务 success/failure 的模型更新几何。
- **指标与效果：** 完成 255 个 exact-task mixed-outcome cohorts 的真实配对、ATIF trajectory parsing、
  student-visible context rendering 与 C1-C7 falsification manifests；在单卡 RTX 4090 上完成
  Qwen3-8B COGA gradient gate `【待填：real-control 差 / CI】`，构造双臂各 1,000,000 target loss tokens，
  Terminal-Bench 2.1 全 89 tasks `【待填：Base / Rejection / COGA】`，BFCL V4 all-scoring
  `【待填：Base / Rejection / COGA】`，峰值显存 `【待填】GB`。
- **我的职责：** 0→1 设计并实现
  `Exact-Task Outcome Pairing -> Qwen3-8B 4-bit QLoRA Target-only Gradient -> 8K CountSketch ->
  Family/Prompt Cross-Fit Prototype -> 7-way Negative Control -> Paired Bootstrap Gate ->
  Exact-Token SFT -> Harbor/Terminus-2 + BFCL`；将 dense LoRA gradient 压缩为可恢复的低存储 embedding，
  以 self-pair identity、turn/reward shuffle、cross-cohort 和 length-matched swap 拆解顺序、标签、任务与长度混杂。
- **技术关键词：** Qwen3-8B、QLoRA、NF4、BF16、LoRA q/k/v/o、Target-only Causal Loss、CountSketch、
  Cross-Fitting、Gradient Influence、ATIF、RST、Hard Negative、Paired Bootstrap、Harbor、Terminus-2、
  Terminal-Bench 2.1（89 tasks）、BFCL V4 all-scoring、vLLM、Tool Calling。

## 一句话版本

从 0→1 构建长程 Terminal Agent 的对比梯度数据引擎 COGA，在 255 个同任务成败 cohort 上将
Qwen3-8B LoRA gradient geometry、7 类证伪控制、精确 1M-token QLoRA 与 Terminal-Bench/BFCL 双评测串成
端到端闭环，最终相对 Rejection-Success 提升 `【待填】`。

## 面试开场（约 45 秒）

“这个项目最开始想解决长程 Agent 的 credit assignment，但数据只有终局 reward。我们先试过动作风格、
轨迹长度和若干局部启发式，固定 token 训练后没有拿到稳定增益。我后来把问题换成模型更新几何：同一道题、
同一个 policy，成功轨迹和失败轨迹对 Qwen3-8B LoRA 参数分别产生什么 gradient，它们的差向量是否在不同
任务之间形成稳定方向。我把这个方法叫 COGA。工程上不是直接存几千万维 gradient，而是做 8K
CountSketch；统计上用 family/prompt cross-fit 防止 self-alignment，再用七类 negative controls 证明不是
长度、顺序或工具风格。gate 通过以后才构造等 1M target-token 的 SFT 两臂，最后同时跑
Terminal-Bench 全 89 tasks 和 BFCL V4 all-scoring，一个看长程终端闭环，一个看完整 Function Calling 与
agentic tool use。”
