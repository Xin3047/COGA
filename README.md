# COGA：当前方法的独立实现

本目录只包含 2026-08-28 之后的最新方法，不承载 ECHO、SAC-RST、Stage 5 replay、旧 Agentic Data
Recipe 或其它历史路线。历史文件保留在仓库原位置；这里是从真实 RST 轨迹到 Qwen3-8B 训练与双
Benchmark 评测的最短闭环。

正式实验状态：`FULL_TRAIN_EVAL_CODE_READY_NOT_RUN`。Claude 已完成的 255-cohort 数据投影、真实轨迹
解析、7 个控制 manifest 与 0.5B mechanical smoke 仍是已运行事实；本目录当前只暴露完整的 Qwen3-8B
gradient、双 1M-token QLoRA、89-task Terminal-Bench（Base/Rejection/COGA 共 267 trials）和 BFCL V4
`all_scoring` production 路径，不再提供 toy/orchestration smoke CLI。上述 production 组件均未运行，
结果字段保持空值。

## 一条链路

```text
255 个 exact-task mixed-outcome cohorts
  -> 同 family / prompt / policy 的 success-failure 配对
  -> Qwen3-8B 4-bit QLoRA target-only per-example gradients
  -> success-minus-failure CountSketch gradient contrast
  -> family/prompt 5-fold cross-fitted contrast prototype
  -> real vs C1-C7 falsification controls + paired bootstrap gate
  -> Top-50% cohort success trajectories + exact 1M target-token SFT
  -> Rejection-Success / COGA-Selected 两个独立 QLoRA adapters
  -> Terminal-Bench 2.1 全部 89 tasks + BFCL V4 all_scoring
```

## 目录

- `configs/qwen3_8b_4090.json`：24GB RTX 4090 的唯一主配置和空结果位。
- `src/coga/`：路径、真实数据解析、控制构造、Qwen3 target-only loss 公共实现。
- `scripts/01...07`：manifest、梯度、SFT、训练、Terminal-Bench、BFCL、paired bootstrap。
- `docs/METHOD.md`：公式、设计选择和面试讲法。
- `docs/RESUME_STORY.md`：可直接放进简历的阿酥式项目经历，未跑指标留空。
- `docs/RUNBOOK.md`：工作站顺序命令与产物表。
- `docs/REFERENCES.md`：LESS、GIST、BFCL、RST 与相关开源入口。

## 关键工程约束

- Qwen3-8B 固定 revision `b968826...18`，4-bit NF4、BF16 compute、LoRA r=16。
- 单样本只计算当前 assistant target 的 causal loss；历史 token 全部 mask 为 `-100`。
- 梯度只取 LoRA 参数，并用 8,192 维 CountSketch 保存几何关系，避免落盘数 GB dense gradients。
- 同一 real prototype 给全部负控制打分；self-pair 应接近零。
- 方法 gate 通过前默认拒绝物化 COGA 训练臂。
- 两臂各消费精确 1,000,000 target loss tokens，从同一 base 独立初始化。
- Benchmark 结果不回流到选样或训练。
