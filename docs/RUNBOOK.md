# RTX 4090 工作站运行顺序

当前只提供冻结的全量 production 命令，状态为 `FULL_TRAIN_EVAL_CODE_READY_NOT_RUN`。本轮没有执行本页
任何命令。默认从仓库根目录运行，所有产物写入 `coga/runs/coga_qwen3_8b_4090_v1/`。

## 0. 全量执行边界

- 配置必须包含 `execution_scope=FULL_PRODUCTION_ONLY`；活动脚本不提供 toy、dry-run 或
  `--smoke-test` 分支。
- “全量训练”指两个 arm 各完整消费自身 Parquet 一次，各精确 1,000,000 target loss tokens；两个进程
  都从同一原始 Qwen3-8B 独立初始化。
- “全量 Terminal-Bench”指 Base、Rejection、COGA 依次覆盖本地完整 export 的全部 89 tasks，共 267
  model-task trials；代码同时核对 89/89 `task.toml`、frozen task inventory 与 dataset tree hash，少任一
  task 都拒绝生成正式 paired summary。
- “全量 BFCL”指三模型依次运行官方 V4 `all_scoring` group，包括 non-live、live、multi-turn 与 agentic
  memory/web-search 全部计分类别；generate/evaluate 返回成功且 official overall row/accuracy 可解析才记为
  `COMPLETE`。正式运行前必须配置 BFCL 所需的 agentic backends/credentials。
- 2026-08-30 更早的轻量 smoke 报告仅为本地历史文件，CLI 已移除，不属于以下流程。

## 1. 环境与路径

```bash
python3.11 -m venv envs/coga-4090
envs/coga-4090/bin/python -m pip install -U pip
envs/coga-4090/bin/python -m pip install -e './coga[eval]'
```

确认以下本地路径与 `configs/qwen3_8b_4090.json` 一致：Qwen3-8B 模型、RST raw tree、两个 SQLite、
Harbor/vLLM/BFCL executable、Terminal-Bench 89-task inventory 与 dataset export，以及 BFCL V4 agentic
memory/web-search backends 和 credentials。

## 2. 数据与梯度 gate

```bash
envs/coga-4090/bin/python coga/scripts/01_build_manifests.py --config coga/configs/qwen3_8b_4090.json
envs/coga-4090/bin/python coga/scripts/02_score_qwen3_gradients.py --config coga/configs/qwen3_8b_4090.json
```

梯度脚本按 row 保存 `gradients/cache/*.npy + *.json`，中断后用同一命令续跑。先查看
`gradients/gradient_report.json`；默认只有 `method_gate=PASS` 才允许构造训练臂。

## 3. 等 token 数据与两臂训练

```bash
envs/coga-4090/bin/python coga/scripts/03_materialize_sft.py --config coga/configs/qwen3_8b_4090.json
envs/coga-4090/bin/python coga/scripts/04_train_qwen3_qlora.py --config coga/configs/qwen3_8b_4090.json --arm rejection_success
envs/coga-4090/bin/python coga/scripts/04_train_qwen3_qlora.py --config coga/configs/qwen3_8b_4090.json --arm coga_selected
```

两个 arm 必须各自从原始 Qwen3-8B 初始化，不允许让 COGA adapter 接续 Rejection adapter。

## 4. 双评测

```bash
envs/coga-4090/bin/python coga/scripts/05_eval_terminal_bench.py --config coga/configs/qwen3_8b_4090.json
envs/coga-4090/bin/python coga/scripts/07_summarize_terminal_bench.py --config coga/configs/qwen3_8b_4090.json
envs/coga-4090/bin/python coga/scripts/06_eval_bfcl.py --config coga/configs/qwen3_8b_4090.json
```

Terminal-Bench 结果入口：`evaluation/terminal_bench/terminal_bench_report.json`。
BFCL 官方 score 入口：`evaluation/bfcl/<arm>/score/`，本项目只编排官方 evaluator，不重写它的 grader。

## 5. 回填简历

只回填以下真实输出：gradient gate 的主差与 CI、selected cohorts/rows/tokens、两臂训练时长与峰值显存、
Terminal-Bench 三模型成功率与 paired CI、BFCL 三模型 official accuracy。不要用 warm-up loss、
CountSketch norm 或论文数字替代 downstream 指标。
