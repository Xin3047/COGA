# 方法与评测引用

- LESS: *Selecting Influential Data for Targeted Instruction Tuning*, arXiv:2402.04333.
- GIST: *Targeted Data Selection for Instruction Tuning via Coupled Optimization Geometry*, arXiv:2602.18584.
- *Influential Language Data Selection via Gradient Trajectory Pursuit*, arXiv:2410.16710.
- RST: *Recursive Task Synthesis*, arXiv:2608.05466.
- BFCL / Gorilla official repository: <https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard>.
- BFCL v3 multi-turn description: <https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html>.
- BFCL live leaderboard: <https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard>.
- Terminal-Bench: <https://www.tbench.ai/>.

Benchmark 选择依据：BFCL 是公开、可执行、无需 Docker task fleet 的 function-calling evaluator，官方实现已
支持 self-hosted Qwen3-8B 与 vLLM LoRA；公开榜单中 Qwen3-8B (FC) 与 xLAM-2-8B-fc-r 已有 8B 级可用
成绩。2026-08-30 按用户要求改为官方 `all_scoring` group，不再排除 agentic memory/web-search；可用 test
groups 与 categories 以官方
[TEST_CATEGORIES.md](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/TEST_CATEGORIES.md)
为准。

2026-08-28 查看 BFCL V4 官方榜单时，`Qwen3-8B (FC)` overall 为 `42.57`，
`xLAM-2-8b-fc-r (FC)` overall 为 `46.68`；这些只用于说明 8B 级模型在该公开评测上已有可比较基线，
不是本项目结果，也不会回填到简历指标。
