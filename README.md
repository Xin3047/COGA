# COGA

Contrastive Outcome Gradient Alignment (COGA) is a reproducible data-curation and QLoRA
pipeline for long-horizon terminal agents. It turns same-task success/failure trajectory
pairs into target-only gradient contrasts, validates the signal against seven controls,
and compares a COGA-selected SFT arm with a rejection-sampling baseline.

This repository is a reproduction guide and does not publish experiment results. It provides
the code, frozen configuration, data provenance, and commands needed to reproduce the full
pipeline independently.

## Pipeline

```text
RST tasks + trajectories (Hugging Face, CC BY 4.0)
  -> resumable download -> SHA-256 verification -> safe extraction
  -> exact prompt/policy matching -> mixed-outcome cohorts
  -> real pairs + seven falsification controls
  -> Qwen3-8B target-only LoRA gradients -> 8,192-d CountSketch
  -> family/prompt cross-fit -> paired bootstrap gate
  -> COGA-selected vs rejection-success, exactly 1M loss tokens each
  -> independent QLoRA training -> Terminal-Bench 2.1 + BFCL V4 Multi-Turn
```

The source datasets accompany
[Recursive Synthesis for Long-Horizon Terminal Tasks](https://arxiv.org/abs/2608.05466):

- [Recursive-Task-Synthesis](https://huggingface.co/datasets/Zhongzhi1228/Recursive-Task-Synthesis):
  37,484 validated command-line tasks.
- [Recursive-Task-Synthesis-Trajectories](https://huggingface.co/datasets/Zhongzhi1228/Recursive-Task-Synthesis-Trajectories):
  327,189 completed trajectories.

The datasets are not redistributed here. They remain under the publisher's CC BY 4.0
license; the COGA source code is Apache-2.0.

## Repository layout

```text
configs/                 frozen experiment configuration
src/coga/                reusable data, cohort, rendering, and training utilities
scripts/data/            RST preparation, pair manifests, SFT materialization
scripts/scoring/         contrastive gradient scorer
scripts/training/        QLoRA training entrypoints
scripts/evaluation/      Terminal-Bench/BFCL orchestration and reporting
scripts/setup/           environment and pinned-model setup
tests/                   CPU-only reproducibility and safety tests
docs/                    method, architecture, data, and reproduction notes
```

Generated data, models, checkpoints, logs, and benchmark outputs are ignored by Git.

## Requirements

- Linux, Python 3.11, Git, and Git LFS.
- About 60 GB free disk for the complete RST download plus extraction.
- One CUDA GPU with approximately 24 GB VRAM for the frozen Qwen3-8B NF4 pipeline.
- Docker, Harbor/Terminus-2, and a local Terminal-Bench 2.1 export for agent evaluation.
- The official BFCL evaluator for the V4 Multi-Turn test group.

The data preparation and unit tests are CPU-only. Training/evaluation dependencies are
kept in optional extras so contributors can work on data code without installing CUDA.

## Quick start

```bash
git clone https://github.com/Xin3047/COGA.git
cd COGA
bash scripts/setup/install.sh
source .venv/bin/activate
bash scripts/setup/download_model.sh
```

Prepare all public RST inputs. Downloads are resumable, every TAR is verified against the
publisher's manifest, and extraction markers make reruns idempotent:

```bash
coga-data all --config configs/qwen3_8b_4090.json
```

Run the production data-selection and training chain:

```bash
python scripts/data/build_manifests.py --config configs/qwen3_8b_4090.json
python scripts/scoring/score_gradients.py --config configs/qwen3_8b_4090.json
python scripts/data/materialize_sft.py --config configs/qwen3_8b_4090.json
bash scripts/training/train.sh configs/qwen3_8b_4090.json
```

Or use the single launcher. Evaluation is deliberately opt-in because it requires external
Docker images, services, and credentials:

```bash
bash scripts/run_pipeline.sh
COGA_RUN_EVALUATION=1 bash scripts/run_pipeline.sh
```

Commands are resumable where scientifically safe. Gradient rows are cached under
`runs/<run>/gradients/cache/`; the manifest/config hash contract prevents stale cache reuse.
Training refuses to overwrite a non-empty output directory.

## Evaluation setup

Install and export the official external benchmarks using their upstream documentation:

- [Terminal-Bench / Harbor](https://www.tbench.ai/)
- [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)

Place the complete 89-task Terminal-Bench 2.1 export at
`data/benchmarks/terminal-bench-2-1`, then freeze it before observing model results:

```bash
python scripts/evaluation/freeze_terminal_bench.py --config configs/qwen3_8b_4090.json
bash scripts/evaluation/evaluate.sh configs/qwen3_8b_4090.json
```

The evaluator rejects missing tasks, changed task trees, partial paired result sets, and BFCL
runs without an official parsable overall score.

## Tests and Docker

```bash
python -m pip install -e ".[dev]"
pytest -q
ruff check src tests scripts
docker compose run --rm test
```

The Docker image is intentionally a lightweight CPU test image. GPU training uses the host
CUDA stack; no model weights or datasets are baked into an image.

## Reproducibility contract

The production configuration freezes Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`, NF4/BF16 QLoRA, LoRA rank 16 on
q/k/v/o projections, target-only causal loss, 8,192-dimensional CountSketch, five cross-fit
folds, and equal 1,000,000 loss-token training budgets. Benchmark outcomes never feed back
into selection.

See [data pipeline](docs/data-pipeline.md), [method](docs/METHOD.md),
[architecture](docs/architecture.md), and [full reproduction guide](docs/reproduction.md).

## Contributing

Issues, falsification controls, data audits, and reproducibility reports are especially
welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and use GitHub Issues before a large
change. Security concerns should follow [SECURITY.md](SECURITY.md).

## Citation and licenses

Use [CITATION.cff](CITATION.cff) for this implementation and cite the RST paper/datasets when
using their data. Code is licensed under [Apache-2.0](LICENSE); upstream data and benchmark
assets retain their own licenses.
