# RST data pipeline

## Sources and license

COGA consumes the public RST task and trajectory datasets linked in the root README. Both are
published under CC BY 4.0. The repository stores only source code and configuration; users
download data directly from the publisher.

Expected public inventory:

| Entity | Count |
|---|---:|
| Tasks | 37,484 |
| Trajectories | 327,189 |
| Reward 1 | 62,511 |
| Reward 0 | 183,043 |
| Censored/null | 81,635 |

Reward 0 is retained as a real failure. Null reward is not silently converted to failure and
is excluded from success/failure pairing.

## Reproducible stages

```bash
coga-data download --config configs/qwen3_8b_4090.json
coga-data verify   --config configs/qwen3_8b_4090.json
coga-data extract  --config configs/qwen3_8b_4090.json
coga-data cohorts  --config configs/qwen3_8b_4090.json
```

`download` uses the Hugging Face Hub resumable downloader. `verify` reads the publisher's
`metadata/shard_manifest.jsonl` and checks size plus SHA-256 for every TAR. `extract` first
rejects absolute paths, parent traversal, links, and device members; a hash marker makes a
verified extraction resumable. Original archives are never overwritten.

## Exact-task matching

RST trajectories identify a task family but do not always expose an exact task ID. COGA uses
the same conservative `rst_prompt_norm_v1` contract as the original research pipeline:

1. extract the text between `Task Description:` and `Current terminal state:` from the first
   ATIF user message;
2. normalize BOM, CRLF, Unicode NFC, trailing line whitespace, and outer whitespace;
3. SHA-256 the normalized prompt with a version prefix;
4. retain prompts mapping to exactly one public task;
5. group by `(family_key, prompt_key, policy_key)`;
6. require at least four successes and four failures.

The frozen public projection is expected to produce 255 cohorts. The builder stops if the
count differs, which catches incomplete downloads and schema drift before GPU work begins.

## Outputs

- `data/processed/cohorts/population.sqlite`: cohort identity and trajectory references.
- `data/processed/cohorts/outcomes.sqlite`: binary outcomes for known-reward trajectories.
- `data/processed/cohorts/cohort_report.json`: counts and normalization versions.
- `runs/<run>/manifests/gradient_pairs.jsonl`: real plus seven equal-cardinality controls.

All stored trajectory paths are relative to the configured data root. Generated files are
ignored by Git.
