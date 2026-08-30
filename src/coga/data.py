from __future__ import annotations

import dataclasses
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .common import PROJECT_ROOT, resolve_project_path, stable_int


FORBIDDEN_TRAINING_TERMS = (
    "solution/solve.sh",
    "rewrite_target.json",
    "tests/test_state.py",
    "/tests/",
    "verifier/reward.txt",
    "verifier/ctrf.json",
    "teacher future",
    "successful peer suffix",
    "style score",
    "utility score",
    "selection tier",
)


@dataclasses.dataclass(frozen=True)
class Cohort:
    cohort_id: str
    family: str
    prompt: str
    policy: str
    task_id: str
    operator: str


@dataclasses.dataclass(frozen=True)
class TrajectoryRef:
    trajectory_id: str
    cohort_id: str
    relpath: str
    expected_steps: int


@dataclasses.dataclass(frozen=True)
class Turn:
    turn_index: int
    action: str
    observation: str
    commands: tuple[str, ...]
    task_complete: bool


@dataclasses.dataclass(frozen=True)
class PairSpec:
    control: str
    cohort: Cohort
    success: TrajectoryRef
    failure: TrajectoryRef
    success_transform: str = "identity"
    failure_transform: str = "identity"

    def to_json(self) -> dict[str, Any]:
        return {
            "control": self.control,
            "cohort_id": self.cohort.cohort_id,
            "family": self.cohort.family,
            "prompt": self.cohort.prompt,
            "policy": self.cohort.policy,
            "task_id": self.cohort.task_id,
            "operator": self.cohort.operator,
            "success_trajectory_id": self.success.trajectory_id,
            "success_relpath": self.success.relpath,
            "failure_trajectory_id": self.failure.trajectory_id,
            "failure_relpath": self.failure.relpath,
            "success_transform": self.success_transform,
            "failure_transform": self.failure_transform,
        }


def readonly_db(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)


def load_population(config: Mapping[str, Any]) -> tuple[dict[str, Cohort], dict[str, list[TrajectoryRef]]]:
    population_path = resolve_project_path(config["paths"]["population_db"])
    database = readonly_db(population_path)
    cohorts = {
        row[0]: Cohort(*row)
        for row in database.execute(
            "SELECT cohort_id, family, prompt, policy, task_id, operator FROM cohorts ORDER BY cohort_id"
        )
    }
    members: dict[str, list[TrajectoryRef]] = defaultdict(list)
    for row in database.execute(
        "SELECT trajectory_id, cohort_id, relpath, expected_steps FROM trajectories ORDER BY trajectory_id"
    ):
        ref = TrajectoryRef(row[0], row[1], row[2], int(row[3] or 0))
        members[ref.cohort_id].append(ref)
    database.close()
    return cohorts, dict(members)


def load_rewards(config: Mapping[str, Any]) -> dict[str, int]:
    outcomes_path = resolve_project_path(config["paths"]["outcomes_db"])
    database = readonly_db(outcomes_path)
    rewards = {
        row[0]: int(row[1])
        for row in database.execute("SELECT trajectory_id, reward FROM outcomes WHERE reward IN (0, 1)")
    }
    database.close()
    return rewards


def _real_pairs(
    cohorts: Mapping[str, Cohort],
    members: Mapping[str, Sequence[TrajectoryRef]],
    rewards: Mapping[str, int],
) -> list[PairSpec]:
    pairs: list[PairSpec] = []
    for cohort_id in sorted(cohorts):
        group = members.get(cohort_id, ())
        successes = sorted((ref for ref in group if rewards.get(ref.trajectory_id) == 1), key=lambda x: x.trajectory_id)
        failures = sorted((ref for ref in group if rewards.get(ref.trajectory_id) == 0), key=lambda x: x.trajectory_id)
        if successes and failures:
            pairs.append(PairSpec("real", cohorts[cohort_id], successes[0], failures[0]))
    return pairs


def build_pair_specs(config: Mapping[str, Any]) -> list[PairSpec]:
    """Build the real same-task contrast and seven equal-cardinality controls.

    Every control keeps one row per real cohort so distribution comparisons are
    paired and do not confound the method with a different sample count.
    """
    seed = int(config["seed"])
    cohorts, members = load_population(config)
    rewards = load_rewards(config)
    real = _real_pairs(cohorts, members, rewards)
    rows: list[PairSpec] = list(real)

    by_operator_failure: dict[str, list[tuple[str, TrajectoryRef]]] = defaultdict(list)
    all_failures: list[tuple[str, TrajectoryRef]] = []
    for cohort_id, group in members.items():
        for ref in group:
            if rewards.get(ref.trajectory_id) == 0:
                by_operator_failure[cohorts[cohort_id].operator].append((cohort_id, ref))
                all_failures.append((cohort_id, ref))

    for pair in real:
        group = sorted(
            (
                ref for ref in members[pair.cohort.cohort_id]
                if rewards.get(ref.trajectory_id) in (0, 1)
            ),
            key=lambda x: x.trajectory_id,
        )

        shuffled_labels = [rewards.get(ref.trajectory_id, 0) for ref in group]
        random.Random(stable_int("reward_shuffle", seed, pair.cohort.cohort_id)).shuffle(shuffled_labels)
        fake_success = [ref for ref, label in zip(group, shuffled_labels) if label == 1]
        fake_failure = [ref for ref, label in zip(group, shuffled_labels) if label == 0]
        if fake_success and fake_failure:
            rows.append(PairSpec("reward_shuffle", pair.cohort, fake_success[0], fake_failure[0]))

        rows.append(dataclasses.replace(
            pair,
            control="turn_shuffle",
            success_transform="turn_shuffle",
            failure_transform="turn_shuffle",
        ))

        operator_candidates = sorted(
            (item for item in by_operator_failure[pair.cohort.operator] if item[0] != pair.cohort.cohort_id),
            key=lambda item: (item[0], item[1].trajectory_id),
        )
        if operator_candidates:
            chosen = operator_candidates[
                stable_int("cross_cohort", seed, pair.cohort.cohort_id) % len(operator_candidates)
            ][1]
            rows.append(dataclasses.replace(pair, control="cross_cohort_pairing", failure=chosen))

        length_candidates = sorted(
            (item for item in all_failures if item[0] != pair.cohort.cohort_id),
            key=lambda item: (
                abs(item[1].expected_steps - pair.success.expected_steps),
                stable_int("length_tie", seed, pair.cohort.cohort_id, item[1].trajectory_id),
            ),
        )
        if length_candidates:
            rows.append(dataclasses.replace(pair, control="length_matched_swap", failure=length_candidates[0][1]))

        rows.append(dataclasses.replace(
            pair,
            control="bag_of_actions",
            success_transform="bag_of_actions",
            failure_transform="bag_of_actions",
        ))

        random_group = list(group)
        random.Random(stable_int("random_label", seed, pair.cohort.cohort_id)).shuffle(random_group)
        split = max(1, len(random_group) // 2)
        random_success, random_failure = random_group[:split], random_group[split:]
        rows.append(PairSpec("random_label", pair.cohort, random_success[0], random_failure[0]))

        rows.append(dataclasses.replace(pair, control="self_pair_identity", failure=pair.success))

    return sorted(rows, key=lambda row: (row.control, row.cohort.cohort_id))


def conservative_json(message: Any) -> dict[str, Any] | None:
    if not isinstance(message, str):
        return None
    try:
        value = json.loads(message)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start, end = message.find("{"), message.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(message[start : end + 1])
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None


def command_values(message: Any) -> tuple[list[str], bool]:
    value = conservative_json(message)
    if value is None:
        return [], False
    commands: list[str] = []
    for row in value.get("commands", []):
        if isinstance(row, Mapping) and isinstance(row.get("keystrokes"), str):
            commands.append(row["keystrokes"])
    return commands, value.get("task_complete") is True


def observation_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        return ""
    results = value.get("results")
    if isinstance(results, list):
        return "\n".join(
            str(row.get("content", ""))
            for row in results
            if isinstance(row, Mapping) and row.get("content")
        )
    return value.get("content", "") if isinstance(value.get("content"), str) else ""


def trajectory_path(config: Mapping[str, Any], relpath: str) -> Path:
    raw_root = resolve_project_path(config["paths"]["raw_root"])
    posix = PurePosixPath(relpath)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"unsafe trajectory path: {relpath}")
    path = (raw_root / posix).resolve()
    path.relative_to(raw_root)
    return path


def load_turns(config: Mapping[str, Any], relpath: str) -> tuple[str, list[Turn]]:
    document = json.loads(trajectory_path(config, relpath).read_text(encoding="utf-8", errors="replace"))
    steps = document.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"trajectory has no ATIF steps: {relpath}")
    header = ""
    turns: list[Turn] = []
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        source, message = step.get("source"), step.get("message")
        if source == "user" and not header and isinstance(message, str):
            header = message
        if source != "agent" or not isinstance(message, str):
            continue
        commands, complete = command_values(message)
        turns.append(Turn(
            turn_index=len(turns),
            action=message,
            observation=observation_text(step.get("observation")),
            commands=tuple(commands),
            task_complete=complete,
        ))
    return header, turns


def transform_turns(turns: Sequence[Turn], transform: str, seed: int, identity: str) -> list[Turn]:
    output = list(turns)
    rng = random.Random(stable_int(transform, seed, identity))
    if transform == "identity":
        return output
    if transform == "turn_shuffle":
        rng.shuffle(output)
        return [dataclasses.replace(turn, turn_index=index) for index, turn in enumerate(output)]
    if transform == "bag_of_actions":
        actions = [turn.action for turn in output]
        rng.shuffle(actions)
        return [dataclasses.replace(turn, action=action) for turn, action in zip(output, actions)]
    raise ValueError(f"unknown trajectory transform: {transform}")


def leakage_hits(text: str) -> list[str]:
    lowered = text.casefold()
    return sorted(term for term in FORBIDDEN_TRAINING_TERMS if term.casefold() in lowered)


def render_target_only(tokenizer: Any, header: str, turns: Sequence[Turn], target_index: int, max_length: int):
    history = list(turns[:target_index])
    target = turns[target_index].action
    while True:
        messages: list[dict[str, str]] = [{"role": "user", "content": header}]
        for turn in history:
            messages.append({"role": "assistant", "content": turn.action})
            messages.append({"role": "tool", "content": turn.observation})
        messages.append({"role": "assistant", "content": target})
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if len(input_ids) <= max_length:
            break
        if not history:
            raise ValueError("task and target exceed max_length")
        history.pop(0)
    target_start = -1
    for start in range(len(input_ids) - len(target_ids), -1, -1):
        if input_ids[start : start + len(target_ids)] == target_ids:
            target_start = start
            break
    if target_start < 0:
        raise RuntimeError("target tokens not found in rendered Qwen3 chat sequence")
    labels = [-100] * len(input_ids)
    labels[target_start : target_start + len(target_ids)] = target_ids
    visible_text = "\n".join(message["content"] for message in messages)
    return messages, list(input_ids), labels, leakage_hits(visible_text)
