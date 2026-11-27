"""Verify the frozen E0 collection, optionally including its release archive."""

import argparse
import base64
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import zlib

import numpy as np


def digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(block)
    return result.hexdigest()


def sha256(path):
    with path.open("rb") as stream:
        return digest(stream)


def read(path):
    return json.loads(path.read_text())


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def check_trace(steps, start, horizon, label):
    require(0 < len(steps) <= horizon, f"{label}: trace length")
    for offset, step in enumerate(steps):
        require(step["tick"] == start + offset and step["control_steps"] == start + offset + 1, label)
        require(len(step["applied_env_action"]) == 7 and all(
            math.isfinite(action) and -1 <= action <= 1 for action in step["applied_env_action"]
        ), f"{label}: applied action outside physical bounds")
        require(math.isfinite(step["reward"]), f"{label}: nonfinite reward")
        require(offset == len(steps) - 1 or not (step["terminated"] or step["truncated"]), label)
        require(re.fullmatch(r"[0-9a-f]{64}", step["integration_state_sha256"]), label)
    require(len(steps) == horizon or steps[-1]["terminated"] or steps[-1]["truncated"], label)


def check_actor_observation(observations):
    expected_shapes = {
        "observation.state": (1, 9),
        "observation.images.agentview": (1, 3, 84, 84),
        "observation.images.robot0_eye_in_hand": (1, 3, 84, 84),
        "observation.base_action": (1, 7),
    }
    require(observations.keys() == expected_shapes.keys(), "actor observation has an unexpected channel")
    decoded = {}
    for name, shape in expected_shapes.items():
        entry = observations[name]
        require(entry["encoding"] == "zlib+base64" and tuple(entry["shape"]) == shape, name)
        require(np.dtype(entry["dtype"]) == np.dtype("float32"), name)
        raw = zlib.decompress(base64.b64decode(entry["data"], validate=True))
        require(hashlib.sha256(raw).hexdigest() == entry["sha256"], f"{name}: actor observation hash")
        decoded[name] = np.frombuffer(raw, dtype=entry["dtype"]).reshape(shape)
        require(np.isfinite(decoded[name]).all(), name)
    return decoded


def check_group(run, row, schema, expected_labels, validate_branches):
    name = row["source_episode_id"]
    for key, path in {"raw_group": f"groups/{name}.json.gz", "features": f"features/{name}.npz"}.items():
        require(row[key] == path and sha256(run / path) == row[f"{key}_sha256"], f"{name}: {key} hash")
    with gzip.open(run / row["raw_group"], "rt") as stream:
        group = json.load(stream)
    for key in ("seed", "selector_seed", "source_episode_id", "distribution", "bank", "split", "source_tick"):
        require(group[key] == row[key], f"{name}: {key} index binding")
    tick = row["source_tick"]
    scout = group["nominal_scout"]
    source_steps = scout["steps"]
    check_trace(source_steps, 0, 300, f"{name} scout")
    require(0 <= tick < len(source_steps), f"{name}: source tick")
    require(group["source_episode_length"] == scout["environment_steps"] == row["source_episode_steps"] == len(source_steps), name)
    require(scout["act_calls"] == row["source_episode_act_calls"] == 1 + source_steps[-1]["act_calls_since_initial_chunk"], name)
    require(scout["wall_seconds"] == row["source_episode_wall_seconds"] and scout["wall_seconds"] > 0, name)
    require(scout["return"] == sum(step["reward"] for step in source_steps), name)
    require(scout["success"] == source_steps[-1]["terminated"], name)
    require(group["source_prefix"] == source_steps[:tick] and group["source_prefix_steps"] == tick, name)
    require(group["source_prefix_act_calls"] == 1 + source_steps[tick]["act_calls_since_initial_chunk"], name)
    require(not any(step["refresh_requested_after_step"] for step in source_steps), name)

    selection = group["selection"]
    require(selection["distribution"] == row["distribution"] and selection["selector_seed"] == row["selector_seed"], name)
    norm = lambda step: math.fsum(x * x for x in step["proposed_residual"])
    require(selection["selected_residual_l2"] == math.sqrt(norm(source_steps[tick])), name)
    if row["distribution"] == "natural":
        rng = random.Random(row["selector_seed"])
        sampled_tick = 0
        for i in range(len(source_steps)):
            if rng.randrange(i + 1) == 0:
                sampled_tick = i
        require(selection["rule"] == "uniform_reservoir" and selection["within_episode_probability"] == 1 / len(source_steps), name)
    else:
        sampled_tick = max(range(len(source_steps)), key=lambda i: norm(source_steps[i]))
        require(selection["rule"] == "max_residual_l2_earliest_tie" and selection["within_episode_probability"] is None, name)
    require(sampled_tick == tick and group["candidate_residual"] == source_steps[tick]["proposed_residual"], name)

    active = group["active_chunk"]
    chunk_index = tick - active["source_tick"]
    require(0 <= chunk_index < 20 and len(active["actions"]) == 20, name)
    require(active["actions"][chunk_index] == source_steps[tick]["base_action"], name)
    actor = check_actor_observation(group["actor_observation"])
    with np.load(run / row["features"], allow_pickle=False) as features:
        require(set(features.files) == set(schema), name)
        for feature, expected in schema.items():
            value = features[feature]
            require(list(value.shape) == expected["shape"] and str(value.dtype) == "float32", name)
            require(np.isfinite(value).all(), name)
        suffix = np.zeros((20, 7), dtype=np.float32)
        suffix[:20 - chunk_index] = active["actions"][chunk_index:]
        expected = {
            "base": actor["observation.base_action"][0],
            "chunk_index": [chunk_index],
            "chunk_mask": [1] * (20 - chunk_index) + [0] * chunk_index,
            "chunk_suffix": suffix,
            "residual": np.asarray(group["candidate_residual"], dtype=np.float32),
            "state": actor["observation.state"][0],
            "tick": [tick],
        }
        require(all(np.array_equal(features[key], value) for key, value in expected.items()), name)

    branches = group["branches"]
    require(branches.keys() == set(expected_labels) == row["branches"].keys(), name)
    require(group["remaining_horizon"] == 300 - tick, name)
    for label, branch in branches.items():
        steps = branch["steps"]
        check_trace(steps, tick, 300 - tick, f"{name}/{label}")
        require({key: value for key, value in branch.items() if key != "steps"} == row["branches"][label], name)
        require(branch["environment_steps"] == len(steps) and branch["wall_seconds"] > 0, name)
        require(branch["return"] == sum(step["reward"] for step in steps) and branch["success"] == steps[-1]["terminated"], name)
        discounted = sum(0.99 ** i * step["reward"] for i, step in enumerate(steps))
        require(branch["discounted_return"] == discounted, name)
        require(group["factorial_values"][label.lower()] == discounted, name)
    snapshot = SimpleNamespace(
        logical_tick=tick,
        active_chunk=SimpleNamespace(**active),
        next_generation=active["generation"] + 1,
        counters=SimpleNamespace(act_calls=group["source_prefix_act_calls"] - 1),
    )
    validate_branches(branches, scheduler_snapshot=snapshot, delay=2, remaining=300 - tick,
                      candidate=group["candidate_residual"], low=[-1.0] * 7, high=[1.0] * 7)
    require(all({k: v for k, v in b.items() if k != "steps"} == row["branches"][label]
                for label, b in branches.items()), name)
    require(branches["V10"]["steps"] == source_steps[tick:], f"{name}: V10 kept-policy replay")


def check_archive(archive, expected):
    seen = set()
    with tarfile.open(archive, "r|") as stream:
        for member in stream:
            if member.isdir() and member.name in {"features", "groups"}:
                continue
            require(member.isfile() and member.name in expected and member.name not in seen, member.name)
            with stream.extractfile(member) as file:
                require(digest(file) == expected[member.name], f"archive: {member.name}")
            seen.add(member.name)
    require(seen == set(expected), "archive is missing expected entries")
    return {"name": archive.name, "bytes": archive.stat().st_size, "sha256": sha256(archive), "files": len(seen)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path[:0] = [str(repo), str(repo / "scripts")]
    from cf_resrl.e0_collector import _validate_branches
    from cf_resrl.e0_report import summarize_e0
    from collect_e0 import _plan

    run = Path(__file__).resolve().parent
    status = read(run / "status.json")
    manifest = read(run / "run_manifest.json")
    schema = read(run / "feature_schema.json")
    summary = read(run / "summary.json")
    protocol = manifest["protocol"]
    expected = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{manifest['git_commit']}:configs/cf_resrl/square_e0.json"]
    )
    require(json.loads(expected) == protocol and not manifest["git_status_at_start"], "frozen launch protocol")
    require(status["status"] == "COMPLETED" and status["completed_groups"] == status["planned_groups"] == 720, "status")
    require(sha256(run / "group_index.jsonl") == status["index_sha256"], "index checksum")
    require(sha256(run / "summary.json") == status["summary_sha256"], "summary checksum")
    gate = repo / protocol["gate_attempt"]
    require(sha256(gate / "summary.json") == protocol["gate_summary_sha256"], "gate checksum")
    require(read(gate / "run_manifest.json")["identities"] == manifest["identities"], "gate policy identity")
    source = repo / manifest["source_resfit_attempt"]
    for name, info in manifest["source_resfit_dependency_locks"].items():
        require(sha256(source / "runtime" / name) == info["sha256"], f"runtime lock {name}")

    rows = [json.loads(line) for line in (run / "group_index.jsonl").read_text().splitlines()]
    plan = _plan(protocol)
    require(plan == manifest["source_plan"] and len(rows) == len(plan) == 720, "frozen source plan")
    require(len({row["source_episode_id"] for row in rows}) == len({row["seed"] for row in rows}) == 720,
            "source leakage or duplicate")
    require(not {row["seed"] for row in rows} & {g["seed"] for g in read(gate / "summary.json")["groups"]},
            "source seeds overlap the gate")
    require(set(schema) == {"base", "chunk_index", "chunk_mask", "chunk_suffix", "residual", "state", "tick", "visual"},
            "unexpected feature channel")
    log_rows = [line for line in (run / "stdout_stderr.log").read_text().splitlines() if line.startswith("E0 episode ")]
    require(len(log_rows) == len(rows), "episode counts in the collection log")
    files = {}
    for i, (row, planned, log_row) in enumerate(zip(rows, plan, log_rows), start=1):
        require(all(row[key] == value for key, value in planned.items()), f"source {i} differs from plan")
        require(log_row == (
            f"E0 episode {i}/720: {row['source_episode_id']}, {row['distribution']} "
            f"{row['bank']}, tick {row['source_tick']}"
        ), f"source {i} differs from log")
        labels = ("V00", "V01", "V10", "V11") if row["bank"] == "factorial" else ("V10", "V11")
        check_group(run, row, schema, labels, _validate_branches)
        for key in ("raw_group", "features"):
            files[row[key]] = row[f"{key}_sha256"]
    actual = {str(path.relative_to(run)) for key in ("groups", "features") for path in (run / key).iterdir()}
    require(actual == set(files) and len(files) == 1440, "unexpected collection files")
    recomputed = summarize_e0(rows, costs=protocol["decision_costs"],
                             bootstrap_samples=protocol["bootstrap_episodes"], bootstrap_seed=protocol["bootstrap_seed"])
    require(recomputed == summary and recomputed["budgets"]["overall"]["branches"]["rollouts"] == 2240,
            "summary reproducibility or branch budget")
    splits = Counter((row["bank"], row["distribution"], row["split"]) for row in rows)
    verification = {
        "status": "VERIFIED",
        "source_groups": len(rows),
        "source_group_splits": {"/".join(key): count for key, count in sorted(splits.items())},
        "exact_v10_nominal_suffix_replays": len(rows),
        "files_verified": len(files),
        "collection_log_sha256": sha256(run / "stdout_stderr.log"),
        "feature_schema_sha256": sha256(run / "feature_schema.json"),
        "run_manifest_sha256": sha256(run / "run_manifest.json"),
        "registered_config_sha256": hashlib.sha256(expected).hexdigest(),
        "index_sha256": status["index_sha256"],
        "summary_sha256": status["summary_sha256"],
        "original_gate_summary_sha256": protocol["gate_summary_sha256"],
        "budget": recomputed["budgets"]["overall"],
        "recomputed_bootstrap_summary_exact": True,
    }
    if args.archive:
        verification["archive"] = check_archive(args.archive, files)
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
