"""Dependency-free contracts for the pinned Square ACT checkpoint candidate."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


EXPECTED_INPUTS = {
    "observation.state": {"type": "STATE", "shape": [9]},
    "observation.images.agentview": {"type": "VISUAL", "shape": [3, 84, 84]},
    "observation.images.robot0_eye_in_hand": {
        "type": "VISUAL",
        "shape": [3, 84, 84],
    },
}
EXPECTED_OUTPUTS = {"action": {"type": "ACTION", "shape": [7]}}


def validate_act_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the architecture fields needed by the Square execution path."""
    expected_scalars = {
        "type": "act",
        "chunk_size": 20,
        "n_action_steps": 20,
        "vision_backbone": "resnet18",
        "use_vae": False,
    }
    for key, expected in expected_scalars.items():
        if config.get(key) != expected:
            raise ValueError(f"checkpoint config {key} is {config.get(key)!r}, expected {expected!r}")
    if config.get("input_features") != EXPECTED_INPUTS:
        raise ValueError("checkpoint input features do not match Square state/cameras")
    if config.get("output_features") != EXPECTED_OUTPUTS:
        raise ValueError("checkpoint output features do not match the 7-D Square action")
    return {
        "policy_type": "act",
        "state_dim": 9,
        "action_dim": 7,
        "camera_keys": [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_hand",
        ],
        "camera_shape_chw": [3, 84, 84],
        "chunk_size": 20,
        "n_action_steps": 20,
        "vision_backbone": "resnet18",
        "use_vae": False,
    }


def validate_file_inventory(
    observed: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]
) -> None:
    """Require the remote repository file inventory to match the registered one."""
    normalized_observed = sorted((dict(row) for row in observed), key=lambda row: row["path"])
    normalized_expected = sorted((dict(row) for row in expected), key=lambda row: row["path"])
    if normalized_observed != normalized_expected:
        raise ValueError(
            f"checkpoint file inventory mismatch: observed={normalized_observed}, "
            f"expected={normalized_expected}"
        )


def validate_unpublished_license(
    card_data: Any, tags: Sequence[str], paths: Sequence[str]
) -> dict[str, Any]:
    """Verify that the pinned revision publishes no model-card license metadata."""
    license_tags = sorted(tag for tag in tags if tag.startswith("license:"))
    readme_present = any(path.lower() == "readme.md" for path in paths)
    if card_data is not None or license_tags or readme_present:
        raise ValueError("checkpoint revision now publishes model-card license metadata")
    return {
        "card_data": None,
        "license_tags": [],
        "readme_present": False,
    }
