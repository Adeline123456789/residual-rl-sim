"""Exact Square environment snapshots at robosuite control-step boundaries.

This module deliberately covers only the simulator and Python-side state owned by
``RobosuiteGymWrapper``. ACT queues and the CF-ResRL scheduler have separate
lifecycles and must be snapshotted by their eventual owners.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import random
from typing import Any


ARM_CONTROLLER_FIELDS = (
    "goal_pos",
    "goal_ori",
    "relative_ori",
    "ori_ref",
    "_goal_update_mode",
    "new_update",
    "joint_pos",
    "joint_vel",
    "ref_pos",
    "ref_ori_mat",
    "ref_pos_vel",
    "ref_ori_vel",
    "J_pos",
    "J_ori",
    "J_full",
    "mass_matrix",
    "torques",
    "origin_pos",
    "origin_ori",
)
GRIPPER_CONTROLLER_FIELDS = (
    "goal_qvel",
    "new_update",
    "joint_pos",
    "joint_vel",
    "previous_qpos",
)
ROBOT_BUFFER_FIELDS = (
    "recent_qpos",
    "recent_actions",
    "recent_torques",
    "recent_ee_forcetorques",
    "recent_ee_pose",
    "recent_ee_vel",
    "recent_ee_vel_buffer",
    "recent_ee_acc",
)
OBSERVABLE_FIELDS = (
    "_sampling_timestep",
    "_time_since_last_sample",
    "_current_delay",
    "_current_observed_value",
    "_sampled",
    "_enabled",
    "_active",
)


@dataclass(frozen=True)
class ModelIdentity:
    xml_sha256: str
    nq: int
    nv: int
    na: int
    nu: int
    nmocap: int
    nuserdata: int
    integration_state_size: int


@dataclass(frozen=True)
class SquareEnvSnapshot:
    """Reusable snapshot of one Square environment, excluding policy state."""

    schema_version: int
    scope: str
    model: ModelIdentity
    integration_state: tuple[float, ...]
    wrapper_state: dict[str, Any]
    environment_state: dict[str, Any]
    controller_states: dict[str, dict[str, Any]]
    gripper_actions: dict[str, Any]
    robot_buffers: dict[str, Any]
    observable_states: dict[str, dict[str, Any]]
    python_random_state: object
    numpy_random_state: object
    environment_rng_state: object


def _modules(mujoco_api: Any | None) -> tuple[Any, Any]:
    import numpy as np

    if mujoco_api is None:
        import mujoco as mujoco_api

    return np, mujoco_api


def _copy_fields(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: deepcopy(getattr(value, field)) for field in fields}


def _restore_fields(value: Any, state: dict[str, Any]) -> None:
    for field, field_value in state.items():
        setattr(value, field, deepcopy(field_value))


def _square_components(wrapper: Any) -> tuple[Any, Any, dict[str, Any]]:
    if wrapper.env_name != "NutAssemblySquare":
        raise ValueError(f"snapshot requires NutAssemblySquare, got {wrapper.env_name!r}")
    env = wrapper.env
    if len(env.robots) != 1 or list(env.robots[0].arms) != ["right"]:
        raise ValueError("snapshot requires one right-arm Panda robot")
    robot = env.robots[0]
    controllers = robot.composite_controller.part_controllers
    if list(controllers) != ["right", "right_gripper"]:
        raise ValueError(
            "Square controller topology must be ['right', 'right_gripper'], "
            f"got {list(controllers)!r}"
        )
    if set(robot.gripper) != {"right"}:
        raise ValueError(f"Square gripper topology must be ['right'], got {list(robot.gripper)!r}")
    return env, robot, controllers


def _model_identity(sim: Any, mujoco_api: Any) -> ModelIdentity:
    raw_model = sim.model._model
    state_spec = mujoco_api.mjtState.mjSTATE_INTEGRATION
    return ModelIdentity(
        xml_sha256=hashlib.sha256(sim.model.get_xml().encode("utf-8")).hexdigest(),
        nq=int(raw_model.nq),
        nv=int(raw_model.nv),
        na=int(raw_model.na),
        nu=int(raw_model.nu),
        nmocap=int(raw_model.nmocap),
        nuserdata=int(raw_model.nuserdata),
        integration_state_size=int(mujoco_api.mj_stateSize(raw_model, state_spec)),
    )


def capture_square_snapshot(
    wrapper: Any, *, mujoco_api: Any | None = None
) -> SquareEnvSnapshot:
    """Capture one reusable Square snapshot without advancing the environment."""
    np, mujoco_api = _modules(mujoco_api)
    env, robot, controllers = _square_components(wrapper)
    identity = _model_identity(env.sim, mujoco_api)
    state_spec = mujoco_api.mjtState.mjSTATE_INTEGRATION
    integration_state = np.empty(identity.integration_state_size, dtype=np.float64)
    mujoco_api.mj_getState(
        env.sim.model._model,
        env.sim.data._data,
        integration_state,
        state_spec,
    )

    controller_states = {
        "right": _copy_fields(controllers["right"], ARM_CONTROLLER_FIELDS),
        "right_gripper": _copy_fields(
            controllers["right_gripper"], GRIPPER_CONTROLLER_FIELDS
        ),
    }
    observable_states = {
        name: _copy_fields(observable, OBSERVABLE_FIELDS)
        for name, observable in env._observables.items()
    }
    return SquareEnvSnapshot(
        schema_version=1,
        scope="square_environment_only__act_and_scheduler_excluded",
        model=identity,
        integration_state=tuple(float(value) for value in integration_state),
        wrapper_state=deepcopy(
            {"episode_steps": wrapper.episode_steps, "_last_obs": wrapper._last_obs}
        ),
        environment_state=deepcopy(
            {
                "timestep": env.timestep,
                "cur_time": env.cur_time,
                "done": env.done,
                "_ep_meta": env._ep_meta,
                "objects_on_pegs": env.objects_on_pegs,
                "nut_id": env.nut_id,
                "obj_to_use": env.obj_to_use,
                "_obs_cache": env._obs_cache,
            }
        ),
        controller_states=deepcopy(controller_states),
        gripper_actions={
            "right": deepcopy(robot.gripper["right"].current_action)
        },
        robot_buffers=_copy_fields(robot, ROBOT_BUFFER_FIELDS),
        observable_states=deepcopy(observable_states),
        python_random_state=deepcopy(random.getstate()),
        numpy_random_state=deepcopy(np.random.get_state()),
        environment_rng_state=deepcopy(env.rng.bit_generator.state),
    )


def restore_square_snapshot(
    wrapper: Any,
    snapshot: SquareEnvSnapshot,
    *,
    mujoco_api: Any | None = None,
) -> None:
    """Restore a snapshot without calling reset or forcing an observation update."""
    np, mujoco_api = _modules(mujoco_api)
    env, robot, controllers = _square_components(wrapper)
    observed_identity = _model_identity(env.sim, mujoco_api)
    if observed_identity != snapshot.model:
        raise ValueError(
            f"snapshot model mismatch: observed={observed_identity}, expected={snapshot.model}"
        )
    if len(snapshot.integration_state) != snapshot.model.integration_state_size:
        raise ValueError("snapshot integration state length does not match its model identity")
    if set(snapshot.observable_states) != set(env._observables):
        raise ValueError("snapshot observable topology does not match the environment")

    state_spec = mujoco_api.mjtState.mjSTATE_INTEGRATION
    integration_state = np.asarray(snapshot.integration_state, dtype=np.float64)
    mujoco_api.mj_setState(
        env.sim.model._model,
        env.sim.data._data,
        integration_state,
        state_spec,
    )
    mujoco_api.mj_forward(env.sim.model._model, env.sim.data._data)

    _restore_fields(env, snapshot.environment_state)
    _restore_fields(controllers["right"], snapshot.controller_states["right"])
    _restore_fields(
        controllers["right_gripper"], snapshot.controller_states["right_gripper"]
    )
    robot.gripper["right"].current_action = deepcopy(
        snapshot.gripper_actions["right"]
    )
    _restore_fields(robot, snapshot.robot_buffers)
    for name, state in snapshot.observable_states.items():
        _restore_fields(env._observables[name], state)
    _restore_fields(wrapper, snapshot.wrapper_state)

    random.setstate(deepcopy(snapshot.python_random_state))
    np.random.set_state(deepcopy(snapshot.numpy_random_state))
    env.rng.bit_generator.state = deepcopy(snapshot.environment_rng_state)
