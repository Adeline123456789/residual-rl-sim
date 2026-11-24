import copy
from collections import OrderedDict
import random
import sys
import types
import unittest
from unittest.mock import patch

from cf_resrl.snapshot import capture_square_snapshot, restore_square_snapshot


class FakeGlobalRandom:
    state = ("numpy", 0)

    @classmethod
    def get_state(cls):
        return copy.deepcopy(cls.state)

    @classmethod
    def set_state(cls, state):
        cls.state = copy.deepcopy(state)


class FakeNumpy(types.ModuleType):
    float64 = float
    random = FakeGlobalRandom

    @staticmethod
    def empty(size, dtype=None):
        return [0.0] * size

    @staticmethod
    def asarray(values, dtype=None):
        return list(values)


class FakeMujoco:
    class mjtState:
        mjSTATE_INTEGRATION = 4111

    @staticmethod
    def mj_stateSize(model, state_spec):
        return len(model.integration_shape)

    @staticmethod
    def mj_getState(model, data, target, state_spec):
        target[:] = data.integration

    @staticmethod
    def mj_setState(model, data, values, state_spec):
        data.integration = list(values)

    @staticmethod
    def mj_forward(model, data):
        data.forward_calls += 1


class StateObject:
    def __init__(self, **values):
        self.__dict__.update(values)


def _buffer(value):
    return StateObject(current=[value], last=[value - 1])


def _controller(fields, base):
    return StateObject(**{name: [base, index] for index, name in enumerate(fields)})


ARM_FIELDS = (
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
GRIPPER_FIELDS = (
    "goal_qvel",
    "new_update",
    "joint_pos",
    "joint_vel",
    "previous_qpos",
)


def make_wrapper(xml="<mujoco model='square'/>"):
    raw_model = StateObject(
        nq=4,
        nv=4,
        na=0,
        nu=3,
        nmocap=1,
        nuserdata=2,
        integration_shape=range(9),
    )
    model = StateObject(_model=raw_model, get_xml=lambda: xml)
    raw_data = StateObject(integration=[float(i) for i in range(9)], forward_calls=0)
    sim = StateObject(model=model, data=StateObject(_data=raw_data))

    arm = _controller(ARM_FIELDS, 10)
    gripper_controller = _controller(GRIPPER_FIELDS, 20)
    arm.new_update = True
    arm._goal_update_mode = "achieved"
    gripper_controller.new_update = True
    controllers = OrderedDict(
        [("right", arm), ("right_gripper", gripper_controller)]
    )
    gripper = StateObject(current_action=[0.4])
    robot = StateObject(
        arms=["right"],
        composite_controller=StateObject(part_controllers=controllers),
        gripper={"right": gripper},
        recent_qpos=_buffer(1),
        recent_actions=_buffer(2),
        recent_torques=_buffer(3),
        recent_ee_forcetorques={"right": _buffer(4)},
        recent_ee_pose={"right": _buffer(5)},
        recent_ee_vel={"right": _buffer(6)},
        recent_ee_vel_buffer={"right": _buffer(7)},
        recent_ee_acc={"right": _buffer(8)},
    )
    observable = StateObject(
        _sampling_timestep=0.05,
        _time_since_last_sample=0.02,
        _current_delay=0.01,
        _current_observed_value=[3.0],
        _sampled=True,
        _enabled=True,
        _active=True,
    )
    bit_generator = StateObject(state={"counter": 7})
    env = StateObject(
        sim=sim,
        robots=[robot],
        timestep=12,
        cur_time=0.6,
        done=False,
        _ep_meta={"seed": 5},
        objects_on_pegs=[0.0, 1.0],
        nut_id=0,
        obj_to_use="SquareNut",
        _obs_cache={"robot0_joint_pos": [1.0]},
        _observables=OrderedDict([("robot0_joint_pos", observable)]),
        rng=StateObject(bit_generator=bit_generator),
    )
    wrapper = StateObject(
        env_name="NutAssemblySquare",
        env=env,
        episode_steps=12,
        _last_obs={"observation.state": [1.0, 2.0]},
    )
    return wrapper


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.numpy = FakeNumpy("numpy")
        self.numpy.random.state = ("numpy", 17)
        self.numpy_patch = patch.dict(sys.modules, {"numpy": self.numpy})
        self.numpy_patch.start()
        random.seed(29)

    def tearDown(self):
        self.numpy_patch.stop()

    def test_restores_physics_python_state_and_rng(self):
        wrapper = make_wrapper()
        snapshot = capture_square_snapshot(wrapper, mujoco_api=FakeMujoco)

        wrapper.env.sim.data._data.integration = [99.0] * 9
        wrapper.env.timestep = 90
        wrapper.env.objects_on_pegs[0] = 9.0
        wrapper.env.robots[0].gripper["right"].current_action[0] = -1.0
        wrapper.env.robots[0].recent_actions.current[0] = 77
        arm = wrapper.env.robots[0].composite_controller.part_controllers["right"]
        arm.goal_pos[0] = -10
        arm.ref_pos[0] = -11
        arm.new_update = False
        gripper_controller = (
            wrapper.env.robots[0].composite_controller.part_controllers[
                "right_gripper"
            ]
        )
        gripper_controller.goal_qvel[0] = -20
        gripper_controller.new_update = False
        wrapper.env._observables["robot0_joint_pos"]._current_observed_value[0] = 88
        wrapper.episode_steps = 90
        random.seed(3)
        self.numpy.random.state = ("changed", 9)
        wrapper.env.rng.bit_generator.state = {"counter": 99}

        restore_square_snapshot(wrapper, snapshot, mujoco_api=FakeMujoco)

        self.assertEqual(wrapper.env.sim.data._data.integration, list(range(9)))
        self.assertEqual(wrapper.env.sim.data._data.forward_calls, 1)
        self.assertEqual(wrapper.env.timestep, 12)
        self.assertEqual(wrapper.env.objects_on_pegs, [0.0, 1.0])
        self.assertEqual(
            wrapper.env.robots[0].gripper["right"].current_action, [0.4]
        )
        self.assertEqual(wrapper.env.robots[0].recent_actions.current, [2])
        self.assertEqual(arm.goal_pos, [10, 0])
        self.assertEqual(arm.ref_pos, [10, 8])
        self.assertTrue(arm.new_update)
        self.assertEqual(gripper_controller.goal_qvel, [20, 0])
        self.assertTrue(gripper_controller.new_update)
        self.assertEqual(
            wrapper.env._observables["robot0_joint_pos"]._current_observed_value,
            [3.0],
        )
        self.assertEqual(wrapper.episode_steps, 12)
        self.assertEqual(random.getstate(), snapshot.python_random_state)
        self.assertEqual(self.numpy.random.state, snapshot.numpy_random_state)
        self.assertEqual(wrapper.env.rng.bit_generator.state, {"counter": 7})

    def test_snapshot_can_be_restored_twice_without_mutation(self):
        wrapper = make_wrapper()
        snapshot = capture_square_snapshot(wrapper, mujoco_api=FakeMujoco)
        original_state = snapshot.integration_state

        restore_square_snapshot(wrapper, snapshot, mujoco_api=FakeMujoco)
        wrapper.env.sim.data._data.integration[0] = 800.0
        wrapper.env._obs_cache["robot0_joint_pos"][0] = 500.0
        restore_square_snapshot(wrapper, snapshot, mujoco_api=FakeMujoco)

        self.assertEqual(snapshot.integration_state, original_state)
        self.assertEqual(wrapper.env.sim.data._data.integration[0], 0.0)
        self.assertEqual(wrapper.env._obs_cache["robot0_joint_pos"], [1.0])

    def test_model_mismatch_is_rejected_before_restore(self):
        source = make_wrapper()
        snapshot = capture_square_snapshot(source, mujoco_api=FakeMujoco)
        target = make_wrapper(xml="<mujoco model='different'/>")

        with self.assertRaisesRegex(ValueError, "snapshot model mismatch"):
            restore_square_snapshot(target, snapshot, mujoco_api=FakeMujoco)
        self.assertEqual(target.env.sim.data._data.forward_calls, 0)

    def test_controller_topology_mismatch_is_rejected(self):
        wrapper = make_wrapper()
        wrapper.env.robots[0].composite_controller.part_controllers["extra"] = (
            StateObject()
        )
        with self.assertRaisesRegex(ValueError, "controller topology"):
            capture_square_snapshot(wrapper, mujoco_api=FakeMujoco)

    def test_snapshot_scope_excludes_policy_and_scheduler(self):
        snapshot = capture_square_snapshot(make_wrapper(), mujoco_api=FakeMujoco)
        self.assertEqual(
            snapshot.scope, "square_environment_only__act_and_scheduler_excluded"
        )


if __name__ == "__main__":
    unittest.main()
