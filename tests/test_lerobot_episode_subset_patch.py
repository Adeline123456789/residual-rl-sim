import ast
import copy
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_ROOT = REPO_ROOT / "third_party/resfit/deps/lerobot"
SOURCE_RELATIVE_PATH = Path("lerobot/common/datasets/lerobot_dataset.py")
SOURCE_PATH = LEROBOT_ROOT / SOURCE_RELATIVE_PATH
PATCH_PATH = REPO_ROOT / "patches/lerobot/noncontiguous_episode_subset.patch"
PINNED_LEROBOT_COMMIT = "69901b9b6a2300914ca3de0ea14b6fa6e0203bd4"


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _FakeTorch:
    @staticmethod
    def BoolTensor(values):
        return tuple(values)


def _patched_source_and_reversal() -> tuple[str, bytes]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        target = root / SOURCE_RELATIVE_PATH
        target.parent.mkdir(parents=True)
        shutil.copy2(SOURCE_PATH, target)
        original = target.read_bytes()
        subprocess.run(
            ["git", "apply", "--check", str(PATCH_PATH)],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "apply", str(PATCH_PATH)],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        patched = target.read_text(encoding="utf-8")
        subprocess.run(
            ["git", "apply", "--reverse", "--check", str(PATCH_PATH)],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "apply", "--reverse", str(PATCH_PATH)],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return patched, target.read_bytes()


def _class_method(source: str, method_name: str, globals_dict: dict | None = None):
    tree = ast.parse(source)
    dataset_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LeRobotDataset"
    )
    method = next(
        node for node in dataset_class.body if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    function = copy.deepcopy(method)
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(SOURCE_PATH), "exec"), globals_dict or {}, namespace)
    return namespace[method_name]


def _initialization_mapping(source: str, episodes, meta_episodes):
    tree = ast.parse(source)
    dataset_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LeRobotDataset"
    )
    init = next(
        node for node in dataset_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    mapping_index = next(
        index
        for index, node in enumerate(init.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "_episode_index_to_data_index"
            for target in node.targets
        )
    )
    mapping_statements = [copy.deepcopy(init.body[mapping_index - 1]), copy.deepcopy(init.body[mapping_index])]
    module = ast.fix_missing_locations(ast.Module(body=mapping_statements, type_ignores=[]))
    dataset = SimpleNamespace(episodes=episodes, meta=SimpleNamespace(episodes=meta_episodes))
    exec(compile(module, str(SOURCE_PATH), "exec"), {}, {"self": dataset})
    return dataset._episode_index_to_data_index


class LeRobotEpisodeSubsetPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.patched_source, cls.reversed_source = _patched_source_and_reversal()

    def test_patch_targets_the_pinned_checkout_and_reverses_exactly(self) -> None:
        commit = subprocess.run(
            ["git", "-C", str(LEROBOT_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(commit, PINNED_LEROBOT_COMMIT)
        self.assertEqual(self.reversed_source, SOURCE_PATH.read_bytes())

    def test_subset_and_full_dataset_episode_maps_preserve_original_ids(self) -> None:
        subset_map = _initialization_mapping(
            self.patched_source,
            episodes=[2, 243],
            meta_episodes={0: {}, 1: {}, 2: {}, 243: {}},
        )
        full_map = _initialization_mapping(
            self.patched_source,
            episodes=None,
            meta_episodes={0: {}, 1: {}, 2: {}},
        )
        contiguous_map = _initialization_mapping(
            self.patched_source,
            episodes=[0, 1, 2],
            meta_episodes={0: {}, 1: {}, 2: {}},
        )
        nonzero_contiguous_map = _initialization_mapping(
            self.patched_source,
            episodes=[10, 11],
            meta_episodes={10: {}, 11: {}},
        )
        reordered_map = _initialization_mapping(
            self.patched_source,
            episodes=[243, 2],
            meta_episodes={2: {}, 243: {}},
        )
        self.assertEqual(subset_map, {2: 0, 243: 1})
        self.assertEqual(full_map, {0: 0, 1: 1, 2: 2})
        self.assertEqual(contiguous_map, full_map)
        self.assertEqual(nonzero_contiguous_map, {10: 0, 11: 1})
        self.assertEqual(reordered_map, {243: 0, 2: 1})

    def test_noncontiguous_high_episode_uses_subset_boundary_and_padding(self) -> None:
        get_query_indices = _class_method(
            self.patched_source,
            "_get_query_indices",
            {"torch": _FakeTorch},
        )
        dataset = SimpleNamespace(
            _episode_index_to_data_index={2: 0, 243: 1},
            episode_data_index={
                "from": [_Scalar(0), _Scalar(3)],
                "to": [_Scalar(3), _Scalar(5)],
            },
            delta_indices={"action": [-2, 0, 2]},
        )

        query_indices, padding = get_query_indices(dataset, idx=3, ep_idx=243)

        self.assertEqual(query_indices, {"action": [3, 3, 4]})
        self.assertEqual(padding, {"action_is_pad": (True, False, True)})

    def test_video_query_keeps_original_episode_id(self) -> None:
        getitem = _class_method(self.patched_source, "__getitem__")
        observed_episode_ids = []

        class _Rows:
            def __getitem__(self, index):
                self.last_index = index
                return {
                    "episode_index": _Scalar(243),
                    "timestamp": _Scalar(1.25),
                    "task_index": _Scalar(7),
                }

        dataset = SimpleNamespace(
            hf_dataset=_Rows(),
            delta_indices=None,
            meta=SimpleNamespace(
                video_keys=["observation.images.agentview"],
                camera_keys=[],
                tasks={7: "task"},
            ),
            image_transforms=None,
        )
        dataset._get_query_timestamps = lambda current_ts, query_indices: {
            "observation.images.agentview": [current_ts]
        }

        def query_videos(query_timestamps, episode_id):
            observed_episode_ids.append(episode_id)
            return {"observation.images.agentview": "decoded"}

        dataset._query_videos = query_videos

        item = getitem(dataset, 0)

        self.assertEqual(observed_episode_ids, [243])
        self.assertEqual(item["observation.images.agentview"], "decoded")
        self.assertEqual(item["task"], "task")


if __name__ == "__main__":
    unittest.main()
