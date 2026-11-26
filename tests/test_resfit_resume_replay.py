from concurrent.futures import ThreadPoolExecutor
import importlib.util
import tempfile
import unittest
from pathlib import Path

from cf_resrl.resfit_resume_replay import load_replay_state, save_replay_state


@unittest.skipUnless(importlib.util.find_spec("torchrl"), "requires the ResFiT runtime")
class ResFiTReplayResumeTests(unittest.TestCase):
    def test_restores_wrapped_storage_n_step_tail_and_prefetch(self):
        import torch
        from tensordict import TensorDict
        from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
        from torchrl.envs.transforms import MultiStepTransform

        def make_buffer(seed):
            buffer = TensorDictPrioritizedReplayBuffer(
                storage=LazyTensorStorage(8),
                alpha=0.0,
                beta=0.0,
                priority_key="_priority",
                transform=MultiStepTransform(n_steps=3, gamma=0.99),
                prefetch=2,
                batch_size=3,
                generator=torch.Generator().manual_seed(seed),
            )
            buffer._prefetch_executor.shutdown()
            buffer._prefetch_executor = ThreadPoolExecutor(max_workers=1)
            self.addCleanup(buffer._prefetch_executor.shutdown)
            return buffer

        def sample(buffer):
            result = buffer.sample()
            for future in buffer._prefetch_queue:
                future.result()
            return result

        def transition(step):
            return TensorDict(
                {
                    "obs": torch.tensor([[step]]),
                    "action": torch.tensor([[float(step)]]),
                    "next": {
                        "obs": torch.tensor([[step + 1]]),
                        "done": torch.tensor([[step % 5 == 0]]),
                        "reward": torch.tensor([[float(step % 7)]]),
                    },
                    "_priority": torch.tensor([10.0]),
                },
                batch_size=[1],
            )

        online, offline = make_buffer(123), make_buffer(456)
        for step in range(13):
            online.add(transition(step))
            offline.add(transition(step + 20))

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            offline_cache = directory / "offline_cache"
            offline.dumps(offline_cache)
            sample(online)
            sample(offline)
            save_replay_state(online, offline, directory)
            self.assertTrue((directory / "online_replay.tar.zst").is_file())
            self.assertFalse((directory / "online_replay").exists())
            resumed_online, resumed_offline = make_buffer(2), make_buffer(3)
            resumed_offline.loads(offline_cache)
            load_replay_state(resumed_online, resumed_offline, directory)

            self.assertEqual(online.write_count, resumed_online.write_count)
            self.assertEqual(len(resumed_online), resumed_online.storage.max_size)
            for step in range(13, 17):
                online.add(transition(step))
                resumed_online.add(transition(step))
                self.assertTrue((sample(online) == sample(resumed_online)).all())
                self.assertTrue((sample(offline) == sample(resumed_offline)).all())
