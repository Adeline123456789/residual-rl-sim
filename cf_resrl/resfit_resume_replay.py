"""Capture the replay state of a paused Square ResFiT training loop."""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


_DUMP = "online_replay"
_ARCHIVE = "online_replay.tar.zst"


def _capture_runtime(buffer):
    # TorchRL 0.9.2 dumps storage, sampler and writer, but omits these live fields.
    with buffer._futures_lock:
        prefetched = [future.result() for future in buffer._prefetch_queue]
    generator = buffer._rng
    return {
        "size": len(buffer),
        "prefetch_capacity": buffer._prefetch_cap,
        "prefetched": prefetched,
        "n_step_tail": buffer._transform[0]._buffer,
        "write_count": buffer._writer._write_count,
        "last_cursor": buffer.storage._last_cursor,
        "rng": (str(generator.device), generator.get_state()) if generator is not None else None,
    }


def _restore_runtime(buffer, state):
    import torch

    if len(buffer) != state["size"] or buffer._prefetch_cap != state["prefetch_capacity"]:
        raise ValueError("Replay size or prefetch configuration does not match the checkpoint")
    with buffer._futures_lock:
        if buffer._prefetch_queue:
            raise ValueError("Replay must be restored before sampling any batches")
        for data, info in state["prefetched"]:
            if buffer._pin_memory:
                data = data.pin_memory()
            future = Future()
            future.set_result((data, info))
            buffer._prefetch_queue.append(future)
    buffer._transform[0]._buffer = state["n_step_tail"]
    buffer._writer._write_count = state["write_count"]
    buffer.storage._last_cursor = state["last_cursor"]
    rng = state["rng"]
    if rng is None:
        buffer.set_rng(None)
    else:
        device, rng_state = rng
        generator = torch.Generator(device=device)
        generator.set_state(rng_state)
        buffer.set_rng(generator)


def _require_uniform_offline_sampling(offline_rb):
    if offline_rb.sampler._alpha != 0.0 or offline_rb.sampler._beta != 0.0:
        raise ValueError("Square ResFiT replay restore requires an immutable uniform offline buffer")


def save_replay_state(online_rb, offline_rb, directory: Path) -> None:
    """Write replay to a fresh staging directory before capturing global RNG state.

    The caller pauses rollout and optimization at the step boundary. It also owns
    durability and publication of the whole checkpoint. The offline transition
    storage is immutable and belongs to the original run's offline cache.
    """
    import torch

    _require_uniform_offline_sampling(offline_rb)
    runtime = {"online": _capture_runtime(online_rb), "offline": _capture_runtime(offline_rb)}
    dump = directory / _DUMP
    archive = directory / _ARCHIVE
    online_rb.dumps(dump)
    subprocess.run(
        ["tar", "--create", "--use-compress-program=zstd -3 -T1 -q",
         "--file", str(archive.resolve()), "--directory", str(directory), _DUMP],
        check=True,
    )
    subprocess.run(["zstd", "-q", "--test", "--", str(archive)], check=True)
    shutil.rmtree(dump)
    torch.save(runtime, directory / "replay_runtime.pt")


def load_replay_state(online_rb, offline_rb, directory: Path) -> None:
    """Restore a fresh online replay and the already loaded offline cache."""
    import torch

    _require_uniform_offline_sampling(offline_rb)
    # Resume uses the same device as training. The n-step tail can hold CUDA tensors.
    runtime = torch.load(directory / "replay_runtime.pt", weights_only=False)
    with tempfile.TemporaryDirectory(prefix="cf-resrl-online-replay-") as scratch:
        with subprocess.Popen(
            ["zstd", "-dq", "-c", "--", str(directory / _ARCHIVE)], stdout=subprocess.PIPE
        ) as decompressor:
            with tarfile.open(fileobj=decompressor.stdout, mode="r|") as archive:
                archive.extractall(scratch, filter="data")
            # Consume the remainder of the compressed stream so zstd verifies the frame.
            while decompressor.stdout.read(1024 * 1024):
                pass
            if decompressor.wait():
                raise RuntimeError("Unable to decompress the online replay checkpoint")
        online_rb.loads(Path(scratch) / _DUMP)
    _restore_runtime(online_rb, runtime["online"])
    _restore_runtime(offline_rb, runtime["offline"])
