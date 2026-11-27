# Initial resume smoke startup failure

The isolated trainer stopped before the first rollout because the resumable overlay still imported the unsupported diffusion dependency while loading the frozen local ACT. The original error and command are preserved here. The corrected overlay reuses the fixed-evaluation ACT loader and ran the same smoke protocol in `20261126T205420598039Z`.
