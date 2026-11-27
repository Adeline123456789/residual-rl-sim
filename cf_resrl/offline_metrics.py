"""Episode-level decision regret and gain calibration for matched branches."""

from __future__ import annotations

import numpy as np


def bootstrap_mean(values, *, samples: int, seed: int):
    """Resample whole source episodes, retaining all decisions from each draw."""
    episode_values = np.asarray(values, dtype=np.float64)
    if episode_values.ndim != 1 or not len(episode_values) or samples <= 0:
        raise ValueError("Episode bootstrap requires observations and a positive draw count")
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(episode_values), size=(samples, len(episode_values)))
    interval = np.quantile(episode_values[draws].mean(axis=1), (0.025, 0.975))
    return {"mean": float(episode_values.mean()), "ci95": interval.tolist()}


def evaluate_probe(
    data, prediction, *, costs, critical_margin, calibration_bins, bootstrap_samples, bootstrap_seed
):
    rows = len(data.source_episode_ids)
    if rows == 0 or len(set(data.source_episode_ids)) != rows:
        raise ValueError("Probe evaluation requires unique source episodes")
    actual = np.asarray(data.g_delta, dtype=np.float64)
    predicted = np.asarray(prediction["g_delta"], dtype=np.float64)
    if actual.shape != (rows,) or predicted.shape != (rows,) or not np.isfinite(predicted).all():
        raise ValueError("Predictions and labels must contain one finite gain per episode")
    cost = np.asarray(costs, dtype=np.float64)
    excess = actual[:, None] - cost
    decision = predicted[:, None] > cost
    optimal = excess > 0
    regret = np.maximum(excess, 0) - np.where(decision, excess, 0)
    error = decision != optimal
    critical = np.abs(excess) >= critical_margin
    interactions = None
    if "interaction" in prediction:
        interactions = np.asarray(prediction["interaction"], dtype=np.float64)
        if interactions.shape != (rows,) or not np.isfinite(interactions).all():
            raise ValueError("Expected one finite interaction prediction per source episode")

    sorted_episodes = np.argsort(predicted, kind="stable")
    calibration = [
        {
            "episodes": len(bin_ids),
            "predicted_gain": float(predicted[bin_ids].mean()),
            "observed_gain": float(actual[bin_ids].mean()),
        }
        for bin_ids in np.array_split(sorted_episodes, min(calibration_bins, rows))
    ]
    metrics = {
        "episode_count": rows,
        "regret": bootstrap_mean(regret.mean(axis=1), samples=bootstrap_samples, seed=bootstrap_seed),
        "g_delta_mae": float(np.abs(predicted - actual).mean()),
        "interaction_mae": (
            float(np.abs(interactions - (actual - data.g0)).mean()) if interactions is not None else None
        ),
        "gain_calibration": calibration,
        "calibration_mean_absolute_bin_error": sum(
            row["episodes"] * abs(row["predicted_gain"] - row["observed_gain"])
            for row in calibration
        ) / rows,
        "costs": [
            {
                "cost": float(value),
                "regret": bootstrap_mean(
                    regret[:, index], samples=bootstrap_samples, seed=bootstrap_seed
                ),
                "error_rate": float(error[:, index].mean()),
                "critical_opportunities": int(critical[:, index].sum()),
                "critical_error_count": int((error[:, index] & critical[:, index]).sum()),
                "critical_error_rate": (
                    float(error[critical[:, index], index].mean()) if critical[:, index].any() else None
                ),
                "refresh_rate": float(decision[:, index].mean()),
            }
            for index, value in enumerate(cost)
        ],
    }
    predictions = []
    for index, episode_id in enumerate(data.source_episode_ids):
        predictions.append({
            "source_episode_id": episode_id,
            "true_g_delta": float(actual[index]),
            "true_g_zero": float(data.g0[index]),
            "v10": float(data.v10[index]),
            "v11": float(data.v11[index]),
            "predicted_g_delta": float(predicted[index]),
            "predicted_g_zero": float(prediction["g_zero"][index]) if "g_zero" in prediction else None,
            "predicted_interaction": float(interactions[index]) if interactions is not None else None,
            "refresh": decision[index].tolist(),
            "oracle_refresh": optimal[index].tolist(),
            "regret": regret[index].tolist(),
        })
    return metrics, predictions
