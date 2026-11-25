"""Fit the existing ACT swing readout on training-only heading perturbations."""
import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from excavator_vlm.policies.act.inference import load_act
from safetensors.torch import load_file

import run_act_training_alternatives as common

ROOT = common.ROOT
SOURCE = ROOT / "outputs/experiments/act_training_alternatives_20261119_01/velocity/train/checkpoints/008000/pretrained_model"
ROW_INDICES = [0, 4, 8]  # Swing actions consumed before the next prediction.
HISTORY_LENGTH = 8
MAX_ANCHORS_PER_GROUP = 24


def rotate_delta(x, y, offset, lever):
    """Change the upper-body frame about an approximate radial reference."""
    cosine, sine = torch.cos(offset), torch.sin(offset)
    return cosine * (x + lever) + sine * y - lever, -sine * (x + lever) + cosine * y


def perturb(features, offset, lever, mean, std):
    result = features.clone()
    state = features[..., :len(mean)] * std + mean
    state[..., 0] += offset
    for i in (19, 22):
        state[..., i], state[..., i + 1] = rotate_delta(state[..., i], state[..., i + 1], offset, lever)
    sine, cosine = state[..., 24].clone(), state[..., 25].clone()
    state[..., 24] = sine * torch.cos(offset) - cosine * torch.sin(offset)
    state[..., 25] = cosine * torch.cos(offset) + sine * torch.sin(offset)
    result[..., :len(mean)] = (state - mean) / std
    return result


def annotations(episodes, data):
    wanted = set(episodes)
    result = {}
    columns = ["episode_index", "observation.stage_current_id", "observation.return_to_pile",
               "observation.scoop_index"]
    for path in sorted((data / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=columns)
        episode = table["episode_index"][0].as_py()
        if episode not in wanted:
            continue
        if episode in result or set(table["episode_index"].to_pylist()) != {episode}:
            raise ValueError(f"Expected one file per parent: {episode}")
        result[episode] = [np.asarray(table[key].to_pylist()).reshape(-1) for key in columns[1:]]
    if set(result) != wanted:
        raise ValueError("Missing split parents")
    return result


def encode(memory, features, annotations, cache, stats, rng):
    state_mean, state_std, action_mean, action_std = stats
    features = features.unsqueeze(0)
    clean, _ = memory.gru(memory.project(features))
    actions = torch.from_numpy(cache["target"][:, :3, 0])
    base = torch.from_numpy(cache["base"][:, :3, 0])
    valid = ~cache["padding"][:, :3].any(axis=1)
    frames = cache["frame_index"]
    stage, returning, scoop = [x[frames] for x in annotations]
    returning = returning.astype(bool)
    state = features[0, :, :len(state_mean)] * state_std + state_mean
    # Late transport and returning observations both occur in the original experts.
    # Pick low-rate near-alignment frames, away from initial lift and from digging.
    steady = (np.abs(actions.numpy()[:, 0] * action_std.item() + action_mean.item()) < .18)
    groups = {
        "carry": (stage == 9) & ~returning & (state[:, 26].numpy() > .002)
                 & (state[:, 22].abs().numpy() < 2.5),
        "return": returning & (state[:, 20].abs().numpy() < 2),
    }
    perturbed = {}
    for group, mask in groups.items():
        indices = np.flatnonzero(mask & valid & steady)
        indices = indices[indices >= HISTORY_LENGTH]
        indices = indices[scoop[indices] == scoop[indices - HISTORY_LENGTH]]
        if not len(indices):
            continue
        indices = np.sort(rng.choice(indices, min(len(indices), MAX_ANCHORS_PER_GROUP), replace=False))
        indices = torch.as_tensor(indices)
        indices = indices.repeat_interleave(4)
        history = indices[:, None] + torch.arange(1 - HISTORY_LENGTH, 1)
        start = history[:, 0]
        offsets = torch.from_numpy(rng.uniform(.18, 1.15, len(history)).astype(np.float32))
        signs = torch.tensor([-1., 1., -1., 1.]).repeat(len(history) // 4)
        # A constant offset through the window preserves angular rates.
        yaw = state[history, 0]
        offsets = torch.maximum(torch.minimum(offsets * signs, math.pi - yaw.amax(dim=1)),
                                -math.pi - yaw.amin(dim=1))
        lever = torch.from_numpy(rng.uniform(4., 8., len(history)).astype(np.float32))[:, None]
        shifted = perturb(features[0, history], offsets[:, None], lever, state_mean, state_std)
        encoded, _ = memory.gru(memory.project(shifted), clean[:, start - 1].contiguous())
        speed_limit = .48 if group == "carry" else .85
        desired = (actions[indices] * action_std + action_mean - .8 * offsets[:, None]).clamp(-speed_limit, speed_limit)
        perturbed[group] = (encoded[:, -1], base[indices], desired, offsets, actions[indices])
    return (clean[0, valid], base[valid], actions[valid], returning[valid]), perturbed


def add_bias(tensor):
    return torch.cat((tensor, torch.ones_like(tensor[:, :1])), dim=1)


def metrics(output, data, clean, stats):
    _, _, mean, std = stats
    clean_mae = {}
    recovery = {}
    for name, (matrix, base, action) in clean.items():
        error = (base + matrix @ output - action).abs() * std
        clean_mae[name] = error.mean().item()
    for group, (matrix, base, desired, offset, original) in data.items():
        pred = (base + matrix @ output) * std + mean
        correction = pred[:, 0] - (original[:, 0] * std + mean)
        large = offset.abs() >= .45
        recovery[group] = {
            "samples": len(pred),
            "mae": (pred - desired).abs().mean().item(),
            "large_offset_wrong_direction_rate": ((correction[large] * offset[large]) >= 0).float().mean().item(),
        }
    return {"clean_mae_rad_s": clean_mae, "synthetic_heading_recovery": recovery}


@torch.no_grad()
def train(output, smoke):
    common.SOURCE = SOURCE
    metadata = common.metadata()
    split = metadata["split_manifest"]
    train_ids, val_ids = split["train"], split["validation"]
    if set(train_ids) & set(val_ids + split["test"]) or set(val_ids) & set(split["test"]):
        raise ValueError("Parent split overlap")
    if smoke:
        train_ids, val_ids = train_ids[:3], val_ids[:3]
    source_hash = common.digest(SOURCE / "model.safetensors")
    cached_source_path = Path(json.loads((common.PREVIOUS / "experiment.json").read_text())["source"])
    frozen_source = load_file(str(cached_source_path / "model.safetensors"))
    velocity_weights = load_file(str(SOURCE / "model.safetensors"))
    for name, value in frozen_source.items():
        if not name.startswith("model.memory_adapter."):
            torch.testing.assert_close(value, velocity_weights[name], rtol=0, atol=0)
    del frozen_source, velocity_weights
    cache_hashes = json.loads((SOURCE.parents[4] / "cache_hashes.json").read_text())
    stats_path = SOURCE.parents[2] / "normalization_stats.json"
    if common.digest(stats_path) != metadata["normalization_stats_sha256"]:
        raise ValueError("Normalization statistics changed")
    stats = json.loads(stats_path.read_text())
    stats = tuple(torch.tensor(x, dtype=torch.float32) for x in (
        stats["observation.state"]["mean"], np.asarray(stats["observation.state"]["std"]) + 1e-8,
        stats["action"]["mean"][0], stats["action"]["std"][0],
    ))
    policy, _, _ = load_act(SOURCE, "cpu", original_act=True)
    memory = policy.model.memory_adapter
    prior = torch.cat((memory.output.weight[ROW_INDICES], memory.output.bias[ROW_INDICES, None]), dim=1).T
    notes = annotations(train_ids + val_ids, common.DATA)
    validation_clean = {"all": [], "return": []}
    validation_aug = {"carry": [], "return": []}
    gram = {"clean": 0., "carry": 0., "return": 0.}
    rhs = dict(gram)
    count = dict.fromkeys(gram, 0)
    rng = np.random.default_rng(20261123)

    def accumulate(group, matrix, residual):
        matrix, residual = matrix.double(), residual.double()
        gram[group] += matrix.T @ matrix
        rhs[group] += matrix.T @ residual
        count[group] += len(matrix)

    for episode in train_ids + val_ids:
        if common.digest(common.CACHE / f"episode_{episode:04d}.npz") != cache_hashes[str(episode)]:
            raise ValueError(f"Feature cache changed: {episode}")
        cache = common.cpu_data(episode)
        features = torch.from_numpy(cache["features"])
        (clean, base, action, returning), augmented = encode(memory, features, notes[episode], cache, stats, rng)
        clean = add_bias(clean)
        if episode in train_ids:
            # Use predominantly source anchoring on clean examples with a small expert term.
            clean_target = .2 * (action - base - clean @ prior)
            accumulate("clean", clean, clean_target)
            for group, (matrix, base_aug, desired, _, _) in augmented.items():
                matrix = add_bias(matrix)
                target = (desired - stats[2]) / stats[3] - base_aug - matrix @ prior
                accumulate(group, matrix, target)
        else:
            validation_clean["all"].append((clean, base, action))
            validation_clean["return"].append((clean[returning], base[returning], action[returning]))
            for group, (matrix, base_aug, desired, offset, original) in augmented.items():
                validation_aug[group].append((add_bias(matrix), base_aug, desired, offset, original))

    if not all(count.values()) or not all(validation_aug.values()):
        raise ValueError(f"Insufficient training/validation recovery coverage: {count}")
    validation_clean = {key: tuple(map(torch.cat, zip(*parts, strict=True))) for key, parts in validation_clean.items()}
    validation_aug = {key: tuple(map(torch.cat, zip(*parts, strict=True))) for key, parts in validation_aug.items()}
    initial = metrics(prior, validation_aug, validation_clean, stats)
    candidates = []
    for weight in (.00025, .0005, .00055, .00060, .00065, .00075, .0015, .003):
        system = gram["clean"] / count["clean"] + torch.eye(len(prior), dtype=torch.float64) * .01
        target = rhs["clean"] / count["clean"]
        for group in ("carry", "return"):
            system += weight * gram[group] / count[group]
            target += weight * rhs[group] / count[group]
        readout = prior + torch.linalg.solve(system, target).float()
        validation = metrics(readout, validation_aug, validation_clean, stats)
        candidates.append((weight, readout, validation))
    output.mkdir(parents=True, exist_ok=True)
    common.write(output / "validation.json", {
        "source": initial, "candidates": {str(weight): result for weight, _, result in candidates},
        "training_samples": count, "training_parents": train_ids, "validation_parents": val_ids,
        "test_rollouts_used_for_fitting": False, "smoke": smoke,
        "scope": "Cached-feature heading perturbation with frozen visual/base features; a sensitivity check, not simulated recovery.",
    })

    def eligible(result):
        clean = result["clean_mae_rad_s"]
        recovery = result["synthetic_heading_recovery"]
        return (all(clean[group] <= 1.05 * value for group, value in initial["clean_mae_rad_s"].items())
                and all(recovery[group]["mae"] <= .98 * value["mae"]
                        for group, value in initial["synthetic_heading_recovery"].items())
                and all(recovery[group]["large_offset_wrong_direction_rate"] <= value["large_offset_wrong_direction_rate"]
                        for group, value in initial["synthetic_heading_recovery"].items()))

    options = [candidate for candidate in candidates if eligible(candidate[2])]
    if not options:
        common.write(output / "status.json", {"status": "offline_gate_rejected", "baseline_replaced": False})
        return
    weight, selected, result = min(options, key=lambda entry: sum(
        item["mae"] for item in entry[2]["synthetic_heading_recovery"].values()))
    if smoke:
        common.write(output / "status.json", {"status": "smoke_passed", "weight": weight})
        return
    memory.output.weight[ROW_INDICES] = selected[:-1].T
    memory.output.bias[ROW_INDICES] = selected[-1]
    original = load_file(str(SOURCE / "model.safetensors"))
    for name, value in policy.state_dict().items():
        if name in ("model.memory_adapter.output.weight", "model.memory_adapter.output.bias"):
            preserved = [i for i in range(len(value)) if i not in ROW_INDICES]
            torch.testing.assert_close(value[preserved], original[name][preserved], rtol=0, atol=0)
            if torch.equal(value, original[name]):
                raise ValueError(f"Readout unchanged: {name}")
        else:
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    checkpoint = output / "checkpoint/pretrained_model"
    checkpoint.mkdir(parents=True)
    policy.save_pretrained(checkpoint)
    for file in SOURCE.glob("policy_*"):
        shutil.copy2(file, checkpoint / file.name)
    metadata["swing_recovery"] = {
        "source": str(SOURCE), "source_sha256": common.digest(SOURCE / "model.safetensors"),
        "training_uses_only_original_training_parents": True,
        "trained_readout_rows": ROW_INDICES, "perturbation_weight": weight,
    }
    common.write(checkpoint / "benchmark_provenance.json", metadata)
    restored, pre, _ = load_act(checkpoint, "cpu", original_act=True)
    from excavator_vlm.policies.act.visual_training import RGBEpisodes, batch_at
    rgb = RGBEpisodes(common.DATA).load(val_ids[0], 3)
    reference, _, _ = load_act(SOURCE, "cpu", original_act=True)
    for frame in range(3):
        raw, _, _, _ = batch_at([rgb], frame)
        batch = pre(raw)
        a = policy.predict_action_chunk(dict(batch))
        b = restored.predict_action_chunk(dict(batch))
        c = reference.predict_action_chunk(dict(batch))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        preserved = torch.ones_like(c, dtype=torch.bool)
        preserved[:, :3, 0] = False
        torch.testing.assert_close(b[preserved], c[preserved], rtol=0, atol=0)
    if common.digest(SOURCE / "model.safetensors") != source_hash:
        raise ValueError("Source checkpoint changed")
    common.write(output / "status.json", {
        "status": "ready_for_closed_loop", "selected_weight": weight, "validation": result,
        "checkpoint": str(checkpoint), "strict_native_rgb_reload": "passed",
        "baseline_replaced": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    train(args.output.resolve(), args.smoke)
