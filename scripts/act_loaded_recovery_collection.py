"""Training-only collection of loaded swing recovery demonstrations in Newton."""
import copy
from pathlib import Path

import numpy as np

from act_recovery_expert import RecoveryExpert, RecoveryFailure
from excavator_reachability import AuthoredExcavatorChain
from eval_smolvla_repeat40 import atomic_json, read


def loaded_overshoot(previous, state, distance):
    """Check that the loaded bucket is moving farther past its closest approach."""
    closest_distance, closest_swing = previous
    swing_offset = abs(float(state[0]) - closest_swing)
    return (closest_distance <= 2.75 and distance >= closest_distance + .35
            and swing_offset >= .12 and float(state[4]) * (float(state[0]) - closest_swing) > .005)


class LoadedRecoveryCollection:
    def __init__(self, plan_path, folder, scene, reference, repeated):
        plan = read(plan_path)
        provenance = reference["provenance"]
        episode = provenance["source_export_episode"]
        split = read(Path(plan["split"]))["split_manifest"]
        if episode != plan["parent"] or episode not in split["train"] or episode in split["validation"] + split["test"]:
            raise ValueError("Recovery collection requires an original training parent")
        self.model = read(Path(plan["robot_geometry"]))
        self.chain = AuthoredExcavatorChain(self.model)
        self.scene = copy.deepcopy(scene)
        if scene["simulation_settings"]["soil_backend"] != "newton_mpm":
            raise ValueError("Recovery collection requires Newton MPM")
        self.folder = folder
        self.repeated = repeated
        self.cycle = None
        self.best = (float("inf"), 0.)
        self.expert = None
        self.handoff = None
        self.limit = None
        self.expert_start = None
        self.failure = None
        self.accepted = False
        self.stable_deposits = 0
        self.expert_actions = 0
        self.contiguous_unclipped = True
        self.complete = False
        self.controlled_overshoot = plan["controlled_overshoot"]
        self.perturbation_start = None
        self.perturbation_direction = None
        self.perturbation_steps = 0
        self.mass = repeated.spec["metric_point_mass_kg"]
        self.minimum_kg = float(repeated.spec["minimum_per_cycle_kg"])

    def action(self, step, state, reply):
        if self.complete:
            return None, "collection_finished"
        retained = int(self.repeated.transfer["retained_particles"])
        if self.expert is None:
            cycle = len(self.repeated.cycles)
            if cycle != self.cycle:
                self.best = (float("inf"), 0.)
                self.perturbation_start = None
                self.perturbation_direction = None
                self.cycle = cycle
            ready = (self.repeated.phase == "carry_to_unload"
                     and self.repeated.carry_step is not None
                     and retained * self.mass >= self.minimum_kg
                     and state[3] <= -1.65
                     and self.chain.carry_pitch(state[:4]) <= np.radians(-125))
            distance = float(np.linalg.norm(state[22:24]))
            if not ready:
                if self.perturbation_start is not None:
                    self.failure = "Lost loaded carrying posture during the controlled deviation"
                    self.complete = True
                    return None, "collection_finished"
                return None, "collection_model_prefix"
            active = loaded_overshoot(self.best, state, distance)
            if distance < self.best[0]:
                self.best = distance, float(state[0])
            if self.controlled_overshoot and self.perturbation_start is None and not active:
                bounds = self.scene["canonical_joint_limits_rad"][0]
                if distance <= 2.35 and abs(state[4]) >= .12 and bounds[0] + .45 < state[0] < bounds[1] - .45:
                    self.perturbation_start = float(state[0])
                    self.perturbation_direction = float(np.sign(state[4]))
            if self.perturbation_start is not None and not active:
                moved = abs(float(state[0]) - self.perturbation_start)
                bounds = self.scene["canonical_joint_limits_rad"][0]
                if (moved < .32 and self.perturbation_steps < 24
                        and bounds[0] + .12 < state[0] < bounds[1] - .12):
                    self.perturbation_steps += 1
                    return np.asarray([.28 * self.perturbation_direction, 0., 0., 0.], dtype=np.float32), "collection_perturbation"
                active = distance >= self.best[0] + .30 and moved >= .12
                if not active:
                    self.failure = "Controlled deviation did not produce the intended overshoot"
                    self.complete = True
                    return None, "collection_finished"
            if not active:
                return None, "collection_model_prefix"
            self.handoff = {
                "step": step, "cycle": cycle + 1, "bucket_source_particles": retained,
                "baseline_delivered_particles": self.repeated.transfer["payload_particles"],
                "distance_to_unload_target_m": distance, "closest_distance_m": self.best[0],
                "swing_pos_rad": float(state[0]), "swing_rate_rad_s": float(state[4]),
                "controlled_perturbation_steps": self.perturbation_steps,
            }
            atomic_json(self.folder / "recovery_handoff.json", self.handoff)
            self.scene["observation_context"] = dict(reply["observation_context"])
            self.expert = RecoveryExpert(self.model, self.scene, state)
            self.expert_start = step
        try:
            action, details = self.expert.action(
                state, reply["joint_command_positions"], self.repeated.transfer, reply["bucket_landmarks_world"])
        except RecoveryFailure as error:
            self.failure = str(error)
            self.complete = True
            return None, "collection_finished"
        self.expert_actions += 1
        return action, "collection_expert"

    def observe(self, post, source, clipped_dimensions):
        if source != "collection_expert":
            return
        self.contiguous_unclipped &= clipped_dimensions == 0
        delivered_kg = (self.repeated.transfer["payload_particles"]
                        - self.handoff["baseline_delivered_particles"]) * self.mass
        valid_deposit = (self.repeated.dump_step is not None
                         and self.repeated.dump_step >= self.expert_start
                         and post["joint_positions"][3] >= -.8
                         and delivered_kg >= self.minimum_kg)
        self.stable_deposits = self.stable_deposits + 1 if valid_deposit else 0
        if self.stable_deposits >= 5:
            self.accepted = self.contiguous_unclipped
            self.complete = True

    def save(self, status):
        accepted = self.accepted and status["status"] == "completed" and not status.get("error")
        end = self.expert_start + self.expert_actions if self.expert_start is not None else None
        atomic_json(self.folder / "recovery_supervision.json", {
            "accepted_for_recovery_dataset": accepted, "training_only": True,
            "model_evaluation": False, "handoff": self.handoff,
            "positive_label_range_half_open": [self.expert_start, end] if accepted else None,
            "source_for_positive_labels": "collection_expert", "other_steps_have_zero_loss_weight": True,
            "minimum_additional_deposit_kg": self.minimum_kg,
            "unclipped_expert_actions": self.contiguous_unclipped,
            "reason": self.failure or ("accepted" if accepted else "no accepted physical recovery"),
            "expert_transitions": [] if self.expert is None else self.expert.transitions,
        })
