"""Post-rollout observations kept separate from scored policy observations."""
import json


def observe_after_rollout(folder, initial_reply, exchange, steps, scene_seed, video=None):
    """Continue with zero command velocities; existing position targets remain.

    This is not a frozen robot, a new success verdict, or proof of settled sand.
    The caller must freeze scored metrics/timing before entering this function.
    """
    rows = []

    def record(reply, index):
        metric = reply['particle_transfer_metrics']
        if not metric.get('available') or reply['truck_bed_load_metrics']['scene_seed'] != scene_seed:
            raise ValueError('Invalid post-rollout particle observations')
        row = dict(observation=index, particle_transfer_metrics=metric,
                   state=reply['observation_state_28d'],
                   joint_command_positions=reply['joint_command_positions'])
        rows.append(row)
        return row

    record(initial_reply, 0)
    with (folder / 'post_rollout_steps.jsonl').open('x') as stream:
        stream.write(json.dumps(rows[0]) + '\n')
        for index in range(1, steps + 1):
            reply = exchange({'joint_velocities': [0., 0., 0., 0.]})
            row = record(reply, index)
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            if video is not None:
                video(reply, row)
    keys = ('payload_particles', 'retained_particles', 'spillage_particles', 'scooped_particles')
    terminal, final = rows[0]['particle_transfer_metrics'], rows[-1]['particle_transfer_metrics']
    recent = rows[-min(10, len(rows)):]
    return dict(status='completed', action_steps=steps,
                intervention='zero command velocities; hold last integrated targets, not measured joint pose',
                terminal={key: terminal[key] for key in keys},
                final={key: final[key] for key in keys},
                last_ten_observation_ranges={key: max(r['particle_transfer_metrics'][key] for r in recent)
                    - min(r['particle_transfer_metrics'][key] for r in recent) for key in keys},
                note='Outside-bucket-and-truck includes airborne particles. Original task result is unchanged.')
