"""Explicit, bounded checkpoint transfer to a tighter execution action profile."""
from excavator_vlm.data.action_limits import action_contract, checkpoint_contract


def validate_action_transfer(manifest, info, *, allow_tighter=False):
    source, target = checkpoint_contract(manifest), action_contract(info)
    if source == target:
        return None
    if not allow_tighter:
        raise ValueError('Checkpoint and evaluation dataset use different action-limit profiles')
    if (source['kind'] != target['kind'] or source['units'] != target['units']
            or any(t > s for s, t in zip(source['limits_rad_s'], target['limits_rad_s']))):
        raise ValueError('Diagnostic transfer may only tighten limits for the same action semantics')
    return {'mode': 'explicit_tighter_execution_limits', 'checkpoint_action_contract': source,
            'execution_action_contract': target,
            'scope': 'Cross-domain diagnostic; checkpoint normalization/weights preserved and physical commands clipped to target limits.'}
