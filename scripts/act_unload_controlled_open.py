"""Experimental bounded opening inside unload_pose_hold; not the default policy."""
import math


def allow_controlled_open(event,loaded,over_bed,legacy_active,enabled=True):
    """Require the existing recovery's measured, contained opening-arc check."""
    residual=event.get('contained_opening_arc_residual_m')
    return bool(enabled and loaded and over_bed and not legacy_active
        and event.get('handback') and event.get('phase')=='completed_secure_align'
        and isinstance(residual,(float,int)) and math.isfinite(residual) and 0<=residual<=.015)


def bucket_open_velocity(command_bucket,dt,action_limit,target=-.4,speed=.8):
    """Open toward the verified arc endpoint without commanding beyond it."""
    if not all(math.isfinite(v) for v in (command_bucket,dt,action_limit,target,speed)) or min(dt,action_limit,speed)<=0:
        raise ValueError('Invalid controlled unloading command')
    return min(max((target-command_bucket)/dt,0.),speed,action_limit)


def patch_controller(code):
    if 'allow_controlled_open' in code:raise ValueError('Controlled opening already installed')
    imports='import numpy as np\n'
    init='\n            self.unload_empty_steps = 0\n'
    trigger='                hold_started = True\n'
    release='            hold_release_reason = None\n'
    marker="            closure_phase = trace.get('phase')"
    if any(code.count(s)!=1 for s in (imports,init,trigger,release,marker)):
        raise ValueError('Unexpected frozen unloading controller')
    code=code.replace(imports,imports+'from act_unload_controlled_open import allow_controlled_open, bucket_open_velocity\n')
    code=code.replace(init,init+'            self.unload_controlled_open_active = False\n')
    # Every ordinary ACT-triggered activation retains the old behavior.
    code=code.replace(trigger,'                self.unload_controlled_open_active = False\n'+trigger)
    entry='''            controlled_started = allow_controlled_open(
                trace, loaded, over_bed, self.unload_pose_hold_active,
                enabled=self.unload_pose_hold_enabled)
            if controlled_started:
                self.unload_pose_hold_active = True
                self.unload_pose_hold_steps = 0
                self.unload_empty_steps = 0
                self.unload_controlled_open_active = True
                hold_started = True
'''
    code=code.replace(release,entry+release)
    apply='''            controlled_applied = hold_applied and self.unload_controlled_open_active
            if controlled_applied:
                chosen[3] = bucket_open_velocity(command_positions[3], self.dt, self.action_limits[3])
                trace['unload_pose_hold']['bucket_action_unchanged'] = False
            trace['unload_pose_hold']['controlled_open'] = dict(
                started=controlled_started, active=controlled_applied,
                target_bucket_rad=-.4, maximum_bucket_speed_rad_s=.8,
                bucket_command_source='bounded_unload_controller' if controlled_applied else 'ACT',
                contained_arc_required=True)
            if not hold_applied:
                self.unload_controlled_open_active = False
'''
    return code.replace(marker,apply+marker)
