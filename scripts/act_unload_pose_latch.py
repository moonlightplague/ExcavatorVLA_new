"""Bounded ACT handoff waiting and an anchored parent-joint unloading target."""
import numpy as np


class UnloadPoseLatch:
    """First development candidate; full-opening anchoring was not accepted.

    Retained to reproduce the first trial. See the handoff-wait variant below.
    """
    def __init__(self,dt,action_limits,empty_fill=.002,wait_steps=20,open_steps=50):
        self.dt=float(dt);self.limits=np.asarray(action_limits,dtype=float)
        if (not np.isfinite(self.dt) or self.dt<=0 or self.limits.shape!=(4,)
                or not np.isfinite(self.limits).all() or np.any(self.limits<=0) or min(wait_steps,open_steps)<1):
            raise ValueError('Invalid unloading latch timing or action limits')
        self.empty_fill=float(empty_fill);self.maximum_wait=int(wait_steps);self.maximum_open=int(open_steps)
        self.active=False;self.goal=None;self.waited=0;self.opened=0;self.empty=0;self.phase=None
        self.trigger=None

    def update(self,state,model_action,chosen,command,over_bed,recovery_in_charge,verified_handoff=False):
        state=np.asarray(state,dtype=float);model_action=np.asarray(model_action,dtype=float)
        command=np.asarray(command,dtype=float);chosen=np.asarray(chosen,dtype=float).copy()
        if (state.shape!=(28,) or model_action.shape!=(4,) or chosen.shape!=(4,) or command.shape!=(4,)
                or not all(np.isfinite(v).all() for v in (state,model_action,chosen,command))):
            raise ValueError('Invalid unload latch state/action/target')
        loaded=bool(state[26]>=self.empty_fill);intent=bool(model_action[3]>.12)
        started=False;release=None
        if not self.active and loaded and over_bed and not recovery_in_charge and (verified_handoff or intent):
            self.active=True;self.goal=state[:3].copy();self.waited=0;self.opened=0;self.empty=0
            self.phase='open' if intent else 'wait_for_act_opening'
            self.trigger='verified_secure_align_handoff' if verified_handoff else 'model_opening_intent'
            started=True
        if self.active:
            self.empty=self.empty+1 if not loaded else 0
            if recovery_in_charge:release='recovery_in_charge'
            elif self.empty>=3:release='payload_empty'
            elif self.phase=='wait_for_act_opening' and intent:self.phase='open'
            if release is None:
                if self.phase=='wait_for_act_opening':
                    if self.waited>=self.maximum_wait:release='act_opening_wait_timeout'
                    else:self.waited+=1
                elif self.opened>=self.maximum_open:release='opening_hold_timeout'
                else:self.opened+=1
            if release is not None:self.active=False
        if self.active:
            # Cancel the old integrated target error and keep the same goal.
            # Replacing this goal with each new measured pose would permit drift.
            chosen[:3]=np.clip((self.goal-command[:3])/self.dt,-self.limits[:3],self.limits[:3])
        trace=dict(enabled=True,started=started,active=self.active,held_parent_actions=self.active,
            bucket_action_unchanged=self.active,loaded=loaded,opening_intent=intent,over_bed=bool(over_bed),
            steps=self.waited+self.opened,release_reason=release,phase=self.phase,trigger=self.trigger,
            wait_steps=self.waited,opening_steps=self.opened,
            latched_parent_goal=self.goal.tolist() if self.goal is not None else None,
            measured_parent_error=(state[:3]-self.goal).tolist() if self.goal is not None else None)
        return chosen,trace


def patch_controller(code):
    """Reproduce the rejected first trial in a copied legacy controller."""
    if 'UnloadPoseLatch' in code:raise ValueError('Controller already has a pose latch')
    declaration='import numpy as np\n'
    original_init=('            self.unload_pose_hold_active = False\n'
        '            self.unload_pose_hold_steps = 0\n'
        '            self.unload_empty_steps = 0\n')
    if code.count(declaration)!=1 or code.count(original_init)!=1:
        raise ValueError('Unexpected frozen controller initialization')
    code=code.replace(declaration,declaration+'from act_unload_pose_latch import UnloadPoseLatch\n')
    code=code.replace(original_init,'            self.unload_pose_latch = UnloadPoseLatch(self.dt, self.action_limits, self.settings.empty_fill_fraction)\n')
    begin='            loaded = bool(state[26] >= self.settings.empty_fill_fraction)\n'
    end="            closure_phase = trace.get('phase')"
    if code.count(begin)!=1 or code.count(end)!=1:raise ValueError('Unexpected frozen hold block')
    start=code.index(begin);stop=code.index(end,start)
    replacement='''            over_bed, _ = self.release_geometry(state, landmarks)
            recovery_in_charge = self.recovery is not None and not self.recovery.finished
            chosen, hold_trace = self.unload_pose_latch.update(
                state, action, chosen, command_positions, over_bed, recovery_in_charge,
                verified_handoff=bool(trace.get('handback') and trace.get('phase') == 'completed_secure_align'))
            if hold_trace['active']:
                trace['override'] = True
            trace['unload_pose_hold'] = hold_trace
'''
    return code[:start]+replacement+code[stop:]


class UnloadHandoffWait:
    """Anchor only the gap before the existing unloading hold takes charge."""
    def __init__(self,dt,action_limits,empty_fill=.002,maximum_steps=20):
        self.dt=float(dt);self.limits=np.asarray(action_limits,dtype=float)
        self.empty_fill=float(empty_fill);self.maximum_steps=int(maximum_steps)
        self.active=False;self.goal=None;self.steps=0;self.empty=0
        if not np.isfinite(self.dt) or self.dt<=0 or self.limits.shape!=(4,) or not np.isfinite(self.limits).all() or np.any(self.limits<=0) or self.maximum_steps<1:
            raise ValueError('Invalid handoff wait configuration')

    def update(self,state,chosen,command,over_bed,recovery_in_charge,verified_handoff,legacy_hold_active):
        state=np.asarray(state,dtype=float);command=np.asarray(command,dtype=float)
        if state.shape!=(28,) or command.shape!=(4,) or not np.isfinite(state).all() or not np.isfinite(command).all():
            raise ValueError('Invalid handoff wait observation')
        loaded=bool(state[26]>=self.empty_fill);started=False;release=None
        if legacy_hold_active:
            if self.active:release='legacy_unloading_hold_started'
            self.active=False
        elif not self.active and verified_handoff and loaded and over_bed and not recovery_in_charge:
            self.goal=state[:3].copy();self.active=True;self.steps=0;self.empty=0;started=True
        if self.active:
            self.empty=self.empty+1 if not loaded else 0
            if recovery_in_charge:release='recovery_in_charge'
            elif self.empty>=3:release='payload_empty'
            elif self.steps>=self.maximum_steps:release='act_opening_wait_timeout'
            if release is not None:self.active=False
            else:self.steps+=1
        if self.active:
            chosen=np.asarray(chosen).copy()
            chosen[:3]=np.clip((self.goal-command[:3])/self.dt,-self.limits[:3],self.limits[:3])
        return chosen,dict(enabled=True,started=started,active=self.active,steps=self.steps,
            release_reason=release,goal=self.goal.tolist() if self.goal is not None else None,
            legacy_hold_unchanged=True,bucket_command_unchanged=True)


def patch_wait_controller(code):
    """Add pre-opening waiting while preserving the legacy hold block verbatim."""
    if 'UnloadHandoffWait' in code:raise ValueError('Controller already has a handoff wait')
    init='            self.attempted = False\n'
    marker="            closure_phase = trace.get('phase')"
    if code.count(init)!=1 or code.count(marker)!=1 or code.count('import numpy as np\n')!=1:
        raise ValueError('Unexpected frozen controller for handoff waiting')
    code=code.replace('import numpy as np\n','import numpy as np\nfrom act_unload_pose_latch import UnloadHandoffWait\n')
    code=code.replace(init,init+'            self.unload_handoff_wait = UnloadHandoffWait(self.dt, self.action_limits, self.settings.empty_fill_fraction)\n')
    insertion='''            chosen, wait_trace = self.unload_handoff_wait.update(
                state, chosen, command_positions, over_bed, recovery_in_charge,
                verified_handoff=bool(trace.get('handback') and trace.get('phase') == 'completed_secure_align'),
                legacy_hold_active=hold_applied)
            if wait_trace['active']:
                trace['override'] = True
            trace['unload_handoff_wait'] = wait_trace
'''
    return code.replace(marker,insertion+marker)
