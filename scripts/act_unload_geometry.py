"""Experimental measured aperture alignment before bounded unloading.

The policy observation landmarks and the selected 57.5% controller stay intact.
"""
import math
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares
from excavator_reachability import AuthoredExcavatorChain, TruckBedColumn


def aperture_from_mesh(model):
    """Conservative aperture of the verified, open U-shaped bucket asset."""
    vertices=np.unique(np.asarray(model['bucket_collision_vertices'],dtype=float),axis=0)
    axis={'X':0,'Y':1,'Z':2}[model['joints'][3]['axis']]
    plane=[i for i in range(3) if i!=axis]
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all():
        raise ValueError('Invalid bucket collision geometry')
    hull=ConvexHull(vertices[:,plane]);profile=vertices[hull.vertices][:,plane]
    lengths=np.linalg.norm(np.roll(profile,-1,axis=0)-profile,axis=1)
    order=np.argsort(lengths);index=int(order[-1]);a=profile[index];b=profile[(index+1)%len(profile)]
    width=float(np.ptp(vertices[:,axis]))
    if lengths[index]<.5 or width<.3 or lengths[index]<1.5*lengths[order[-2]]:
        raise ValueError('Aperture is ambiguous; asset-specific calibration required')
    corners=np.zeros((4,3))
    corners[:,plane]=np.asarray([a,b,b,a])
    corners[:,axis]=[vertices[:,axis].min(),vertices[:,axis].min(),vertices[:,axis].max(),vertices[:,axis].max()]
    return corners,dict(hinge_axis=model['joints'][3]['axis'],width_m=width,
        length_m=float(lengths[index]),corners_local=corners.tolist(),
        method='Verified open-bucket side-profile chord and full authored collision width')


class ApertureAlignment:
    target_bucket=-.4
    parent_delta=np.array([.12,.12,.16])

    def __init__(self,model,scene,dt=.1):
        self.chain=AuthoredExcavatorChain(model)
        self.bed=TruckBedColumn(scene,.15)
        self.bounds=np.asarray(scene['canonical_joint_limits_rad'],dtype=float)
        self.vertices=np.unique(np.asarray(model['bucket_collision_vertices'],dtype=float),axis=0)
        self.corners,self.calibration=aperture_from_mesh(model)
        self.dt=float(dt)
        self.top=float(self.bed.top_corners_world()[:,2].max())

    def sweep(self,q,samples=17):
        q=np.asarray(q,dtype=float)
        sequence=np.repeat(q[None,:],samples,axis=0)
        sequence[:,3]=np.linspace(q[3],max(q[3],self.target_bucket),samples)
        transforms=self.chain.bucket_transform(sequence)
        return (np.einsum('nij,vj->nvi',transforms[:,:3,:3],self.corners)+transforms[:,None,:3,3]).reshape(-1,3)

    def clearance(self,q):
        points=self.sweep(q);local=self.bed.local(points)
        xy=np.minimum((local[:,:2]-self.bed.minimum[:2])*self.bed.scales[:2],
                      (self.bed.maximum[:2]-local[:,:2])*self.bed.scales[:2])
        z=(local[:,2]-self.bed.maximum[2])*self.bed.scales[2]
        return dict(minimum_xy_m=float(xy.min()),minimum_height_m=float(z.min()))

    def safe(self,q,margin=.10):
        c=self.clearance(q)
        return c['minimum_xy_m']>=margin and c['minimum_height_m']>=margin

    def bucket_min_z(self,q):
        transform=self.chain.bucket_transform(q)
        return float((self.vertices@transform[:3,:3].T+transform[:3,3])[:,2].min())

    def opening_min_z(self,q):
        sequence=np.repeat(np.asarray(q)[None,:],17,axis=0)
        sequence[:,3]=np.linspace(q[3],max(q[3],self.target_bucket),17)
        t=self.chain.bucket_transform(sequence)
        return float((np.einsum('nij,vj->nvi',t[:,:3,:3],self.vertices)+t[:,None,:3,3])[:,:,2].min())

    def plan(self,reference):
        reference=np.asarray(reference,dtype=float)
        closed=reference.copy();closed[3]=self.bounds[3,0]
        lower=np.maximum(self.bounds[:3,0],reference[:3]-self.parent_delta)
        upper=np.minimum(self.bounds[:3,1],reference[:3]+self.parent_delta)
        if (self.safe(closed,.15) and self.opening_min_z(closed)>=self.top+.05
                and self.chain.carry_pitch(closed)<=math.radians(-130)+.001):
            return closed,dict(already_contained=True,clearance=self.clearance(closed))
        def residual(parent):
            q=np.r_[parent,closed[3]]
            spatial=self.bed.residual(self.sweep(q)).reshape(-1)
            pitch=max(0.,float(self.chain.carry_pitch(q))-math.radians(-130))
            return np.r_[10*spatial,10*pitch,.03*(parent-reference[:3])]
        seeds=[reference[:3],np.clip(reference[:3]+[0,.04,-.04],lower+1e-6,upper-1e-6)]
        best=None
        for seed in seeds:
            result=least_squares(residual,np.clip(seed,lower+1e-7,upper-1e-7),bounds=(lower,upper),
                max_nfev=100,ftol=1e-9,xtol=1e-9,gtol=1e-9)
            goal=np.r_[result.x,closed[3]]
            if not self.safe(goal,.12) or self.chain.carry_pitch(goal)>math.radians(-130)+.001:continue
            path=reference+(goal-reference)*np.linspace(0,1,21)[:,None]
            minimum_z=min(self.bucket_min_z(q) for q in path)
            if minimum_z<self.top+.05 or self.opening_min_z(goal)<self.top+.05:continue
            score=float(np.linalg.norm(goal[:3]-reference[:3]))
            if best is None or score<best[0]:best=(score,goal,minimum_z)
        if best is None:raise ValueError('No contained aperture within bounded local correction')
        return best[1],dict(already_contained=False,parent_delta_rad=(best[1][:3]-reference[:3]).tolist(),
            clearance=self.clearance(best[1]),closed_path_minimum_z_m=best[2])


class GeometryUnload:
    """Contained aperture -> measured alignment -> held, bounded opening."""
    version='aperture_geometry_v2_measured_servo'
    parent_speeds=np.array([.12,.12,.16])

    def __init__(self,model,scene,dt,limits,empty_fill=.002):
        self.geometry=ApertureAlignment(model,scene,dt)
        self.dt=float(dt);self.limits=np.asarray(limits,dtype=float);self.empty_fill=float(empty_fill)
        self.active=False;self.attempted=False;self.phase=None;self.goal=None
        self.reference=None
        self.steps=0;self.align_steps=0;self.empty=0;self.stable=0;self.plan_details=None

    def update(self,state,chosen,command,landmarks,event,legacy_active,step,enabled=True):
        state=np.asarray(state,dtype=float);command=np.asarray(command,dtype=float)
        started=False;reason=None;error=None;handback=False;blocked=False
        verified=bool(event.get('handback') and event.get('phase')=='completed_secure_align')
        residual=event.get('contained_opening_arc_residual_m',math.inf)
        if (not self.attempted and enabled and verified and not legacy_active and state[26]>=self.empty_fill
                and isinstance(residual,(float,int)) and math.isfinite(residual) and 0<=residual<=.015):
            self.attempted=True
            if step>450:reason='insufficient_episode_budget'
            else:
                try:
                    self.goal,self.plan_details=self.geometry.plan(state[:4])
                    self.reference=state[:4].copy()
                    low=np.maximum(self.geometry.bounds[:3,0],self.reference[:3]-self.geometry.parent_delta)
                    high=np.minimum(self.geometry.bounds[:3,1],self.reference[:3]+self.geometry.parent_delta)
                    if np.any(command[:3]<low-1e-6) or np.any(command[:3]>high+1e-6):
                        raise ValueError('Current integrated target is outside the bounded correction')
                    self.active=True;self.phase='align_closed';started=True
                    self.steps=0;self.align_steps=0;self.empty=0;self.stable=0
                except ValueError as problem:reason='local_plan_unresolved';error=str(problem)
        if self.active:
            measured_error=max(float(np.linalg.norm(self.geometry.chain.point(state[:4],k)-np.asarray(landmarks[k])))
                               for k in ('tip','load','pour'))
            self.empty=self.empty+1 if state[26]<self.empty_fill else 0
            self.steps+=1
            if measured_error>.02:reason='fk_inconsistent'
            elif self.empty>=3:reason='payload_empty'
            elif self.steps>=50:reason='step_timeout'
            if reason is None and self.phase=='align_closed':
                self.align_steps+=1
                aligned=bool(np.max(np.abs(state[:3]-self.goal[:3]))<.02
                    and abs(state[3]-self.goal[3])<.03 and np.max(np.abs(state[4:7]))<=.10
                    and self.geometry.safe(state[:4])
                    and self.geometry.chain.carry_pitch(state[:4])<=math.radians(-130)+.001)
                self.stable=self.stable+1 if aligned else 0
                if self.stable>=2:self.phase='open_contained'
                elif self.align_steps>=20:reason='alignment_timeout'
            if reason is not None:
                self.active=False;handback=True
        clearance=None
        if self.active:
            chosen=np.asarray(chosen,dtype=float).copy()
            # Actions integrate commanded joint targets. Integrating measured
            # pose error also compensates the persistent gravity/servo offset.
            velocity=2.*(self.goal[:3]-state[:3])
            scale=max(1.,float(np.max(np.abs(velocity)/self.parent_speeds)))
            velocity=velocity/scale
            low=np.maximum(self.geometry.bounds[:3,0],self.reference[:3]-self.geometry.parent_delta)
            high=np.minimum(self.geometry.bounds[:3,1],self.reference[:3]+self.geometry.parent_delta)
            chosen[:3]=np.clip(velocity,(low-command[:3])/self.dt,(high-command[:3])/self.dt)
            chosen[3]=np.clip((self.goal[3]-command[3])/self.dt,-.8,0.)
            if self.phase=='open_contained':
                requested=min(max((-.4-command[3])/self.dt,0.),.8,self.limits[3])
                predicted=command.copy();predicted[:3]+=chosen[:3]*self.dt;predicted[3]+=requested*self.dt
                clearance=self.geometry.clearance(state[:4])
                if self.geometry.safe(state[:4]) and self.geometry.safe(predicted):chosen[3]=requested
                else:chosen[3]=0.;blocked=True
            chosen=np.clip(chosen,-self.limits,self.limits)
        trace=dict(version=self.version,enabled=bool(enabled),attempted=self.attempted,started=started,active=self.active,
            phase=self.phase,steps=self.steps,alignment_steps=self.align_steps,stable_observations=self.stable,
            release_reason=reason,error=error,handback=handback,opening_blocked=blocked,
            parent_goal=self.goal[:3].tolist() if self.goal is not None else None,
            plan=self.plan_details,measured_aperture_clearance=clearance)
        return chosen,trace


def patch_controller(code):
    """Patch only the copied controller; regular ACT-triggered holds survive."""
    if 'GeometryUnload' in code:raise ValueError('Aperture controller already installed')
    imports='import numpy as np\n';init='            self.attempted = False\n'
    trigger='if (self.unload_pose_hold_enabled and not self.unload_pose_hold_active and loaded and opening_intent and over_bed'
    marker="            closure_phase = trace.get('phase')"
    if any(code.count(s)!=1 for s in (imports,init,trigger,marker)):
        raise ValueError('Unexpected selected unloading controller')
    code=code.replace(imports,imports+'from act_unload_geometry import GeometryUnload\n')
    code=code.replace(init,init+'            self.unload_geometry = GeometryUnload(model, scene, self.dt, self.action_limits, self.settings.empty_fill_fraction)\n')
    code=code.replace(trigger,trigger.replace('and not self.unload_pose_hold_active','and not self.unload_geometry.active and not self.unload_pose_hold_active'))
    addition='''            chosen, geometry_trace = self.unload_geometry.update(
                state, chosen, command_positions, landmarks, trace, hold_applied,
                self.detector.step, enabled=self.unload_pose_hold_enabled)
            trace['unload_pose_hold']['geometry_feedback'] = geometry_trace
            if geometry_trace['active']:
                trace['override'] = True
                trace['unload_pose_hold'].update(started=geometry_trace['started'], active=True,
                    held_parent_actions=geometry_trace['phase']=='open_contained',
                    bucket_action_unchanged=False, steps=geometry_trace['steps'])
            if geometry_trace['handback']:
                trace['handback'] = True
                trace['unload_geometry_handback'] = True
'''
    return code.replace(marker,addition+marker)
