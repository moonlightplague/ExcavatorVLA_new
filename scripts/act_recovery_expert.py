"""Online geometry/feedback recovery supervisor, separate from ACT inference."""
import math
import numpy as np

from excavator_reachability import AuthoredExcavatorChain, TruckBedColumn, find_endpoint


class RecoveryFailure(RuntimeError):
    """A physical expert recovery failed; distinct from bridge/configuration errors."""


class RecoveryExpert:
    version = 'geometry_feedback_recovery_v3_progressive_curl'

    def __init__(self, model, scene, initial_state, dt=.1):
        self.chain=AuthoredExcavatorChain(model)
        self.scene=scene
        self.bed=TruckBedColumn(scene,.15)
        self.bounds=np.asarray(scene['canonical_joint_limits_rad'],dtype=float)
        self.dt=dt
        self.phase='secure'
        self.phase_step=0
        self.total_steps=0
        self.transitions=[]
        self.initial_q=np.asarray(initial_state[:4],dtype=float)
        self.minimum_bucket=self.bounds[3,0]
        self.closed_pitch=math.radians(-135)
        self.secure=self.initial_q.copy()
        self.secure[3]=self.minimum_bucket
        self.secure[2]=np.clip(min(self.secure[2],self.closed_pitch-self.secure[1]-self.secure[3]),*self.bounds[2])
        self.lift=self.lift_goal(self.secure)
        self.dump,self.plan=self.dump_goal(self.lift)
        self.slew=self.lift.copy();self.slew[0]=self.dump[0]
        self.empty_observations=0
        self.curl_path=[];self.curl_index=0
        if self.initial_q[3]>-.5 and 'bucket_collision_vertices' in model:
            self.collision_vertices=np.unique(np.asarray(model['bucket_collision_vertices'],dtype=float),axis=0)
            self.curl_path=self.progressive_curl(self.initial_q)
            self.phase='unwind'

    def minimum_bucket_z(self,q):
        transform=self.chain.bucket_transform(q)
        return float((self.collision_vertices@transform[:3,:3].T+transform[:3,3])[:,2].min())

    def progressive_curl(self,initial,dip_m=0.):
        from scipy.optimize import least_squares
        tip=self.chain.point(initial,'tip')
        dig=np.asarray(self.scene['observation_context']['dig_target_xyz'])
        previous=initial.copy();path=[]
        lower=self.bounds[:3,0].copy();upper=self.bounds[:3,1].copy()
        upper[2]=min(upper[2],initial[2]+(.35 if dip_m>0 else 0.))
        initial_pitch=float(self.chain.carry_pitch(initial))
        for index,bucket in enumerate(np.linspace(initial[3],self.minimum_bucket,23)[1:],1):
            fraction=index/22
            target=tip.copy();target[:2]=(1-fraction)*tip[:2]+fraction*dig[:2]
            target[2]=tip[2]+.25*fraction-float(dip_m)*math.sin(math.pi*fraction)
            def residual(parent):
                q=np.r_[parent,bucket];point=self.chain.point(q,'tip')
                floor=max(0.,.035-self.minimum_bucket_z(q))
                pitch_limit=(1-fraction)*initial_pitch+fraction*self.closed_pitch
                pitch=max(0.,float(self.chain.carry_pitch(q))-pitch_limit)
                z_weight=1. if dip_m>0 else .2
                return np.r_[point[:2]-target[:2],z_weight*(point[2]-target[2]),30*floor,10*pitch,.005*(parent-previous[:3])]
            seed=np.clip(previous[:3],lower+1e-8,upper-1e-8)
            result=least_squares(residual,seed,bounds=(lower,upper),
                max_nfev=100,ftol=1e-9,xtol=1e-9,gtol=1e-9)
            q=np.r_[result.x,bucket]
            if self.minimum_bucket_z(q)<.02:raise ValueError('Progressive curl endpoint violates ground clearance')
            # Keep intermediate interpolation above the floor too. This is a
            # bucket/ground check, not a full rigid-obstacle collision proof.
            samples=previous+(q-previous)*np.linspace(0,1,7)[:,None]
            if min(self.minimum_bucket_z(sample) for sample in samples)<.005:
                raise ValueError('Progressive curl interpolation violates ground clearance')
            path.append(q);previous=q
        if float(self.chain.carry_pitch(path[-1]))>math.radians(-130):
            raise ValueError('Progressive curl terminal pose is not gravity-closed')
        return path

    def lift_goal(self,q):
        boom=np.linspace(max(q[1],self.bounds[1,0]),self.bounds[1,1],55)
        arm=np.linspace(self.bounds[2,0],min(.3,self.bounds[2,1]),65)
        parents=np.stack(np.meshgrid(boom,arm,indexing='ij'),axis=-1).reshape(-1,2)
        options=np.column_stack((np.full(len(parents),q[0]),parents,np.full(len(parents),self.minimum_bucket)))
        load=self.chain.point(options,'load');tip=self.chain.point(options,'tip')
        top=float(self.bed.top_corners_world()[:,2].max())
        valid=(load[:,2]>=top+1.25)&(tip[:,2]>=top+.8)&(self.chain.carry_pitch(options)<=self.closed_pitch)
        valid &= np.linalg.norm(load[:,:2]-self.chain.swing_anchor[:2],axis=1)>=3
        if not valid.any():raise ValueError('No high loaded-lift endpoint found')
        candidates=options[valid];points=load[valid]
        score=(points[:,2]-(top+1.35))**2+.15*((candidates[:,1:3]-q[1:3])**2).sum(1)
        return candidates[int(score.argmin())]

    def opening_points(self,q):
        sequence=np.repeat(np.asarray(q,dtype=float)[None,:],10,axis=0)
        sequence[:,3]=np.linspace(self.minimum_bucket,-.4,10)
        return np.vstack((self.chain.point(q,'load'),self.chain.point(sequence,'pour')))

    def dump_goal(self,reference):
        from scipy.optimize import least_squares
        witness=find_endpoint(self.chain,self.scene,'carry',max_starts=4)
        if witness['status']!='endpoint_found_path_unverified':
            raise ValueError('Carry endpoint unresolved; not a mechanical-unreachability claim')
        first=np.asarray(witness['q_rad'])
        top=float(self.bed.top_corners_world()[:,2].max())
        seeds=[first[:3],np.array([first[0],reference[1],reference[2]])]
        for boom,arm in ((.75,-1.05),(.9,-1.2),(.65,-.95),(.85,-1.0)):
            seeds.append(np.array([first[0],boom,arm]))

        def residual(parent):
            q=np.r_[parent,self.minimum_bucket]
            points=self.opening_points(q)
            spatial=self.bed.residual(points).reshape(-1)
            pitch=max(0.,float(self.chain.carry_pitch(q))-math.radians(-130))
            height=.04*(points[:,2].mean()-(top+1.0))
            regular=.002*(parent-first[:3])
            return np.r_[10*spatial,5*pitch,height,regular]

        best=None
        for seed in seeds:
            seed=np.clip(seed,self.bounds[:3,0]+1e-6,self.bounds[:3,1]-1e-6)
            solved=least_squares(residual,seed,bounds=(self.bounds[:3,0],self.bounds[:3,1]),
                max_nfev=160,ftol=1e-9,xtol=1e-9,gtol=1e-9)
            q=np.r_[solved.x,self.minimum_bucket]
            points=self.opening_points(q)
            error=float(np.linalg.norm(self.bed.residual(points),axis=1).max())
            pitch=float(self.chain.carry_pitch(q))
            if error>.01 or pitch>math.radians(-130)+.001:continue
            score=float(((q[:3]-reference[:3])**2).sum())+.1*(points[:,2].mean()-(top+1))**2
            if best is None or score<best[0]:
                best=(score,q,dict(opening_arc_max_residual_m=error,closed_pitch_deg=math.degrees(pitch),
                    opening_points_world=points.tolist(),endpoint_only=True,collision_path_verified=False))
        if best is None:raise ValueError('No contained opening arc found in finite IK search')
        return best[1],best[2]

    def switch(self,phase):
        self.transitions.append(dict(at_expert_step=self.total_steps,from_phase=self.phase,to_phase=phase))
        self.phase=phase;self.phase_step=0

    def velocity(self,goal,command,speeds):
        delta=np.asarray(goal)-np.asarray(command)
        scale=max(1.,float(np.max(np.abs(delta)/np.asarray(speeds)))/self.dt)
        return delta/(scale*self.dt)

    def action(self,state,command,particles,landmarks):
        q=np.asarray(state[:4],dtype=float)
        if not np.isfinite(q).all():raise ValueError('Nonfinite measured recovery state')
        for name in ('tip','load','pour'):
            if np.linalg.norm(self.chain.point(q,name)-np.asarray(landmarks[name]))>.02:
                raise ValueError('Authored FK does not match current measured bucket landmarks')
        retained=int(particles['retained_particles'])
        self.empty_observations=self.empty_observations+1 if retained<=10 and particles['payload_particles']<100 else 0
        if self.empty_observations>=3 and self.phase not in ('open','settle'):
            raise RecoveryFailure('Recovery emptied the bucket before unloading; no expert labels accepted')
        pitch=float(self.chain.carry_pitch(q))
        if self.phase=='unwind':
            if abs(float(command[3])-q[3])<=.09:
                self.curl_path=self.progressive_curl(q);self.curl_index=0
                self.switch('progressive_curl')
                self.transitions[-1]['current_state_curl_path']=[point.tolist() for point in self.curl_path]
        if self.phase=='progressive_curl':
            if np.max(np.abs(q-self.curl_path[self.curl_index]))<.035:
                self.curl_index+=1
                if self.curl_index==len(self.curl_path):
                    if q[3]>-1.3 or pitch>math.radians(-125):raise RecoveryFailure('Progressive curl did not close the loaded bucket')
                    self.lift=self.lift_goal(q);self.slew=self.lift.copy();self.slew[0]=self.dump[0]
                    self.switch('lift')
        if self.phase=='secure':
            if q[3]<=self.minimum_bucket+.06 and pitch<=math.radians(-130):self.switch('lift')
            elif self.phase_step>=40:self.switch('lift_assist')
        if self.phase in ('lift','lift_assist'):
            if np.max(np.abs(q-self.lift))<.04:
                if q[3]>-1.3 or pitch>math.radians(-125):raise RecoveryFailure('Lift reached without a gravity-closed bucket')
                self.switch('slew')
        elif self.phase=='slew' and abs(q[0]-self.slew[0])<.025:self.switch('align')
        elif self.phase=='align':
            residual=float(np.linalg.norm(self.bed.residual(self.opening_points(q)),axis=1).max())
            if np.max(np.abs(q[:3]-self.dump[:3]))<.025 and residual<.015:self.switch('open')
        goal={'secure':self.secure,'lift':self.lift,'lift_assist':self.lift,
              'slew':self.slew,'align':self.dump,'open':self.dump}.get(self.phase,self.dump).copy()
        if self.phase=='unwind':
            goal=q.copy();goal[3]=max(self.minimum_bucket,q[3]-.05)
        if self.phase=='progressive_curl':goal=self.curl_path[self.curl_index].copy()
        if self.phase=='open':goal[3]=-.4
        speeds=np.array([.28,.24,.24,.4 if self.phase!='open' else .2])
        if self.phase=='secure':speeds[:3]=.12
        if self.phase=='lift_assist':speeds[1:3]=.14
        if self.phase=='unwind':speeds=np.array([.2,.2,.2,4.])
        if self.phase=='progressive_curl':speeds=np.array([.15,.18,.18,.25])
        action=self.velocity(goal,command,speeds)
        phase_id={'unwind':4,'progressive_curl':4,'secure':4,'lift_assist':5,'lift':7,'slew':8,'align':8,'open':9}[self.phase]
        details=dict(version=self.version,phase=self.phase,stage_id=phase_id,expert_step=self.total_steps,
            measured_pitch_deg=math.degrees(pitch),measured_retained=retained,goal_q=goal.tolist(),
            max_measured_joint_error=float(np.max(np.abs(q-goal))),feedback=True)
        if self.phase=='progressive_curl':details.update(curl_waypoint=self.curl_index,
            actual_bucket_min_z=self.minimum_bucket_z(q),goal_bucket_min_z=self.minimum_bucket_z(goal))
        self.total_steps+=1;self.phase_step+=1
        if self.phase_step>180:raise RecoveryFailure(f'Expert phase stalled: {self.phase}')
        return action.astype(np.float32),details
