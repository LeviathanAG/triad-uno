"""Development-only public-belief probe retaining censored reward intervals.

Bounds cover every possible continuation of a sampled state under placement
scoring. They do not validate the approximate posterior or rival model. The
Gaussian sampled-world mean test remains a heuristic, not a real-game promise.
Official referee imports are development-only, never deployed inference.
"""
import hashlib,json,math,random,time
from uno_belief_rollout_probe import Policy as Original,engine,sampled_world,bind,terminal_reward
from uno_public_belief import PreparedSampler,key
from uno_policy_v2 import public_beliefs
from uno_route_policy import action_features
from uno_simulation_fast import snapshot,observe_with_actions,resolve_action
from uno_public_fold_fast import CachedRoutePolicy,PublicFold


def reward_interval(state,seat):
    exact=terminal_reward(state,seat)
    if exact is not None:return exact,exact
    remaining=state.config.position_points[len(state.finished_order):]
    if not remaining:raise ValueError('unfinished seat without remaining placement')
    return min(remaining),max(remaining)


def rollout_interval(world,refs,action,seat,weights,deadline,max_steps):
    state=snapshot(world);fold=PublicFold()
    actors=[CachedRoutePolicy(weights,fold) for _ in range(4)]
    if not engine.step(state,bind(action,refs)).ok:raise RuntimeError('known action invalid in sampled world')
    for _ in range(max_steps):
        exact=terminal_reward(state,seat)
        if exact is not None:return (exact,exact),'exact'
        if time.perf_counter()>=deadline:return reward_interval(state,seat),'deadline'
        observation,actions=observe_with_actions(state,state.current_seat)
        selected=actors[state.current_seat].select(observation)
        action=resolve_action(observation,actions,selected)
        if action is None or not engine.step(state,action).ok:raise RuntimeError('illegal interval rollout action')
    bounds=reward_interval(state,seat)
    return bounds,('exact' if bounds[0]==bounds[1] else 'step_cap')


class Policy(Original):
    def __init__(self,weights,**kwargs):
        super().__init__(weights,**kwargs)
        if self.rival_model!='route':raise ValueError('isolated interval probe uses route rival prior')
        self.stats.update({'retained_pairs':0,'interval_pairs':0,'deadline_bounds':0,'step_bounds':0})

    def _select(self,o,base,started):
        if not 2<=len(o['hand'])<=3 or len(o['legal_actions'])<2:return base
        if o['phase'] not in (engine.TURN,engine.AFTER_DRAW):return base
        self.stats['probes']+=1;deadline=started+self.budget
        beliefs,cache=public_beliefs(o),{}
        ranked=sorted(o['legal_actions'],key=lambda a:sum(w*x for w,x in zip(self.weights,action_features(o,a,beliefs,cache))),reverse=True)
        candidates=[];seen=set()
        for action in ranked:
            signature=(action['type'],key(action['card']) if action.get('card') else None,action.get('color'))
            if signature not in seen:candidates.append(action);seen.add(signature)
            if len(candidates)>=self.max_actions:break
        base_index=next(i for i,a in enumerate(candidates) if a['action_id']==base)
        fingerprint={k:o[k] for k in ('seat','phase','active_color','direction','turn_number','hand_sizes','finished_order')}
        fingerprint['hand']=[key(c) for c in o['hand']];fingerprint['top']=key(o['top_discard'])
        synthetic_seed=int.from_bytes(hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).digest()[:8],'big')
        try:prepared=PreparedSampler(o)
        except ValueError:
            self.stats['sampling_declines']+=1;return base
        trials=[]
        for trial in range(self.max_trials):
            if time.perf_counter()>=deadline:break
            possible=sampled_world(o,random.Random(synthetic_seed+trial),prepared,True)
            if possible is None:self.stats['sampling_declines']+=1;continue
            world,refs=possible
            if any(engine._validate(world,bind(a,refs)) is not None for a in candidates):
                self.stats['world_rejections']+=1;return base
            values=[rollout_interval(world,refs,a,o['seat'],self.weights,deadline,self.max_steps) for a in candidates]
            bounds=[value[0] for value in values];trials.append(bounds)
            for _,reason in values:
                if reason=='deadline':self.stats['deadline_bounds']+=1
                elif reason=='step_cap':self.stats['step_bounds']+=1
            if any(lo!=hi for lo,hi in bounds):
                self.stats['incomplete_pairs']+=1;self.stats['interval_pairs']+=1
            else:self.stats['completed_trials']+=1
        self.stats['retained_pairs']+=len(trials);self.stats['probe_seconds']+=time.perf_counter()-started
        if len(trials)<self.min_trials:return base
        gaps=[[0. if i==base_index else row[i][0]-row[base_index][1]
               for row in trials] for i in range(len(candidates))]
        means=[sum(ds)/len(ds) for ds in gaps]
        selected=max(range(len(candidates)),key=lambda i:means[i]);ds=gaps[selected];gain=means[selected]
        variance=sum((d-gain)**2 for d in ds)/(len(ds)-1)
        lower=gain-1.96*math.sqrt(variance/len(ds))
        if selected!=base_index and gain>=.15 and lower>0:
            self.stats['overrides']+=1;return candidates[selected]['action_id']
        return base
