"""Development-only observation-based endgame belief rollout probe.

Uses the official referee ONLY as a development simulator of sampled possible
worlds. This is not deployment code: final inference must not import the local
evaluator. No trusted state, actual hidden hands, or real game seed is accepted.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from uno_public_belief import key,public_unknown_pool,sample_unknown_hands
from uno_policy_v2 import public_beliefs
from uno_route_policy import Policy as RoutePolicy,action_features
from uno_policy import Policy as LinearPolicy,INITIAL_WEIGHTS

ROOT=Path(__file__).resolve().parent
SDK=Path(os.environ.get('UNO_SDK',ROOT.parent/'unoSDK'))
sys.path.insert(0,str(SDK/'local_evaluator/src'))
from triad_uno import engine
from triad_uno.cards import Card
from triad_uno.baselines import RandomAgent,HeuristicAgent


def sampled_world(observation,rng,prepared_sampler=None,fast_copy=False):
    try:
        sampled=prepared_sampler.sample(rng) if prepared_sampler is not None else sample_unknown_hands(observation,rng)
    except ValueError:
        # Incomplete public-history fixtures cannot justify constructing a
        # complete possible world. Decline rather than invent missing facts.
        return None
    if sampled is None:return None
    opponents,draw=sampled
    discards=prepared_sampler.discards if prepared_sampler is not None else public_unknown_pool(observation)[1]
    uid=0
    def card(card_key):
        nonlocal uid
        colour,kind,value=card_key
        result=Card(colour,kind,value,uid);uid+=1
        return result
    own=[];refs={}
    for c in observation['hand']:
        item=card(key(c));own.append(item);refs[c['card_ref']]=item.uid
    hands=[own if seat==observation['seat'] else [card(k) for k in opponents[seat]]
           for seat in range(len(observation['hand_sizes']))]
    discard_cards=[card(key(c)) for c in discards]
    draw_cards=[card(k) for k in draw]
    assert uid==108
    finished=list(observation['finished_order'])
    penalty=public_beliefs(observation)['penalty'];source=None
    if penalty:
        source=next((e.get('seat') for e in reversed(observation['history'])
                     if e.get('type')=='play' and e.get('pending_penalty',0)>0),None)
    world_rng=random.Random(0);world_rng.setstate(rng.getstate())
    if fast_copy:
        from uno_simulation_fast import json_copy
        history=json_copy(observation['history'])
    else:
        history=copy.deepcopy(observation['history'])
    state=engine.GameState(config=engine.GameConfig(max_turns=1000),seed=0,
        rng=world_rng,hands=hands,draw_pile=draw_cards,discard_pile=discard_cards,
        current_seat=observation['seat'],direction=observation['direction'],
        active_color=observation['active_color'],phase=observation['phase'],
        pending_penalty=penalty,pending_source=source,
        drawn_card_uid=refs.get(observation.get('drawn_card_ref')),
        turn_count=observation['turns_completed'],finished_order=finished,
        finished_set=set(finished),history=history)
    return state,refs


def bind(action,refs):
    return engine.EngineAction(action['type'],refs.get(action.get('card_ref')),action.get('color'))


def terminal_reward(state,seat):
    if seat in state.finished_order:
        return state.config.position_points[state.finished_order.index(seat)]
    if state.phase==engine.FINISHED:
        remaining=state.config.position_points[len(state.outcome.placements):]
        return sum(remaining)/len(state.outcome.unfinished)
    return None


def rollout(world,refs,action,seat,weights,deadline,max_steps,fast_simulation=False,cached_public_fold=False,rival_specs=None):
    if fast_simulation:
        from uno_simulation_fast import snapshot,observe_with_actions,resolve_action
        state=snapshot(world)
    else:
        state=copy.deepcopy(world)
    if cached_public_fold:
        from uno_public_fold_fast import CachedRoutePolicy,PublicFold
        fold=PublicFold()
        actors=[CachedRoutePolicy(weights,fold) for _ in range(4)]
    else:
        actors=[RoutePolicy(weights) for _ in range(4)]
    if rival_specs is not None:
        for other,(kind,synthetic_seed) in enumerate(rival_specs):
            if other==seat or kind=='route':continue
            if kind=='random':actors[other]=RandomAgent(synthetic_seed)
            elif kind in ('starter','heuristic'):actors[other]=HeuristicAgent(synthetic_seed)
            else:
                linear=list(weights[:18]) if kind=='v1' else list(INITIAL_WEIGHTS)
                if kind=='attack':
                    for index in (2,4,6,10):linear[index]+=2.
                if kind=='preserve':linear[5]-=2.;linear[6]-=2.
                actors[other]=LinearPolicy(linear)
    if not engine.step(state,bind(action,refs)).ok:
        raise RuntimeError('synthetic world does not preserve current known action legality')
    for _ in range(max_steps):
        reward=terminal_reward(state,seat)
        if reward is not None:return reward
        if time.perf_counter()>=deadline:return None
        if fast_simulation:
            obs,actions=observe_with_actions(state,state.current_seat)
        else:
            obs=engine.observe(state,state.current_seat)
        selected=actors[state.current_seat].select(obs)
        moved=resolve_action(obs,actions,selected) if fast_simulation else engine.engine_action_from_id(state,state.current_seat,selected)
        if moved is None or not engine.step(state,moved).ok:raise RuntimeError('simulated policy selected illegal action')
    return terminal_reward(state,seat)


class Policy:
    def __init__(self,weights,budget=.25,max_trials=12,min_trials=4,max_actions=4,max_steps=128,fast_simulation=False,cached_public_fold=False,rival_model='route'):
        if not 0<budget<=1 or min(max_trials,max_actions,max_steps)<1 or min_trials<2:raise ValueError('invalid bounded rollout configuration')
        self.weights=list(weights);self.base=RoutePolicy(weights)
        self.budget=budget;self.max_trials=max_trials;self.min_trials=min_trials
        self.max_actions=max_actions;self.max_steps=max_steps
        self.fast_simulation=fast_simulation
        self.cached_public_fold=cached_public_fold
        if rival_model not in ('route','mixture'):raise ValueError('unknown approximate rival prior')
        self.rival_model=rival_model
        self.stats={'calls':0,'probes':0,'overrides':0,'sampling_declines':0,'incomplete_pairs':0,
                    'completed_trials':0,'world_rejections':0,'max_decision_seconds':0.,'probe_seconds':0.}

    def select(self,observation):
        started=time.perf_counter();base=self.base.select(observation)
        self.stats['calls']+=1
        try:
            return self._select(observation,base,started)
        finally:
            duration=time.perf_counter()-started
            self.stats['max_decision_seconds']=max(self.stats['max_decision_seconds'],duration)

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
        rewards=[]
        prepared=None
        if self.fast_simulation:
            from uno_public_belief import PreparedSampler
            try:prepared=PreparedSampler(o)
            except ValueError:
                self.stats['sampling_declines']+=1
                return base
        for trial in range(self.max_trials):
            if time.perf_counter()>=deadline:break
            possible=sampled_world(o,random.Random(synthetic_seed+trial),prepared,self.fast_simulation)
            if possible is None:self.stats['sampling_declines']+=1;continue
            world,refs=possible
            if any(engine._validate(world,bind(a,refs)) is not None for a in candidates):
                self.stats['world_rejections']+=1
                return base
            styles=None
            if self.rival_model=='mixture':
                # Hypothetical styles are a fixed approximate prior, not facts
                # about real opponents. Same styles/streams across this pair.
                style_rng=random.Random(synthetic_seed+trial+0x5354594C45)
                styles=[(style_rng.choice(('starter','heuristic','random','route','v1','initial','attack','preserve')),
                         style_rng.randrange(2**63)) for _ in range(4)]
            values=[rollout(world,refs,a,o['seat'],self.weights,deadline,self.max_steps,self.fast_simulation,self.cached_public_fold,styles) for a in candidates]
            if any(v is None for v in values):
                self.stats['incomplete_pairs']+=1
                continue
            rewards.append(values)
        self.stats['completed_trials']+=len(rewards);self.stats['probe_seconds']+=time.perf_counter()-started
        if len(rewards)<self.min_trials:return base
        means=[sum(row[i] for row in rewards)/len(rewards) for i in range(len(candidates))]
        selected=max(range(len(candidates)),key=lambda i:means[i])
        differences=[row[selected]-row[base_index] for row in rewards]
        gain=sum(differences)/len(differences)
        variance=sum((d-gain)**2 for d in differences)/(len(differences)-1)
        lower=gain-1.96*math.sqrt(variance/len(differences))
        # Conservative override: a meaningful paired advantage inside this
        # approximate sampled-world model. It is not a real-game guarantee.
        if selected!=base_index and gain>=.15 and lower>0:
            self.stats['overrides']+=1
            return candidates[selected]['action_id']
        return base
