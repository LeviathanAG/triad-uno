"""Isolated deterministic feature memoization experiment, not integrated.

Cache keys contain only the queried hand-card multiset and top/active colour.
For this experiment queried rival hands exist solely in public-built synthetic
worlds. Each public decision gets a fresh bounded cache. No real hidden state
or game seed is passed to any policy. A temporary patch is process-local and
restored after measuring fixed-work rollouts.
"""
from functools import lru_cache
import json
from pathlib import Path
import random
import time
import uno_route_policy as route
from uno_belief_rollout_probe import engine,sampled_world,rollout
from uno_public_belief import PreparedSampler

ORIGINAL=route.route_length


@lru_cache(maxsize=2048)
def memo(cards,colour,kind,value,max_states,max_depth):
    hand=[{'color':c or None,'kind':k,'value':v} for c,k,v in cards]
    return ORIGINAL(hand,colour,{'kind':kind,'value':value},max_states,max_depth)


def cached_route_length(hand,colour,top,max_states=128,max_depth=8):
    cards=tuple(sorted((c.get('color') or '',c['kind'],
                        c.get('value') if c.get('value') is not None else -1) for c in hand))
    return memo(cards,colour,top.get('kind'),top.get('value',-1),max_states,max_depth)


def main():
    weights=json.loads(Path('unoSDK/models/policy.json').read_text())['weights']
    actor=route.Policy(weights);observations=[]
    for seed in range(123_100_000,123_100_032):
        state=engine.new_game(engine.GameConfig(max_turns=1000),seed)
        while state.phase!=engine.FINISHED:
            o=engine.observe(state,state.current_seat)
            if 2<=len(o['hand'])<=3 and len(o['legal_actions'])>1:
                observations.append(o)
                if len(observations)>=24:break
            assert engine.step(state,engine.engine_action_from_id(state,state.current_seat,actor.select(o))).ok
        if len(observations)>=24:break
    seconds=[0.,0.];checks=0;hits=misses=0
    try:
        for index,o in enumerate(observations):
            route.route_length=ORIGINAL;prepared=PreparedSampler(o);work=[]
            for trial in range(8):
                possible=sampled_world(o,random.Random(index*100+trial+20261029),prepared,True)
                if possible:
                    work.extend((possible,action) for action in o['legal_actions'][:4])
            outputs=[]
            for mode,implementation in enumerate((ORIGINAL,cached_route_length)):
                memo.cache_clear();route.route_length=implementation;values=[]
                started=time.perf_counter()
                for (world,refs),action in work:
                    values.append(rollout(world,refs,action,o['seat'],weights,float('inf'),128,True,True))
                seconds[mode]+=time.perf_counter()-started
                outputs.append(values)
                if mode:hits+=memo.cache_info().hits;misses+=memo.cache_info().misses
            assert outputs[0]==outputs[1]
            checks+=len(work)
    finally:
        route.route_length=ORIGINAL;memo.cache_clear()
    report={'development_only':True,'runtime_and_live_gate_changed':False,
            'public_decisions':len(observations),'identical_fixed_rollouts':checks,
            'whole_seconds':seconds,'whole_speedup':seconds[0]/seconds[1],
            'cache_hits':hits,'cache_misses':misses,'max_cache_entries':2048,
            'cache_scope':'one own/public current decision and its hypothetical rollouts'}
    Path('training/uno-route-memoization-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
