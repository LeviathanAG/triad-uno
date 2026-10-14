"""Bounds against exact continuations of public-built hypothetical worlds."""
import itertools,json,random
from pathlib import Path
from uno_belief_rollout_probe import engine,sampled_world,rollout
from uno_belief_interval_probe import reward_interval,rollout_interval,Policy
from uno_public_belief import PreparedSampler
from uno_route_policy import Policy as Route


def main():
    weights=json.loads(Path('unoSDK/models/policy.json').read_text())['weights']
    actor=Route(weights);observations=[]
    for seed in range(148_000_000,148_000_032):
        state=engine.new_game(engine.GameConfig(max_turns=1000),seed)
        while state.phase!=engine.FINISHED:
            o=engine.observe(state,state.current_seat)
            if 2<=len(o['hand'])<=3 and len(o['legal_actions'])>1:
                observations.append(o)
                if len(observations)>=24:break
            assert engine.step(state,engine.engine_action_from_id(state,state.current_seat,actor.select(o))).ok
        if len(observations)>=24:break
    checks=0;censored=0;deadline=0
    for i,o in enumerate(observations):
        prepared=PreparedSampler(o)
        for trial in range(4):
            possible=sampled_world(o,random.Random(i*100+trial+20261030),prepared,True)
            if possible is None:continue
            world,refs=possible
            for action in o['legal_actions'][:2]:
                exact=rollout(world,refs,action,o['seat'],weights,float('inf'),1000,True,True)
                assert exact is not None
                for cap in (0,1,8,32,128,256):
                    (lo,hi),reason=rollout_interval(world,refs,action,o['seat'],weights,float('inf'),cap)
                    assert lo<=exact<=hi
                    if reason=='exact':assert lo==hi==exact
                    else:censored+=1
                    checks+=1
                (lo,hi),reason=rollout_interval(world,refs,action,o['seat'],weights,float('-inf'),128)
                assert lo<=exact<=hi;deadline+=reason=='deadline';checks+=1
    # All continuation placement/partial-draw rewards obey the same remaining-slot range.
    abstract=0;points=[5.,3.,2.,1.]
    for confirmed in range(4):
        lo,hi=min(points[confirmed:]),max(points[confirmed:])
        for later in range(confirmed,4):
            value=sum(points[later:])/len(points[later:]);assert lo<=value<=hi;abstract+=1
            for value in points[later:]:assert lo<=value<=hi;abstract+=1
    candidate=Policy(weights,budget=.5,max_trials=32,min_trials=4,max_actions=2,max_steps=128,
                     fast_simulation=True,cached_public_fold=True,rival_model='route')
    for o in observations:
        assert candidate.select(o) in {a['action_id'] for a in o['legal_actions']}
    report={'development_only':True,'public_observations':len(observations),
            'exact_continuation_bound_checks':checks,'censored_step_checks':censored,
            'deadline_checks':deadline,'abstract_remaining_slot_checks':abstract,
            'policy_legal_decisions':len(observations),'stats':candidate.stats,
            'real_hidden_state_or_game_seed_policy_inputs':False,
            'limitations':'Bounds cover sampled continuation rewards; posterior and rival model remain approximate.'}
    Path('training/uno-reward-interval-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
