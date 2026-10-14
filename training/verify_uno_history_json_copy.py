"""Isolated JSON-schema history copy speed probe, not a deployed change.

The pinned referee's history/initial setup are JSON objects with string keys,
lists and finite scalars. Round-trip copy is only considered for those fields,
not as a replacement for the generic tuple/set-aware json_copy helper.
"""
import copy
import json
import random
import time
from pathlib import Path
from uno_belief_rollout_probe import engine, sampled_world, rollout
from uno_route_policy import Policy
from uno_public_belief import PreparedSampler
import uno_simulation_fast as fast


def known_json_schema_copy(value):
    return json.loads(json.dumps(value,allow_nan=False))


def main():
    weights=json.loads(Path('unoSDK/models/policy.json').read_text())['weights']
    actor=Policy(weights);observations=[];views=0;snapshots=0
    original=fast.json_copy;timings=[0.,0.]
    for seed in range(117_100_000,117_100_008):
        state=engine.new_game(engine.GameConfig(max_turns=1000),seed)
        while True:
            assert original(state.history)==known_json_schema_copy(state.history)
            assert original(state.initial_seed_setup)==known_json_schema_copy(state.initial_seed_setup)
            views+=1
            for i,method in enumerate((original,known_json_schema_copy)):
                fast.json_copy=method
                start=time.perf_counter();cloned=fast.snapshot(state);timings[i]+=time.perf_counter()-start
                assert all(getattr(cloned,k)==getattr(state,k) for k in vars(state) if k!='rng')
                assert cloned.rng.getstate()==state.rng.getstate()
                # Both top-level and nested public-event data are independent.
                frozen=copy.deepcopy(state.history)
                cloned.history.append({'type':'fixture_mutation'})
                if state.history:
                    cloned.history[0]['fixture_mutation']={'nested':['changed']}
                cloned.initial_seed_setup['fixture_mutation']=['changed']
                assert state.history==frozen and 'fixture_mutation' not in state.initial_seed_setup
                snapshots+=1
            fast.json_copy=original
            if state.phase==engine.FINISHED:break
            o=engine.observe(state,state.current_seat)
            if len(observations)<24 and 2<=len(o['hand'])<=3 and len(o['legal_actions'])>1:
                observations.append(o)
            assert engine.step(state,engine.engine_action_from_id(state,state.current_seat,actor.select(o))).ok
    whole=[0.,0.];checks=0
    for index,o in enumerate(observations):
        prepared=PreparedSampler(o)
        for trial in range(4):
            fast.json_copy=original
            possible=sampled_world(o,random.Random(index*100+trial+20261027),prepared,True)
            if possible is None:continue
            world,refs=possible
            for action in o['legal_actions'][:4]:
                values=[]
                for i,method in enumerate((original,known_json_schema_copy)):
                    fast.json_copy=method;start=time.perf_counter()
                    values.append(rollout(world,refs,action,o['seat'],weights,float('inf'),128,True,True))
                    whole[i]+=time.perf_counter()-start
                assert values[0]==values[1]
                checks+=1
    fast.json_copy=original
    report={'development_only':True,'runtime_and_live_gate_changed':False,
            'exact_json_schema_views':views,'isolated_snapshots':snapshots,
            'exact_fixed_rollouts':checks,'snapshot_seconds':timings,
            'snapshot_speedup':timings[0]/timings[1],'whole_rollout_seconds':whole,
            'whole_rollout_speedup':whole[0]/whole[1],
            'scope':'Pinned referee JSON history/initial setup only; not generic tuple/set copying'}
    Path('training/uno-history-json-copy-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
