"""Exact cached public fold under official append-only referee trajectories."""
import copy,json,random,time
from pathlib import Path
from uno_simulation_fast import engine
from uno_public_fold_fast import PublicFold,CachedRoutePolicy
from uno_route_policy import Policy
from uno_policy_v2 import public_beliefs

weights=json.loads(Path('unoSDK/models/policy.json').read_text())['weights']
fold=PublicFold();policy=CachedRoutePolicy(weights);baseline=Policy(weights)
rng=random.Random(20261026);checks=0;decisions=0;full_seconds=0.;cached_seconds=0.
for seed in range(108_000_000,108_000_032):
 state=engine.new_game(engine.GameConfig(max_turns=1000),seed)
 while True:
  for seat in range(4):
   o=engine.observe(state,seat);start=time.perf_counter();expected=public_beliefs(o);full_seconds+=time.perf_counter()-start
   start=time.perf_counter();actual=fold.update(o);cached_seconds+=time.perf_counter()-start
   assert actual==expected,'public fold changed current policy semantics'
   checks+=1
  if state.phase==engine.FINISHED:break
  o=engine.observe(state,state.current_seat)
  assert baseline.select(o)==policy.select(o)
  # Fresh copied, truncated and different-game observations must reset safely.
  if state.turn_count%17==0:
   fresh=copy.deepcopy(o);assert fold.update(fresh)==public_beliefs(fresh)
   truncated=dict(o,history=o['history'][-5:]);assert fold.update(truncated)==public_beliefs(truncated)
   assert fold.update(o)==public_beliefs(o)
  action=rng.choice(o['legal_actions'])['action_id']
  assert engine.step(state,engine.engine_action_from_id(state,state.current_seat,action)).ok
  decisions+=1
report={'development_only':True,'exact_public_fold_checks':checks,'exact_cached_policy_decisions':decisions,
 'copied_truncated_different_game_resets_checked':True,'full_fold_seconds':full_seconds,'cached_fold_seconds':cached_seconds,
 'fold_speedup':full_seconds/cached_seconds,'assumption':'Pinned SDK histories append events and never mutate older events; cache is not a deployed inference API.'}
Path('training/uno-public-fold-fast-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
