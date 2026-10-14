"""Exact fixed-work rollout equivalence; public-built synthetic worlds only."""
import json,random,time
from pathlib import Path
from uno_belief_rollout_probe import engine,sampled_world,rollout
from uno_route_policy import Policy
from uno_public_belief import PreparedSampler,sample_unknown_hands

weights=json.loads(Path('unoSDK/models/policy.json').read_text())['weights'];actor=Policy(weights)
observations=[]
for seed in range(107_000_000,107_000_032):
 state=engine.new_game(engine.GameConfig(max_turns=1000),seed)
 while state.phase!=engine.FINISHED:
  o=engine.observe(state,state.current_seat)
  if 2<=len(o['hand'])<=3 and len(o['legal_actions'])>1:
   observations.append(o)
   if len(observations)>=24:break
  assert engine.step(state,engine.engine_action_from_id(state,state.current_seat,actor.select(o))).ok
 if len(observations)>=24:break
count=0;declines=0;seconds=[0.,0.,0.]
for index,o in enumerate(observations):
 prepared=PreparedSampler(o)
 for trial in range(4):
  synthetic_seed=index*100+trial+20261025
  first,second=random.Random(synthetic_seed),random.Random(synthetic_seed)
  assert sample_unknown_hands(o,first)==prepared.sample(second) and first.getstate()==second.getstate()
  possible=sampled_world(o,random.Random(synthetic_seed))
  faster=sampled_world(o,random.Random(synthetic_seed),prepared,True)
  assert (possible is None)==(faster is None)
  if possible is not None:
   a,b=possible[0],faster[0]
   assert possible[1]==faster[1] and a.rng.getstate()==b.rng.getstate()
   assert all(getattr(a,k)==getattr(b,k) for k in vars(a) if k!='rng')
  if possible is None:declines+=1;continue
  world,refs=possible
  for action in o['legal_actions'][:4]:
   values=[]
   for mode in (0,1,2):
    start=time.perf_counter();values.append(rollout(world,refs,action,o['seat'],weights,float('inf'),128,mode>0,mode==2));seconds[mode]+=time.perf_counter()-start
   assert values[0]==values[1]==values[2],'fast path changed exact finite rollout return'
   count+=1
report={'development_only':True,'public_observations':len(observations),'equivalent_paired_rollouts':count,
 'sampling_declines':declines,'fixed_work_seconds':{'baseline':seconds[0],'fast_snapshot_observation':seconds[1],'fast_with_cached_fold':seconds[2]},'overall_rollout_speedup':seconds[0]/seconds[2],
 'note':'No actual hidden state or real seed is passed to the probe; simulator used only to generate filtered verification fixtures. Infinite deadline verifies the same work/returns rather than different timing-limited trial subsets.'}
Path('training/uno-fast-rollout-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
