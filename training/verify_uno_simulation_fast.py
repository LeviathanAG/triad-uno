"""Exact offline SDK clone/observation equivalence and fixed-work timing."""
import copy,json,random,statistics,time
from pathlib import Path
from uno_simulation_fast import engine,snapshot,observe_with_actions,resolve_action


def main():
 rng=random.Random(20261025);count=0;views=0;phase={};max_history=0;samples=[];recycles=0
 for game in range(16):
  state=engine.new_game(engine.GameConfig(max_turns=80 if game<8 else 1000),71_000_000+game)
  while True:
   phase[state.phase]=phase.get(state.phase,0)+1;max_history=max(max_history,len(state.history))
   for seat in range(4):
    fast,actions=observe_with_actions(state,seat);assert fast==engine.observe(state,seat)
    views+=1
   if state.turn_count%7==0 or state.phase in (engine.FINISHED,engine.CHOOSE_INITIAL_COLOR):
    original_history=copy.deepcopy(state.history);original_rng=state.rng.getstate()
    a,b=copy.deepcopy(state),snapshot(state)
    assert vars(a)==vars(b) or (a.rng.getstate()==b.rng.getstate() and all(getattr(a,k)==getattr(b,k) for k in vars(a) if k!='rng'))
    for seat in range(4):assert engine.observe(a,seat)==engine.observe(b,seat)
    if state.phase!=engine.FINISHED:
     o,actions=observe_with_actions(a,a.current_seat);chosen=rng.choice(o['legal_actions'])['action_id']
     fa=resolve_action(o,actions,chosen);fb=engine.engine_action_from_id(b,b.current_seat,chosen)
     assert fa==fb
     ta,tb=engine.step(a,fa),engine.step(b,fb)
     assert ta==tb and a.rng.getstate()==b.rng.getstate()
     assert all(getattr(a,k)==getattr(b,k) for k in vars(a) if k!='rng')
     for seat in range(4):assert engine.observe(a,seat)==engine.observe(b,seat)
    assert state.history==original_history and state.rng.getstate()==original_rng
    # Mutation-isolation checks include nested public card dictionaries.
    b.config.max_turns+=1;b.finished_set.add(99);b.hands[0].clear();b.rng.random()
    if b.history:
     b.history[0]['clone_only']=True
     if 'card' in b.history[0]:b.history[0]['card']['color']='clone_only'
    b.initial_seed_setup['clone_only']=True
    assert state.history==original_history and state.rng.getstate()==original_rng and 99 not in state.finished_set
    count+=1
    if len(samples)<32 and len(state.history)>20:samples.append(copy.deepcopy(state))
   if state.phase==engine.FINISHED:break
   o=engine.observe(state,state.current_seat);selected=rng.choice(o['legal_actions'])['action_id']
   before=len(state.draw_pile);assert engine.step(state,engine.engine_action_from_id(state,state.current_seat,selected)).ok
   if before==0 and len(state.draw_pile)>0:recycles+=1
 times={name:[] for name in ('deepcopy','snapshot','official_observe_resolve','fast_observe_resolve')}
 for state in samples:
  for repeat in range(5):
   for name,fn in [('deepcopy',lambda:copy.deepcopy(state)),('snapshot',lambda:snapshot(state))]:
    t=time.perf_counter();fn();times[name].append(time.perf_counter()-t)
   t=time.perf_counter();o=engine.observe(state,state.current_seat)
   if o['legal_actions']:engine.engine_action_from_id(state,state.current_seat,o['legal_actions'][0]['action_id'])
   times['official_observe_resolve'].append(time.perf_counter()-t)
   t=time.perf_counter();o,actions=observe_with_actions(state,state.current_seat)
   if o['legal_actions']:resolve_action(o,actions,o['legal_actions'][0]['action_id'])
   times['fast_observe_resolve'].append(time.perf_counter()-t)
 medians={name:statistics.median(values) for name,values in times.items()}
 report={'development_only':True,'all_mutable_snapshot_fields_isolated':True,'exact_filtered_views':views,
  'exact_snapshot_transitions':count,'phases':phase,'max_history_entries':max_history,'observed_recycles':recycles,
  'timing_samples':len(samples)*5,'median_seconds':medians,'snapshot_speedup':medians['deepcopy']/medians['snapshot'],
  'observe_resolve_speedup':medians['official_observe_resolve']/medians['fast_observe_resolve'],
  'note':'Timing covers fixed work, not wall-clock-limited trial selection; strength requires separate fresh game benchmark.'}
 Path('training/uno-simulation-fast-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
