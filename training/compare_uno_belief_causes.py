"""Common fresh development benchmark:time headroom versus approximate rival prior."""
import concurrent.futures,json,multiprocessing,random,time
from pathlib import Path
import numpy as np
from train_uno_league import engine,make_policy,summarize,paired_interval
from uno_belief_rollout_probe import Policy as Probe
from uno_policy import INITIAL_WEIGHTS

CONFIGS=[{'name':'retained_route'},
 {'name':'original8','max_trials':8,'fast_simulation':False,'cached_public_fold':False,'rival_model':'route'},
 {'name':'fast8','max_trials':8,'fast_simulation':True,'cached_public_fold':True,'rival_model':'route'},
 {'name':'fast16','max_trials':16,'fast_simulation':True,'cached_public_fold':True,'rival_model':'route'},
 {'name':'fast16_mixture','max_trials':16,'fast_simulation':True,'cached_public_fold':True,'rival_model':'mixture'}]


def job(args):
 config,weights,seeds,reference,mode=args;points=[];wins=[]
 stats={'probes':0,'overrides':0,'sampling_declines':0,'incomplete_pairs':0,'completed_trials':0,'world_rejections':0,'max_decision_seconds':0.}
 started=time.perf_counter()
 for seed in seeds:
  rng=random.Random(seed^0x4C4541475545)
  rivals=[rng.choice(reference) for _ in range(3)] if mode=='league' else [{'kind':mode}]*3
  outcomes=[]
  for seat in range(4):
   actors=[None]*4
   args={k:v for k,v in config.items() if k!='name'}
   actors[seat]=Probe(weights,budget=.15,min_trials=4,**args) if args else make_policy({'kind':'route','weights':weights},0)
   for offset,spec in enumerate(rivals,1):actors[(seat+offset)%4]=make_policy(spec,seed*10+offset)
   state=engine.new_game(engine.GameConfig(max_turns=1000),seed)
   while state.phase!=engine.FINISHED and seat not in state.finished_set:
    current=state.current_seat;o=engine.observe(state,current);selected=actors[current].select(o)
    action=engine.engine_action_from_id(state,current,selected)
    if action is None or not engine.step(state,action).ok:raise RuntimeError('illegal development probe action')
   if seat in state.finished_order:
    position=state.finished_order.index(seat);outcomes.append((state.config.position_points[position],float(position==0)))
   else:
    remaining=state.config.position_points[len(state.outcome.placements):];outcomes.append((sum(remaining)/len(state.outcome.unfinished),0.))
   if args:
    for name in stats:
     if name=='max_decision_seconds':stats[name]=max(stats[name],actors[seat].stats[name])
     else:stats[name]+=actors[seat].stats[name]
  points.append(sum(x[0] for x in outcomes)/4);wins.append(sum(x[1] for x in outcomes)/4)
 return {'points':float(np.mean(points)),'win_rate':float(np.mean(wins)),'games':len(seeds)*4,'block_points':points,'block_wins':wins,'probe_stats':stats,'seconds':time.perf_counter()-started}


def main():
 weights=json.loads(Path('unoSDK/models/policy.json').read_text())['weights']
 v1=json.loads(Path('unoSDK/models/incumbent-v1.json').read_text())['weights']
 attack=list(INITIAL_WEIGHTS)
 for index in (2,4,6,10):attack[index]+=2.
 preserve=list(INITIAL_WEIGHTS);preserve[5]-=2.;preserve[6]-=2.
 reference=[{'kind':'starter'},{'kind':'heuristic'},{'kind':'random'},{'kind':'route','weights':weights},
  {'kind':'v1','weights':v1},{'kind':'v1','weights':INITIAL_WEIGHTS},{'kind':'v1','weights':attack},{'kind':'v1','weights':preserve}]
 seeds=list(range(112_000_000,112_000_064));modes=('league','starter','random')
 with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
  results=list(pool.map(job,[(config,weights,seeds,reference,mode) for mode in modes for config in CONFIGS]))
 report={}
 for i,mode in enumerate(modes):
  rows=results[i*len(CONFIGS):(i+1)*len(CONFIGS)]
  report[mode]={'incumbent':summarize(rows[0]),'candidates':[{'config':config,'summary':summarize(result),'paired_gain':paired_interval(result,rows[0])} for config,result in zip(CONFIGS[1:],rows[1:])]}
 chosen=max(range(len(CONFIGS)),key=lambda i:results[i]['points'])
 report['selection']={'selected':chosen,'config':CONFIGS[chosen],'metric':'common development league mean placement points;baseline eligible'}
 report['provenance']={'development_only':True,'seed_start':112_000_000,'blocks':64,'workers':2,'budget_seconds':.15,'min_trials':4,
  'heuristic_regime_alias':'starter and SDKheuristic are identical under canonical legal order; evaluated once',
  'own_private_public_only_policy_inputs':True,'rival_prior':'approximate uniform eight style labels;not actual opponent identities/hidden cards',
  'uncertainty':'model-world Gaussian lowerbound is heuristic;not a guarantee about real games','next_fresh_final_seed_if_needed':113_000_000}
 Path('training/uno-belief-causes-development.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
