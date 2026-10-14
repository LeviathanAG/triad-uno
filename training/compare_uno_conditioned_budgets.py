"""New162M development; old live selections retain their frozen sampler."""
import argparse,concurrent.futures,json,multiprocessing,random,time
from pathlib import Path
import numpy as np
from compare_uno_censoring import merge
from train_uno_league import engine,make_policy,paired_interval,summarize
from uno_belief_interval_probe import Policy as Original
from uno_belief_conditioned_interval_probe import Policy as Conditioned

CONFIGS=[{'name':'retained_route','kind':'route'},
 {'name':'original_half32_actions2','kind':'original','budget':.5,'max_trials':32,'max_actions':2},
 {'name':'conditioned_half32_actions2','kind':'conditioned','budget':.5,'max_trials':32,'max_actions':2},
 {'name':'conditioned_1500ms64_actions2','kind':'conditioned','budget':1.5,'max_trials':64,'max_actions':2},
 {'name':'conditioned_1500ms64_actions4','kind':'conditioned','budget':1.5,'max_trials':64,'max_actions':4}]


def evaluate(args):
    config,weights,seeds,reference,mode=args;points=[];wins=[];started=time.perf_counter()
    keys=['probes','overrides','sampling_declines','incomplete_pairs','completed_trials','world_rejections',
          'retained_pairs','interval_pairs','deadline_bounds','step_bounds','max_decision_seconds']
    stats={key:0 for key in keys}
    for seed in seeds:
        rivals=random.Random(seed^0x5354524F4E47).sample(reference,3) if mode=='trained' else [{'kind':'route','weights':weights}]*3 if mode=='route' else [{'kind':mode}]*3
        outcomes=[]
        for seat in range(4):
            if config['kind']=='route':candidate=make_policy({'kind':'route','weights':weights},0)
            else:
                cls=Original if config['kind']=='original' else Conditioned
                candidate=cls(weights,budget=config['budget'],max_trials=config['max_trials'],
                    max_actions=config['max_actions'],min_trials=4,max_steps=256,
                    fast_simulation=True,cached_public_fold=True,rival_model='route')
            actors=[None]*4;actors[seat]=candidate
            for offset,spec in enumerate(rivals,1):actors[(seat+offset)%4]=make_policy(spec,seed*10+offset)
            state=engine.new_game(engine.GameConfig(max_turns=1000),seed)
            while state.phase!=engine.FINISHED and seat not in state.finished_set:
                current=state.current_seat;o=engine.observe(state,current)
                selected=actors[current].select(o);action=engine.engine_action_from_id(state,current,selected)
                if action is None or not engine.step(state,action).ok:raise RuntimeError('illegal conditioned test action')
            if seat in state.finished_order:
                pos=state.finished_order.index(seat);outcomes.append((state.config.position_points[pos],float(pos==0)))
            else:
                remaining=state.config.position_points[len(state.outcome.placements):]
                outcomes.append((sum(remaining)/len(state.outcome.unfinished),0.))
            if config['kind']!='route':
                for key in keys:
                    value=candidate.stats.get(key,0)
                    stats[key]=max(stats[key],value) if key=='max_decision_seconds' else stats[key]+value
        points.append(sum(x[0] for x in outcomes)/4);wins.append(sum(x[1] for x in outcomes)/4)
    return {'points':float(np.mean(points)),'win_rate':float(np.mean(wins)),
            'games':len(seeds)*4,'block_points':points,'block_wins':wins,
            'probe_stats':stats,'seconds':time.perf_counter()-started}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--trained-pool',type=Path,required=True);p.add_argument('--blocks',type=int,default=64)
    p.add_argument('--seed-start',type=int,default=162000000);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    weights=json.loads(a.model.read_text())['weights'];reference=json.loads(a.trained_pool.read_text())
    chunks=[list(range(s,min(s+16,a.seed_start+a.blocks))) for s in range(a.seed_start,a.seed_start+a.blocks,16)]
    modes=('trained','route','random');tasks=[];keys=[]
    for mode in modes:
        for i,config in enumerate(CONFIGS):
            for chunk in chunks:keys.append((mode,i));tasks.append((config,weights,chunk,reference,mode))
    collected={key:[] for key in keys}
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as executor:
        for index,(key,row) in enumerate(zip(keys,executor.map(evaluate,tasks))):
            collected[key].append(row);print(json.dumps({'completed':index+1,'total':len(tasks),'mode':key[0],'config':CONFIGS[key[1]],'stats':row['probe_stats']}),flush=True)
    report={};means=[]
    for mode in modes:
        rows=[merge(collected[mode,i]) for i in range(len(CONFIGS))]
        if mode=='trained':means=[row['points'] for row in rows]
        report[mode]={'incumbent':rows[0],'candidates':[
            {'configuration':config,'result':row,'paired_gain':paired_interval(row,rows[0])}
            for config,row in zip(CONFIGS[1:],rows[1:])]}
    chosen=max(range(len(CONFIGS)),key=lambda i:means[i])
    report['selection']={'selected':chosen,'configuration':CONFIGS[chosen],
        'criterion':'own-trained distinct-rival league placement points;incumbent eligible'}
    report['provenance']={'development_only':True,'seed_start':a.seed_start,'blocks':a.blocks,
        'selection_family_reserved':163000000,'fresh_final_reserved':164000000,
        'same_uniform_public_prior':True,'one_anchor_condition_then_reject_other_constraints':True,
        'actual_hidden_state_or_private_game_seed_policy_inputs':False,
        'gaussian_sampled_world_confidence_heuristic':True,
        'old131M_and151M_recipes_unchanged':True,'max_configured_action_budget_seconds':1.5}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['selection']),flush=True)


if __name__=='__main__':main()
