"""Fresh strong-league development of explicit censored sampled-score bounds."""
import argparse,concurrent.futures,json,multiprocessing,random,time
from pathlib import Path
import numpy as np
from train_uno_league import engine,make_policy,paired_interval,summarize
from uno_belief_rollout_probe import Policy as Discard
from uno_belief_interval_probe import Policy as Interval

CONFIGS=[{'name':'retained_route','kind':'route'},
         {'name':'discard128','kind':'discard','max_steps':128},
         {'name':'interval128','kind':'interval','max_steps':128},
         {'name':'interval256','kind':'interval','max_steps':256},
         {'name':'interval512','kind':'interval','max_steps':512}]


def evaluate(args):
    config,weights,seeds,reference,mode=args;points=[];wins=[];started=time.perf_counter()
    keys=['probes','overrides','sampling_declines','incomplete_pairs','completed_trials','world_rejections',
          'retained_pairs','interval_pairs','deadline_bounds','step_bounds','max_decision_seconds']
    stats={k:0 for k in keys}
    for seed in seeds:
        rng=random.Random(seed^0x5354524F4E47)
        rivals=rng.sample(reference,3) if mode=='trained' else ([{'kind':'route','weights':weights}]*3 if mode=='route' else [{'kind':mode}]*3)
        outcomes=[]
        for seat in range(4):
            actors=[None]*4
            if config['kind']=='route':candidate=make_policy({'kind':'route','weights':weights},0)
            else:
                cls=Interval if config['kind']=='interval' else Discard
                candidate=cls(weights,budget=.5,max_trials=32,min_trials=4,max_actions=2,
                              max_steps=config['max_steps'],fast_simulation=True,cached_public_fold=True,rival_model='route')
            actors[seat]=candidate
            for offset,spec in enumerate(rivals,1):actors[(seat+offset)%4]=make_policy(spec,seed*10+offset)
            state=engine.new_game(engine.GameConfig(max_turns=1000),seed)
            while state.phase!=engine.FINISHED and seat not in state.finished_set:
                current=state.current_seat;o=engine.observe(state,current)
                selected=actors[current].select(o)
                action=engine.engine_action_from_id(state,current,selected)
                if action is None or not engine.step(state,action).ok:raise RuntimeError('illegal censoring test action')
            if seat in state.finished_order:
                pos=state.finished_order.index(seat);outcomes.append((state.config.position_points[pos],float(pos==0)))
            else:
                remaining=state.config.position_points[len(state.outcome.placements):]
                outcomes.append((sum(remaining)/len(state.outcome.unfinished),0.))
            if config['kind']!='route':
                for k in keys:
                    value=candidate.stats.get(k,0)
                    stats[k]=max(stats[k],value) if k=='max_decision_seconds' else stats[k]+value
        points.append(sum(x[0] for x in outcomes)/4);wins.append(sum(x[1] for x in outcomes)/4)
    return {'points':float(np.mean(points)),'win_rate':float(np.mean(wins)),
            'games':len(seeds)*4,'block_points':points,'block_wins':wins,
            'probe_stats':stats,'seconds':time.perf_counter()-started}


def merge(rows):
    points=[p for r in rows for p in r['block_points']];wins=[p for r in rows for p in r['block_wins']]
    stats={k:(max(r['probe_stats'][k] for r in rows) if k=='max_decision_seconds' else sum(r['probe_stats'][k] for r in rows)) for k in rows[0]['probe_stats']}
    return {'points':float(np.mean(points)),'win_rate':float(np.mean(wins)),
            'games':len(points)*4,'block_points':points,'block_wins':wins,'probe_stats':stats,
            'summed_worker_seconds':sum(r['seconds'] for r in rows)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--trained-pool',type=Path,required=True);p.add_argument('--blocks',type=int,default=64)
    p.add_argument('--seed-start',type=int,default=130_000_000);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    weights=json.loads(a.model.read_text())['weights'];reference=json.loads(a.trained_pool.read_text())
    chunks=[list(range(s,min(s+16,a.seed_start+a.blocks))) for s in range(a.seed_start,a.seed_start+a.blocks,16)]
    modes=('trained','route','random');tasks=[];keys=[]
    for mode in modes:
        for i,config in enumerate(CONFIGS):
            for chunk in chunks:keys.append((mode,i));tasks.append((config,weights,chunk,reference,mode))
    collected={k:[] for k in keys}
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as executor:
        for index,(key,result) in enumerate(zip(keys,executor.map(evaluate,tasks))):
            collected[key].append(result);print(json.dumps({'completed':index+1,'total':len(tasks),'mode':key[0],'configuration':CONFIGS[key[1]],'stats':result['probe_stats']}),flush=True)
    report={};means=[]
    for mode in modes:
        rows=[merge(collected[mode,i]) for i in range(len(CONFIGS))]
        if mode=='trained':means=[r['points'] for r in rows]
        report[mode]={'incumbent':summarize(rows[0]),'candidates':[
            {'configuration':config,'summary':summarize(row),'paired_gain':paired_interval(row,rows[0])}
            for config,row in zip(CONFIGS[1:],rows[1:])]}
    chosen=max(range(len(CONFIGS)),key=lambda i:means[i])
    report['selection']={'selected':chosen,'configuration':CONFIGS[chosen],
                         'criterion':'own-trained distinct-rival league placement points;incumbent eligible'}
    report['provenance']={'development_only':True,'seed_start':a.seed_start,'blocks':a.blocks,
        'selection_family_reserved':131_000_000,'fresh_final_family_reserved':133_000_000,
        'root_numeric_fixture_not_reused':132_000_000,'actual_hidden_state_or_game_seed_policy_inputs':False,
        'bounds':'All valid sampled pairs retained; unfinished score in exact remaining-placement range',
        'uncertainty':'Posterior/rival model and Gaussian world-mean test remain approximate; real games are the strength gate',
        'completed_trials_definition':'exact pairs only; retained_pairs additionally counts interval pairs'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
