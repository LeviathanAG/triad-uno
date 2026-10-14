"""Paired full-game development benchmark of one bounded endgame probe."""
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import random
import time
import numpy as np
from train_uno_league import engine,make_policy,paired_interval,summarize
from uno_policy import INITIAL_WEIGHTS
from uno_belief_rollout_probe import Policy as Probe


def job(args):
    use_probe,weights,seeds,reference,mode,budget=args
    points,wins=[],[];stats={'probes':0,'overrides':0,'sampling_declines':0,'incomplete_pairs':0,'completed_trials':0,'max_decision_seconds':0.}
    started=time.perf_counter()
    for seed in seeds:
        rng=random.Random(seed^0x4C4541475545)
        rivals=[rng.choice(reference) for _ in range(3)] if mode=='league' else [{'kind':mode}]*3
        outcomes=[]
        for seat in range(4):
            actors=[None]*4
            actors[seat]=Probe(weights,budget=budget,max_trials=8,min_trials=4) if use_probe else make_policy({'kind':'route','weights':weights},seed*10+9)
            for offset,spec in enumerate(rivals,1):actors[(seat+offset)%4]=make_policy(spec,seed*10+offset)
            state=engine.new_game(engine.GameConfig(max_turns=1000),seed)
            while state.phase!=engine.FINISHED and seat not in state.finished_set:
                current=state.current_seat;o=engine.observe(state,current);chosen=actors[current].select(o)
                action=engine.engine_action_from_id(state,current,chosen)
                if action is None or not engine.step(state,action).ok:raise RuntimeError('illegal benchmark policy action')
            if seat in state.finished_order:
                position=state.finished_order.index(seat);outcomes.append((state.config.position_points[position],float(position==0)))
            else:
                remaining=state.config.position_points[len(state.outcome.placements):]
                outcomes.append((sum(remaining)/len(state.outcome.unfinished),0.))
            if use_probe:
                for name in stats:
                    if name=='max_decision_seconds':stats[name]=max(stats[name],actors[seat].stats[name])
                    else:stats[name]+=actors[seat].stats[name]
        points.append(sum(x[0] for x in outcomes)/4);wins.append(sum(x[1] for x in outcomes)/4)
    return {'points':float(np.mean(points)),'win_rate':float(np.mean(wins)),'games':len(seeds)*4,
            'block_points':points,'block_wins':wins,'probe_stats':stats,'seconds':time.perf_counter()-started}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--blocks',type=int,default=128)
    p.add_argument('--seed-start',type=int,default=56_000_000);p.add_argument('--workers',type=int,default=2);p.add_argument('--budget',type=float,default=.15)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();weights=json.loads(a.model.read_text())['weights']
    v1=json.loads((a.model.parent/'incumbent-v1.json').read_text())['weights']
    attack=list(INITIAL_WEIGHTS)
    for i in (2,4,6,10):attack[i]+=2.
    preserve=list(INITIAL_WEIGHTS);preserve[5]-=2.;preserve[6]-=2.
    reference=[{'kind':'starter'},{'kind':'heuristic'},{'kind':'random'},{'kind':'route','weights':weights},
               {'kind':'v1','weights':v1},{'kind':'v1','weights':INITIAL_WEIGHTS},{'kind':'v1','weights':attack},{'kind':'v1','weights':preserve}]
    seeds=list(range(a.seed_start,a.seed_start+a.blocks));modes=('league','starter','heuristic','random')
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=list(pool.map(job,[(probe,weights,seeds,reference,mode,a.budget) for mode in modes for probe in (True,False)]))
    report={mode:{'candidate':summarize(results[2*i]),'incumbent':summarize(results[2*i+1]),'paired_gain':paired_interval(results[2*i],results[2*i+1])} for i,mode in enumerate(modes)}
    report['provenance']={'development_only':True,'seed_start':a.seed_start,'blocks':a.blocks,'fixed_budget_seconds':a.budget,'actual_hidden_state_or_seed_policy_inputs':False}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
