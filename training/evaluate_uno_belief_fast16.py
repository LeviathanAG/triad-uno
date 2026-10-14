"""Fresh full-game gate for the configuration fixed by 112M development.

This imports the official referee only for development simulations. No hidden
state or actual game seed is passed to the candidate's select method.
"""
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import numpy as np
from compare_uno_belief_causes import job
from train_uno_league import paired_interval, summarize
from uno_policy import INITIAL_WEIGHTS

FIXED={'name':'fast16','max_trials':16,'fast_simulation':True,
       'cached_public_fold':True,'rival_model':'route'}


def merge(rows):
    points=[x for r in rows for x in r['block_points']]
    wins=[x for r in rows for x in r['block_wins']]
    stats={k:(max(r['probe_stats'][k] for r in rows) if k=='max_decision_seconds'
              else sum(r['probe_stats'][k] for r in rows)) for k in rows[0]['probe_stats']}
    return {'points':float(np.mean(points)),'win_rate':float(np.mean(wins)),
            'games':sum(r['games'] for r in rows),'block_points':points,
            'block_wins':wins,'probe_stats':stats,
            'summed_worker_seconds':sum(r['seconds'] for r in rows)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--v1',type=Path,required=True)
    p.add_argument('--blocks',type=int,default=512)
    p.add_argument('--seed-start',type=int,default=113_000_000)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    weights=json.loads(a.model.read_text())['weights']
    v1=json.loads(a.v1.read_text())['weights']
    attack=list(INITIAL_WEIGHTS)
    for i in (2,4,6,10):attack[i]+=2.
    preserve=list(INITIAL_WEIGHTS);preserve[5]-=2.;preserve[6]-=2.
    reference=[{'kind':'starter'},{'kind':'heuristic'},{'kind':'random'},
               {'kind':'route','weights':weights},{'kind':'v1','weights':v1},
               {'kind':'v1','weights':INITIAL_WEIGHTS},
               {'kind':'v1','weights':attack},{'kind':'v1','weights':preserve}]
    chunks=[list(range(s,min(s+32,a.seed_start+a.blocks)))
            for s in range(a.seed_start,a.seed_start+a.blocks,32)]
    modes=('league','starter','random');tasks=[];keys=[]
    for mode in modes:
        for config in ({'name':'retained_route'},FIXED):
            for chunk in chunks:
                tasks.append((config,weights,chunk,reference,mode))
                keys.append((mode,config['name']))
    collected={k:[] for k in keys}
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        for index,(key,result) in enumerate(zip(keys,pool.map(job,tasks))):
            collected[key].append(result)
            print(json.dumps({'completed_chunk':index+1,'chunks':len(tasks),
                              'mode':key[0],'config':key[1],
                              'games':result['games'],'stats':result['probe_stats']}),flush=True)
    report={}
    for mode in modes:
        incumbent=merge(collected[mode,'retained_route'])
        candidate=merge(collected[mode,'fast16'])
        report[mode]={'candidate':summarize(candidate),'incumbent':summarize(incumbent),
                      'paired_gain':paired_interval(candidate,incumbent),
                      'block_points_candidate':candidate['block_points'],
                      'block_points_incumbent':incumbent['block_points']}
    report['provenance']={'development_only':True,'fixed_configuration':FIXED,
        'selected_on':'112M development league placement points, incumbent eligible',
        'seed_start':a.seed_start,'blocks':a.blocks,'workers':a.workers,
        'budget_seconds':.15,'min_trials':4,
        'policy_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
        'actual_hidden_state_or_game_seed_policy_inputs':False,
        'rival_prior':'all hypothetical rivals use retained route; approximate belief',
        'uncertainty':'sampled-model lower bound is heuristic, not real-game guarantee',
        'heuristic_regime_alias':'identical to starter under canonical legal order; evaluated once'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k]['paired_gain'] for k in modes}),flush=True)


if __name__=='__main__':main()
