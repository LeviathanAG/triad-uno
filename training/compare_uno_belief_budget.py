"""New 118M development search for action count and bounded rollout budget.

Original 113M gate is a completed failure and is never reinterpreted here.
Official referee is a development-only simulator of own/public sampled worlds.
"""
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
from compare_uno_belief_causes import job
from evaluate_uno_belief_fast16 import merge
from train_uno_league import paired_interval,summarize
from uno_policy import INITIAL_WEIGHTS

CONFIGS=[{'name':'retained_route'}]+[
    {'name':f'budget500ms_actions{count}_{prior}','budget':.5,'max_trials':32,
     'max_actions':count,'fast_simulation':True,'cached_public_fold':True,
     'rival_model':prior}
    for count in (4,2) for prior in ('route','mixture')]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True);p.add_argument('--v1',type=Path,required=True)
    p.add_argument('--blocks',type=int,default=64);p.add_argument('--seed-start',type=int,default=118_000_000)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();weights=json.loads(a.model.read_text())['weights'];v1=json.loads(a.v1.read_text())['weights']
    attack=list(INITIAL_WEIGHTS)
    for i in (2,4,6,10):attack[i]+=2.
    preserve=list(INITIAL_WEIGHTS);preserve[5]-=2.;preserve[6]-=2.
    reference=[{'kind':'starter'},{'kind':'heuristic'},{'kind':'random'},
               {'kind':'route','weights':weights},{'kind':'v1','weights':v1},
               {'kind':'v1','weights':INITIAL_WEIGHTS},
               {'kind':'v1','weights':attack},{'kind':'v1','weights':preserve}]
    chunks=[list(range(s,min(s+16,a.seed_start+a.blocks)))
            for s in range(a.seed_start,a.seed_start+a.blocks,16)]
    modes=('league','starter','random');tasks=[];keys=[]
    for mode in modes:
        for i,config in enumerate(CONFIGS):
            for chunk in chunks:
                keys.append((mode,i));tasks.append((config,weights,chunk,reference,mode))
    collected={k:[] for k in keys}
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        for index,(key,result) in enumerate(zip(keys,pool.map(job,tasks))):
            collected[key].append(result)
            print(json.dumps({'completed':index+1,'total_chunks':len(tasks),
                              'mode':key[0],'config':CONFIGS[key[1]],
                              'probe_stats':result['probe_stats']}),flush=True)
    report={};means=[]
    for mode in modes:
        rows=[merge(collected[mode,i]) for i in range(len(CONFIGS))]
        if mode=='league':means=[r['points'] for r in rows]
        report[mode]={'incumbent':summarize(rows[0]),
                      'candidates':[{'config':config,'summary':summarize(row),
                                     'paired_gain':paired_interval(row,rows[0])}
                                    for config,row in zip(CONFIGS[1:],rows[1:])]}
    chosen=max(range(len(CONFIGS)),key=lambda i:means[i])
    report['selection']={'selected':chosen,'config':CONFIGS[chosen],
                          'criterion':'common development league placement points; incumbent eligible'}
    report['provenance']={'development_only':True,'seed_start':a.seed_start,
        'blocks':a.blocks,'workers':a.workers,'min_trials':4,'max_steps':128,
        'policy_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
        'actual_hidden_state_or_game_seed_policy_inputs':False,
        'future_final_seed_if_candidate_selected':119_000_000,
        '113M_gate':'prior fixed fast16 failed and remains recorded without reinterpretation',
        'approximation':'public-constraint hypothetical hands; route or uniform style-mixture rival prior'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
