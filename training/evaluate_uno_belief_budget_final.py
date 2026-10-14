"""Fresh fixed 118M-selected belief gate and separate own-trained crossplay.

Candidate weights/configuration are frozen before any 119M result is read.
Trained rivals are our own archives; no competitor source or private state.
"""
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
from compare_uno_belief_causes import job
from compare_uno_belief_budget import CONFIGS
from evaluate_uno_belief_fast16 import merge
from train_uno_league import paired_interval,summarize,make_policy
from uno_policy import INITIAL_WEIGHTS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True);p.add_argument('--v1',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--trained-pool',type=Path,required=True)
    p.add_argument('--blocks',type=int,default=512);p.add_argument('--seed-start',type=int,default=119_000_000)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();selection=json.loads(a.selection.read_text());fixed=selection['config']
    if fixed not in CONFIGS[1:]:raise ValueError('configuration must be from the predeclared118M grid')
    weights=json.loads(a.model.read_text())['weights'];v1=json.loads(a.v1.read_text())['weights']
    attack=list(INITIAL_WEIGHTS)
    for i in (2,4,6,10):attack[i]+=2.
    preserve=list(INITIAL_WEIGHTS);preserve[5]-=2.;preserve[6]-=2.
    original=[{'kind':'starter'},{'kind':'heuristic'},{'kind':'random'},
              {'kind':'route','weights':weights},{'kind':'v1','weights':v1},
              {'kind':'v1','weights':INITIAL_WEIGHTS},{'kind':'v1','weights':attack},
              {'kind':'v1','weights':preserve}]
    trained=json.loads(a.trained_pool.read_text())
    if len(trained)<4 or any(x['kind'] not in ('route','v1','mlp','mlp_python') for x in trained):
        raise ValueError('own-trained diverse pool required')
    for i,spec in enumerate(trained):make_policy(spec,i)
    chunks=[list(range(s,min(s+32,a.seed_start+a.blocks)))
            for s in range(a.seed_start,a.seed_start+a.blocks,32)]
    modes=('league','starter','random','trained-league');tasks=[];keys=[]
    for mode in modes:
        reference=trained if mode=='trained-league' else original
        for i,config in enumerate(({'name':'retained_route'},fixed)):
            for chunk in chunks:
                keys.append((mode,i));tasks.append((config,weights,chunk,reference,mode))
    collected={k:[] for k in keys}
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        for index,(key,result) in enumerate(zip(keys,pool.map(job,tasks))):
            collected[key].append(result)
            print(json.dumps({'completed':index+1,'total_chunks':len(tasks),'mode':key[0],
                              'candidate':bool(key[1]),'probe_stats':result['probe_stats']}),flush=True)
    report={}
    for mode in modes:
        incumbent=merge(collected[mode,0]);candidate=merge(collected[mode,1])
        report[mode]={'candidate':summarize(candidate),'incumbent':summarize(incumbent),
                      'paired_gain':paired_interval(candidate,incumbent),
                      'block_points_candidate':candidate['block_points'],
                      'block_points_incumbent':incumbent['block_points']}
    report['provenance']={'development_only':True,'fixed_selection':selection,
        'seed_start':a.seed_start,'blocks':a.blocks,'workers':a.workers,
        'actual_hidden_state_or_game_seed_policy_inputs':False,
        'policy_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
        'trained_pool_sha256':hashlib.sha256(a.trained_pool.read_bytes()).hexdigest(),
        'trained_rivals':[{'name':x['name'],'kind':x['kind']} for x in trained],
        'trained_schedule':'three distinct own-trained rivals sampled per block, four seat rotations',
        'scoring':'5/3/2/1; unfinished seats share remaining placement points at actual turn limit',
        'scope':'original league/starter/random retained; trained-league reported separately',
        'censoring':'Original fixed candidate discards incomplete128-step paired sampled rollouts; actual test games run to own confirmed finish or1000turn limit',
        'rival_implementation':'mlp rivals use training NumPy FastPolicy, not a claim of exact exported float scoring'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({mode:report[mode]['paired_gain'] for mode in modes}),flush=True)


if __name__=='__main__':main()
