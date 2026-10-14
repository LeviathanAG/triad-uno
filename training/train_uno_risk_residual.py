"""Frozen24 plus six response residuals against our diverse trained policies.

Training/selection/final families are separate. Conventional non-language ML;
policies receive filtered observations only. No competing private source.
"""
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import time
import numpy as np
from train_uno_league import engine,make_policy,paired_interval,summarize
from uno_risk_policy import Policy,FEATURE_NAMES


def evaluate(args):
    residual,base,seeds,reference,mode=args;points=[];wins=[];declines=0
    started=time.perf_counter()
    for seed in seeds:
        rng=random.Random(seed^0x5354524F4E47)
        rivals=rng.sample(reference,3) if mode=='trained' else ([{'kind':'route','weights':base}]*3 if mode=='route' else [{'kind':mode}]*3)
        outcomes=[]
        for seat in range(4):
            actors=[None]*4;candidate=Policy(base,residual);actors[seat]=candidate
            for offset,spec in enumerate(rivals,1):actors[(seat+offset)%4]=make_policy(spec,seed*10+offset)
            state=engine.new_game(engine.GameConfig(max_turns=1000),seed)
            while state.phase!=engine.FINISHED and seat not in state.finished_set:
                current=state.current_seat;observation=engine.observe(state,current)
                selected=actors[current].select(observation)
                action=engine.engine_action_from_id(state,current,selected)
                if action is None or not engine.step(state,action).ok:raise RuntimeError('illegal risk training decision')
            if seat in state.finished_order:
                pos=state.finished_order.index(seat);outcomes.append((state.config.position_points[pos],float(pos==0)))
            else:
                remaining=state.config.position_points[len(state.outcome.placements):]
                outcomes.append((sum(remaining)/len(state.outcome.unfinished),0.))
            declines+=candidate.declines
        points.append(sum(x[0] for x in outcomes)/4);wins.append(sum(x[1] for x in outcomes)/4)
    return {'points':float(np.mean(points)),'win_rate':float(np.mean(wins)),
            'games':len(seeds)*4,'block_points':points,'block_wins':wins,
            'public_prior_declines':declines,'seconds':time.perf_counter()-started}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--incumbent',type=Path,required=True);p.add_argument('--trained-pool',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=24);p.add_argument('--population',type=int,default=16)
    p.add_argument('--train-blocks',type=int,default=32);p.add_argument('--selection-blocks',type=int,default=256)
    p.add_argument('--test-blocks',type=int,default=512);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--train-seed-start',type=int,default=140_000_000)
    p.add_argument('--selection-seed-start',type=int,default=141_000_000)
    p.add_argument('--test-seed-start',type=int,default=142_000_000)
    p.add_argument('--seed',type=int,default=20261029);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if min(a.iterations,a.train_blocks,a.selection_blocks,a.test_blocks,a.workers)<1 or a.population<4:p.error('positive counts/population>=4 required')
    if a.train_seed_start+a.iterations*a.train_blocks>a.selection_seed_start or a.selection_seed_start+a.selection_blocks>a.test_seed_start:p.error('seed families overlap')
    a.output.mkdir(parents=True,exist_ok=True)
    incumbent=json.loads(a.incumbent.read_text());base=incumbent['weights']
    if len(base)!=24:raise ValueError('retained24-weight prior required')
    pool=json.loads(a.trained_pool.read_text())
    if len(pool)<4 or any(s['kind'] not in ('route','v1','mlp','mlp_python') for s in pool):raise ValueError('own-trained diverse pool required')
    for i,s in enumerate(pool):make_policy(s,i)
    rng=np.random.default_rng(a.seed);mean=np.zeros(6);sigma=np.full(6,2.)
    zero=[0.]*6;finalists=[zero];history=[];started=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as executor:
        for generation in range(a.iterations):
            population=np.clip(rng.normal(mean,sigma,(a.population,6)),-8.,8.)
            population[0]=mean;population[1]=zero
            seeds=list(range(a.train_seed_start+generation*a.train_blocks,a.train_seed_start+(generation+1)*a.train_blocks))
            results=list(executor.map(evaluate,[(w.tolist(),base,seeds,pool,'trained') for w in population]))
            order=sorted(range(a.population),key=lambda i:results[i]['points'],reverse=True)
            elite=population[order[:max(2,a.population//4)]]
            mean=np.clip(.3*mean+.7*elite.mean(axis=0),-8.,8.)
            sigma=np.clip(.3*sigma+.7*elite.std(axis=0),.25,3.)
            best=population[order[0]].tolist()
            if generation%4==3 and not any(np.linalg.norm(np.asarray(x)-best)<.5 for x in finalists):finalists.append(best)
            row={'generation':generation+1,'residual_weights':best,'best':summarize(results[order[0]]),
                 'zero_prior':summarize(results[1]),'seconds':time.perf_counter()-started}
            history.append(row);print(json.dumps(row),flush=True)
            (a.output/'training-progress.json').write_text(json.dumps(history,indent=2)+'\n')
        if not any(np.linalg.norm(np.asarray(x)-mean)<.5 for x in finalists):finalists.append(mean.tolist())
        seeds=list(range(a.selection_seed_start,a.selection_seed_start+a.selection_blocks))
        results=list(executor.map(evaluate,[(w,base,seeds,pool,'trained') for w in finalists]))
        chosen=max(range(len(finalists)),key=lambda i:results[i]['points'])
        selection={'selected':chosen,'residual_weights':finalists,'scores':[summarize(x) for x in results],
                   'paired_gains':[paired_interval(x,results[0]) for x in results[1:]],
                   'seed_start':a.selection_seed_start,'blocks':a.selection_blocks,
                   'criterion':'own-trained distinct-rival league placement points;zero eligible'}
        (a.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        model={'format':'triad-uno-route-unseen-risk-v1','features':FEATURE_NAMES,'base_weights':base,
               'residual_weights':finalists[chosen],
               'training':{'frozen_base':True,'residual_dimensions':6,'iterations':a.iterations,
                           'population':a.population,'train_blocks':a.train_blocks,'bounds':[-8.,8.],
                           'seed':a.seed,'train_seed_start':a.train_seed_start,
                           'selection_seed_start':a.selection_seed_start,'test_seed_start':a.test_seed_start,
                           'incumbent_sha256':hashlib.sha256(a.incumbent.read_bytes()).hexdigest(),
                           'trained_pool_sha256':hashlib.sha256(a.trained_pool.read_bytes()).hexdigest(),
                           'rival_names':[s['name'] for s in pool],
                           'belief_approximation':'uniform per-seat hard-constraint prior;not joint opponent model'}}
        (a.output/'selected-policy.json').write_text(json.dumps(model,indent=2)+'\n')
        seeds=list(range(a.test_seed_start,a.test_seed_start+a.test_blocks));modes=('trained','route','starter','random')
        results=list(executor.map(evaluate,[(w,base,seeds,pool,mode) for mode in modes for w in (finalists[chosen],zero)]))
        report={mode:{'candidate':summarize(results[2*i]),'incumbent':summarize(results[2*i+1]),
                      'paired_gain':paired_interval(results[2*i],results[2*i+1]),
                      'block_points_candidate':results[2*i]['block_points'],
                      'block_points_incumbent':results[2*i+1]['block_points']} for i,mode in enumerate(modes)}
        report['provenance']={'fixed_selection':chosen,'seed_start':a.test_seed_start,'blocks':a.test_blocks,
                              'frozen24_sha256':model['training']['incumbent_sha256'],
                              'rival_implementations':'mlp uses training NumPy FastPolicy; not exact exported float scoring',
                              'actual_hidden_state_or_game_seed_policy_inputs':False,
                              'independent_final_family_if_supported':143_000_000}
        (a.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'selection':chosen,'final':{m:report[m]['paired_gain'] for m in modes}}),flush=True)


if __name__=='__main__':main()
