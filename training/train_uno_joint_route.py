"""Warm-start trust-bounded joint optimization of all24 own/public route weights.

UNO's conventional model has no frozen-base requirement. Shared four-seat blocks
reduce seat/deal variance; this practical search is not a proof of optimality.
"""
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from uno_policy import INITIAL_WEIGHTS
from uno_route_policy import FEATURE_NAMES
from uno_mlp_policy import Policy as NeuralPolicy
from train_uno_league import evaluate_job,summarize,paired_interval


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--incumbent',type=Path,required=True)
    p.add_argument('--v1',type=Path,required=True)
    p.add_argument('--neural-rival',type=Path,action='append',default=[])
    p.add_argument('--iterations',type=int,default=24)
    p.add_argument('--population',type=int,default=24)
    p.add_argument('--train-blocks',type=int,default=32)
    p.add_argument('--selection-blocks',type=int,default=256)
    p.add_argument('--test-blocks',type=int,default=512)
    p.add_argument('--train-seed-start',type=int,default=80_000_000)
    p.add_argument('--selection-seed-start',type=int,default=81_000_000)
    p.add_argument('--test-seed-start',type=int,default=82_000_000)
    p.add_argument('--seed',type=int,default=20261023)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--trust-fraction',type=float,default=.25)
    p.add_argument('--initial-sigma-fraction',type=float,default=.10)
    p.add_argument('--mirrored',action='store_true',help='future experiment:paired opposite perturbations before trust-box clipping')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if min(a.iterations,a.population,a.train_blocks,a.selection_blocks,a.test_blocks,a.workers)<1 or a.population<4:p.error('positive counts/population>=4 required')
    if not 0<a.initial_sigma_fraction<=a.trust_fraction<=.5:p.error('bounded positive trust configuration required')
    if a.mirrored and a.population%2:p.error('mirrored search requires even population including two reserved candidates')
    if not a.train_seed_start+a.iterations*a.train_blocks<=a.selection_seed_start or not a.selection_seed_start+a.selection_blocks<=a.test_seed_start:p.error('seed families overlap')
    a.output.mkdir(parents=True,exist_ok=True)
    original=json.loads(a.incumbent.read_text())
    if original['format']!='triad-uno-route-linear-v1' or original['features']!=FEATURE_NAMES:raise ValueError('retained route schema required')
    center=np.asarray(original['weights'],dtype=float)
    if center.shape!=(24,) or not np.isfinite(center).all():raise ValueError('invalid incumbent')
    incumbent={'kind':'route','weights':center.tolist()}
    v1=json.loads(a.v1.read_text())
    if v1['format']!='triad-uno-linear-v1' or len(v1['weights'])!=18:raise ValueError('learnedv1 rival required')
    attack=list(INITIAL_WEIGHTS)
    for i in (2,4,6,10):attack[i]+=2.
    preserve=list(INITIAL_WEIGHTS);preserve[5]-=2.;preserve[6]-=2.
    reference=[{'kind':'starter'},{'kind':'heuristic'},{'kind':'random'},incumbent,
        {'kind':'v1','weights':v1['weights']},{'kind':'v1','weights':INITIAL_WEIGHTS},
        {'kind':'v1','weights':attack},{'kind':'v1','weights':preserve}]
    rival_documents=[]
    for path in a.neural_rival:
        document=json.loads(path.read_text());NeuralPolicy(document)
        rival_documents.append(document);reference.append({'kind':'mlp','model':document})
    (a.output/'rival-pool.json').write_text(json.dumps(reference,indent=2)+'\n')
    scale=np.maximum(1.,np.abs(center));radius=a.trust_fraction*scale
    lower,upper=center-radius,center+radius
    mean=center.copy();sigma=a.initial_sigma_fraction*scale
    rng=np.random.default_rng(a.seed);finalists=[center.tolist()];history=[];started=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for generation in range(a.iterations):
            if a.mirrored:
                noise=rng.normal(0.,sigma,((a.population-2)//2,24))
                population=np.empty((a.population,24))
                population[2::2]=np.clip(mean+noise,lower,upper)
                population[3::2]=np.clip(mean-noise,lower,upper)
            else:
                population=np.clip(rng.normal(mean,sigma,(a.population,24)),lower,upper)
            population[0]=mean;population[1]=center
            specs=[{'kind':'route','weights':row.tolist()} for row in population]
            seeds=list(range(a.train_seed_start+generation*a.train_blocks,a.train_seed_start+(generation+1)*a.train_blocks))
            results=list(pool.map(evaluate_job,[(s,seeds,reference,'league') for s in specs]))
            order=sorted(range(len(results)),key=lambda i:results[i]['points'],reverse=True)
            elite=population[order[:max(2,a.population//5)]]
            mean=np.clip(.3*mean+.7*elite.mean(axis=0),lower,upper)
            sigma=np.minimum(radius/2,np.maximum(.03*scale,.3*sigma+.7*elite.std(axis=0)))
            best=population[order[0]]
            if generation%3==2 and not any(np.linalg.norm((np.asarray(w)-best)/scale)<.12 for w in finalists):finalists.append(best.tolist())
            row={'generation':generation+1,'best_points':summarize(results[order[0]]),'incumbent_points':summarize(results[1]),
                 'weights':best.tolist(),'normalized_distance':float(np.linalg.norm((best-center)/scale)),
                 'seconds':round(time.time()-started,1)}
            history.append(row);print(json.dumps(row),flush=True)
            (a.output/'training-progress.json').write_text(json.dumps(history,indent=2)+'\n')
        if not any(np.linalg.norm((np.asarray(w)-mean)/scale)<.12 for w in finalists):finalists.append(mean.tolist())
        specs=[{'kind':'route','weights':w} for w in finalists]
        seeds=list(range(a.selection_seed_start,a.selection_seed_start+a.selection_blocks))
        results=list(pool.map(evaluate_job,[(s,seeds,reference,'league') for s in specs]))
        chosen=max(range(len(results)),key=lambda i:results[i]['points'])
        selection={'selected':chosen,'seed_start':a.selection_seed_start,'blocks':a.selection_blocks,'specs':specs,
            'scores':[summarize(r) for r in results],'paired_gains':[paired_interval(r,results[0]) for r in results[1:]]}
        (a.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n');print(json.dumps({'selection':selection}),flush=True)
        document={'format':'triad-uno-route-linear-v1','features':FEATURE_NAMES,'weights':specs[chosen]['weights'],
            'training':{'method':'warm-start joint24-coefficient trust-bounded CEM','frozen_base':False,'seed':a.seed,
            'iterations':a.iterations,'population':a.population,'train_blocks_per_generation':a.train_blocks,
            'trust_fraction':a.trust_fraction,'initial_sigma_fraction':a.initial_sigma_fraction,
            'mirrored_before_clipping':a.mirrored,
            'lower_bounds':lower.tolist(),'upper_bounds':upper.tolist(),'train_seed_start':a.train_seed_start,
            'selection_seed_start':a.selection_seed_start,'test_seed_start':a.test_seed_start,
            'rival_pool_size':len(reference),'neural_rival_formats':[m['format'] for m in rival_documents],
            'base_checkpoint':original,'sdk_commit':'0cda3acc1bb5c05fa1803a8fbd7761770dc6d5fe'}}
        (a.output/'selected-policy.json').write_text(json.dumps(document,indent=2)+'\n')
        seeds=list(range(a.test_seed_start,a.test_seed_start+a.test_blocks));modes=('league','starter','heuristic','random')
        results=list(pool.map(evaluate_job,[(s,seeds,reference,m) for m in modes for s in (specs[chosen],incumbent)]))
        report={m:{'candidate':summarize(results[2*i]),'incumbent':summarize(results[2*i+1]),'paired_gain':paired_interval(results[2*i],results[2*i+1])} for i,m in enumerate(modes)}
        report['provenance']={'fixed_selected_candidate':chosen,'train_seed_start':a.train_seed_start,
            'selection_seed_start':a.selection_seed_start,'test_seed_start':a.test_seed_start,
            'test_blocks':a.test_blocks,'rival_pool_size':len(reference),'joint_dimensions':24,
            'mirrored_before_clipping':a.mirrored,
            'trust_fraction':a.trust_fraction,'strength_warning':'New richer rival pool; compare paired incumbent within this gate only.'}
        (a.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'final':report}),flush=True)


if __name__=='__main__':main()
