"""Low-dimensional residual league training for own-hand UNO planning features."""
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from uno_policy import INITIAL_WEIGHTS
from uno_route_policy import FEATURE_NAMES, EXTRA_NAMES
from train_uno_league import evaluate_job, summarize, paired_interval


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--incumbent',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=24)
    p.add_argument('--population',type=int,default=24)
    p.add_argument('--train-blocks',type=int,default=24)
    p.add_argument('--validation-blocks',type=int,default=256)
    p.add_argument('--test-blocks',type=int,default=512)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261019)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if min(a.iterations,a.population,a.train_blocks,a.validation_blocks,a.test_blocks,a.workers)<1:p.error('positive counts required')
    a.output.mkdir(parents=True,exist_ok=True)
    base=json.loads(a.incumbent.read_text())
    incumbent={'kind':'v1','weights':base['weights']}
    attack=list(INITIAL_WEIGHTS)
    for i in (2,4,6,10):attack[i]+=2.
    preserve=list(INITIAL_WEIGHTS);preserve[5]-=2.;preserve[6]-=2.
    reference=[{'kind':'starter'},{'kind':'heuristic'},{'kind':'random'},incumbent,
               {'kind':'v1','weights':INITIAL_WEIGHTS},{'kind':'v1','weights':attack},{'kind':'v1','weights':preserve}]
    rng=np.random.default_rng(a.seed)
    mean=np.zeros(len(EXTRA_NAMES));sigma=np.full(len(mean),1.5)
    finalists=[mean.tolist()];history=[];started=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for gen in range(a.iterations):
            population=rng.normal(mean,sigma,(a.population,len(mean)))
            population=np.clip(population,-8.,8.)
            population[0]=mean;population[1]=0.
            if gen % 6==0:
                population[-1]=[2.,-2.,1.,1.,.5,-1.]
            seeds=list(range(20_000_000+gen*a.train_blocks,20_000_000+(gen+1)*a.train_blocks))
            specs=[{'kind':'route','weights':base['weights']+w.tolist()} for w in population]
            results=list(pool.map(evaluate_job,[(s,seeds,reference,'league') for s in specs]))
            order=sorted(range(len(results)),key=lambda i:(results[i]['points'],results[i]['win_rate']),reverse=True)
            elite=population[order[:max(2,a.population//5)]]
            mean=.3*mean+.7*elite.mean(axis=0);sigma=np.maximum(.35,.3*sigma+.7*elite.std(axis=0))
            candidate=population[order[0]].tolist()
            if gen % 3==2 and not any(np.linalg.norm(np.asarray(c)-candidate)<.2 for c in finalists):finalists.append(candidate)
            row={'generation':gen+1,'train_score':summarize(results[order[0]]),'extra_weights':candidate,'seconds':round(time.time()-started,1)}
            history.append(row);print(json.dumps(row),flush=True)
            (a.output/'training-progress.json').write_text(json.dumps(history,indent=2)+'\n')
        if not any(np.linalg.norm(np.asarray(c)-mean)<.2 for c in finalists):finalists.append(mean.tolist())
        # All finalists get the same single selection set. The final set is fresh.
        specs=[incumbent]+[{'kind':'route','weights':base['weights']+w} for w in finalists[1:]]
        seeds=list(range(21_000_000,21_000_000+a.validation_blocks))
        results=list(pool.map(evaluate_job,[(s,seeds,reference,'league') for s in specs]))
        chosen=max(range(len(results)),key=lambda i:results[i]['points'])
        selection={'selected':chosen,'scores':[summarize(r) for r in results],'paired_gains':[paired_interval(r,results[0]) for r in results[1:]],'specs':specs}
        (a.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n');print(json.dumps({'selection':selection}),flush=True)
        selected=specs[chosen]
        document={'format':'triad-uno-route-linear-v1','features':FEATURE_NAMES,
                  'weights':selected['weights']+[0.]*(len(FEATURE_NAMES)-len(selected['weights'])),
                  'training':{'method':'frozen-base residual CEM, own-hand bounded sequence features','seed':a.seed,'iterations':a.iterations,
                              'sdk_commit':'0cda3acc1bb5c05fa1803a8fbd7761770dc6d5fe','selection_seed_start':21_000_000,'test_seed_start':22_000_000,'base':base}}
        (a.output/'uno-policy-route.json').write_text(json.dumps(document,indent=2)+'\n')
        modes=('league','starter','heuristic','random');seeds=list(range(22_000_000,22_000_000+a.test_blocks))
        results=list(pool.map(evaluate_job,[(s,seeds,reference,m) for m in modes for s in (selected,incumbent)]))
        report={m:{'candidate':summarize(results[2*i]),'incumbent':summarize(results[2*i+1]),'paired_gain':paired_interval(results[2*i],results[2*i+1])} for i,m in enumerate(modes)}
        (a.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'held_out_evaluation':report}),flush=True)


if __name__=='__main__':main()
