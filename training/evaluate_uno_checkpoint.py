"""Fresh fixed-checkpoint tests against the actual retained UNO policy."""
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
from uno_policy import INITIAL_WEIGHTS
from train_uno_league import evaluate_job, summarize, paired_interval


def spec(document):
    fmt=document['format']
    if fmt in ('triad-uno-residual-mlp-v1','triad-uno-residual-mlp-v2','triad-uno-residual-mlp-v3'):
        return {'kind':'mlp','model':document}
    kind={'triad-uno-linear-v1':'v1','triad-uno-route-linear-v1':'route','triad-uno-linear-v2':'v2'}.get(fmt)
    if kind is None:raise ValueError('unsupported policy format')
    return {'kind':kind,'weights':document['weights']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--incumbent',type=Path,required=True)
    p.add_argument('--source-incumbent',type=Path,required=True);p.add_argument('--blocks',type=int,default=2048)
    p.add_argument('--seed-start',type=int,default=45_000_000);p.add_argument('--workers',type=int,default=4);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate-pure-python',action='store_true',help='final intended-runtime neural scorer rather than NumPy acceleration')
    a=p.parse_args();candidate=spec(json.loads(a.candidate.read_text()));incumbent=spec(json.loads(a.incumbent.read_text()));source=spec(json.loads(a.source_incumbent.read_text()))
    if a.candidate_pure_python:
        if candidate['kind']!='mlp':p.error('pure-Python mode requires a neural candidate')
        candidate['kind']='mlp_python'
    attack=list(INITIAL_WEIGHTS)
    for i in (2,4,6,10):attack[i]+=2.
    preserve=list(INITIAL_WEIGHTS);preserve[5]-=2.;preserve[6]-=2.
    reference=[{'kind':'starter'},{'kind':'heuristic'},{'kind':'random'},incumbent,{'kind':'v1','weights':INITIAL_WEIGHTS},{'kind':'v1','weights':attack},{'kind':'v1','weights':preserve}]
    if source != incumbent:reference.insert(4,source)
    seeds=list(range(a.seed_start,a.seed_start+a.blocks));modes=('league','starter','heuristic','random')
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=list(pool.map(evaluate_job,[(s,seeds,reference,m) for m in modes for s in (candidate,incumbent)]))
    report={m:{'candidate':summarize(results[2*i]),'incumbent':summarize(results[2*i+1]),'paired_gain':paired_interval(results[2*i],results[2*i+1])} for i,m in enumerate(modes)}
    report['provenance']={'seed_start':a.seed_start,'blocks':a.blocks,'fixed_checkpoint':True,'candidate_format':candidate['kind'],'actual_retained_incumbent':incumbent['kind'],'candidate_pure_python':a.candidate_pure_python}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
