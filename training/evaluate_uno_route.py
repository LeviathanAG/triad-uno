"""Independent evaluation of one fixed chosen UNO route checkpoint."""
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
from uno_policy import INITIAL_WEIGHTS
from train_uno_league import evaluate_job, summarize, paired_interval


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--incumbent',type=Path,required=True)
    p.add_argument('--blocks',type=int,default=1024);p.add_argument('--seed-start',type=int,default=26_000_000);p.add_argument('--workers',type=int,default=4);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();c=json.loads(a.candidate.read_text());b=json.loads(a.incumbent.read_text())
    candidate={'kind':'route','weights':c['weights']};incumbent={'kind':'v1','weights':b['weights']}
    attack=list(INITIAL_WEIGHTS)
    for i in (2,4,6,10):attack[i]+=2.
    preserve=list(INITIAL_WEIGHTS);preserve[5]-=2.;preserve[6]-=2.
    reference=[{'kind':'starter'},{'kind':'heuristic'},{'kind':'random'},incumbent,{'kind':'v1','weights':INITIAL_WEIGHTS},{'kind':'v1','weights':attack},{'kind':'v1','weights':preserve}]
    seeds=list(range(a.seed_start,a.seed_start+a.blocks));modes=('league','starter','heuristic','random')
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=list(pool.map(evaluate_job,[(s,seeds,reference,m) for m in modes for s in (candidate,incumbent)]))
    report={m:{'candidate':summarize(results[2*i]),'incumbent':summarize(results[2*i+1]),'paired_gain':paired_interval(results[2*i],results[2*i+1])} for i,m in enumerate(modes)}
    report['provenance']={'seed_start':a.seed_start,'blocks':a.blocks,'candidate_path':str(a.candidate),'incumbent_path':str(a.incumbent),'fixed_checkpoint':True}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
