"""Compare training losses on one common actual-game selection set.

Every supervised pair, including zero observed gaps, has positive uncertainty
weight. No held-out game outcomes are used for training or checkpoint fitting.
Selection precedes one fresh final benchmark; baseline is eligible to win.
"""
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
from uno_policy import INITIAL_WEIGHTS
from train_uno_league import evaluate_job, paired_interval, summarize


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--incumbent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--selection-blocks',type=int,default=256)
    p.add_argument('--test-blocks',type=int,default=512)
    p.add_argument('--selection-seed-start',type=int,default=58_000_000)
    p.add_argument('--test-seed-start',type=int,default=59_000_000)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    incumbent_document=json.loads(a.incumbent.read_text())
    if incumbent_document['format']!='triad-uno-route-linear-v1':raise ValueError('expected retained route')
    incumbent={'kind':'route','weights':incumbent_document['weights']}
    specs=[incumbent];descriptions=[{'objective':'retained_route'}];models=[]
    for objective in ('ranking','mse','huber'):
        out=a.output/objective
        subprocess.run([sys.executable,str(Path(__file__).with_name('train_uno_residual.py')),
            '--data',str(a.data),'--device',a.device,'--width','64','--seed','20261022',
            '--epochs',str(a.epochs),'--objective',objective,'--pair-gap','0',
            '--uncertainty-only-weights','--calibrate-base','--fit-only',
            '--residual-scales','0.25,0.5,1','--output',str(out)],check=True,timeout=1200)
        for i in range(9):
            document=json.loads((out/f'candidate-{i}.json').read_text())
            specs.append({'kind':'mlp','model':document});models.append(document)
            descriptions.append({'objective':objective,'regularization':document['training']['regularization'],
                'residual_scale':document['training']['residual_scale'],'candidate_index':i,
                'base_calibration_scale':document['training']['base_calibration_scale']})
    attack=list(INITIAL_WEIGHTS)
    for i in (2,4,6,10):attack[i]+=2.
    preserve=list(INITIAL_WEIGHTS);preserve[5]-=2.;preserve[6]-=2.
    reference=[{'kind':'starter'},{'kind':'heuristic'},{'kind':'random'},incumbent,
        {'kind':'v1','weights':INITIAL_WEIGHTS},{'kind':'v1','weights':attack},{'kind':'v1','weights':preserve}]
    seeds=list(range(a.selection_seed_start,a.selection_seed_start+a.selection_blocks))
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=list(pool.map(evaluate_job,[(s,seeds,reference,'league') for s in specs]))
        chosen=max(range(len(results)),key=lambda i:results[i]['points'])
        selection={'selected':chosen,'selected_description':descriptions[chosen],
            'scores':[summarize(r) for r in results],'descriptions':descriptions,
            'paired_gains':[paired_interval(r,results[0]) for r in results[1:]],
            'seed_start':a.selection_seed_start,'blocks':a.selection_blocks,
            'warning':'Selection maximizes noisy measured point score; final set is independent.'}
        (a.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        print(json.dumps({'selected':chosen,'description':descriptions[chosen],'score':summarize(results[chosen]),'baseline':summarize(results[0])}),flush=True)
        document=models[chosen-1] if chosen else incumbent_document
        (a.output/'selected-policy.json').write_text(json.dumps(document,indent=2)+'\n')
        selected=specs[chosen]
        seeds=list(range(a.test_seed_start,a.test_seed_start+a.test_blocks))
        modes=('league','starter','heuristic','random')
        results=list(pool.map(evaluate_job,[(s,seeds,reference,m) for m in modes for s in (selected,incumbent)]))
        report={m:{'candidate':summarize(results[2*i]),'incumbent':summarize(results[2*i+1]),
            'paired_gain':paired_interval(results[2*i],results[2*i+1])} for i,m in enumerate(modes)}
        report['provenance']={'selection_seed':a.selection_seed_start,'final_seed':a.test_seed_start,
            'selected_description':descriptions[chosen],'selection_candidates':len(specs),
            'test_blocks':a.test_blocks,'training_pairs':'all zero gaps retained; positive 1/(1+paired_standard_error_squared) weights',
            'base_calibration':'positive least-squares scale fitted on training pairs only',
            'objectives':'same source-game split, architecture, initialization seed and batch schedule'}
        (a.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
