"""Isolated source-diversity diagnostic, never a deployment selection gate.

Each own-trained archive policy faces three copies of each fixed rival on
shared four-seat blocks. This diagnoses pool composition; it cannot establish
strength against unavailable event competitors or solve a multiplayer game.
"""
import argparse,concurrent.futures,json,multiprocessing,time
from pathlib import Path
import numpy as np
from train_uno_league import score_game


def evaluate(args):
    candidate,rival,seeds=args;points=[];wins=[];started=time.perf_counter()
    for seed in seeds:
        values=[score_game(seed,seat,candidate,[rival]*3) for seat in range(4)]
        points.append(sum(x[0] for x in values)/4)
        wins.append(sum(x[1] for x in values)/4)
    return {'points':float(np.mean(points)),'wins':float(np.mean(wins)),
            'block_points':points,'block_wins':wins,'games':4*len(seeds),
            'seconds':time.perf_counter()-started}


def main():
    p=argparse.ArgumentParser();p.add_argument('--pool',type=Path,required=True)
    p.add_argument('--blocks',type=int,default=16);p.add_argument('--seed-start',type=int,default=161000000)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();pool=json.loads(a.pool.read_text());seeds=list(range(a.seed_start,a.seed_start+a.blocks))
    tasks=[(candidate,rival,seeds) for candidate in pool for rival in pool];rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,
            mp_context=multiprocessing.get_context('spawn')) as executor:
        for i,row in enumerate(executor.map(evaluate,tasks)):
            rows.append(row);print(json.dumps({'completed':i+1,'total':len(tasks),'points':row['points']}),flush=True)
    n=len(pool);matrix=[[rows[i*n+j]['points'] for j in range(n)] for i in range(n)]
    report={'seed_start':a.seed_start,'blocks':a.blocks,'names':[x['name'] for x in pool],
            'matrix_points':matrix,'matchups':rows,
            'mean_over_homogeneous_rivals':np.mean(matrix,axis=1).tolist(),
            'scope':'own public/learned archive only; three identical rivals per matchup',
            'not_a_fresh_deployment_or_event_strength_gate':True,
            'rival_implementation':'MLP archive actors use training NumPy FastPolicy'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
