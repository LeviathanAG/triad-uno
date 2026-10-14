"""Own/public context-dependent Huber residual training with fresh fixed gates."""
import argparse,concurrent.futures,hashlib,json,multiprocessing,time
from pathlib import Path
import numpy as np
import evaluate_uno_frozen_huber as games
from train_uno_league import paired_interval,summarize
from uno_huber_stage_policy import Policy,STAGES
from uno_policy import INITIAL_WEIGHTS

WEIGHTS=None;CHECKPOINT=None


def initialize(reference,checkpoint):
    global WEIGHTS,CHECKPOINT
    WEIGHTS=reference['anchor']['weights'];CHECKPOINT=checkpoint
    games.initialize(reference,checkpoint)


def evaluate(args):
    strengths,seeds,mode=args
    games.CANDIDATES[1]=Policy(WEIGHTS,CHECKPOINT,strengths)
    return games.evaluate((1,seeds,mode))


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--incumbent',type=Path,required=True);p.add_argument('--trained-pool',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--iterations',type=int,default=12)
    p.add_argument('--population',type=int,default=12);p.add_argument('--train-blocks',type=int,default=48)
    p.add_argument('--selection-blocks',type=int,default=256);p.add_argument('--test-blocks',type=int,default=512)
    p.add_argument('--seed',type=int,default=20261031);p.add_argument('--train-seed-start',type=int,default=171000000)
    p.add_argument('--selection-seed-start',type=int,default=172000000);p.add_argument('--test-seed-start',type=int,default=173000000)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if min(a.workers,a.iterations,a.train_blocks,a.selection_blocks,a.test_blocks)<1 or a.population<4:p.error('positive counts/population>=4 required')
    if a.train_seed_start+a.iterations*a.train_blocks>a.selection_seed_start or a.selection_seed_start+a.selection_blocks>a.test_seed_start:p.error('overlapping seed families')
    a.output.mkdir(parents=True,exist_ok=True);checkpoint=json.loads(a.checkpoint.read_text())
    weights=json.loads(a.incumbent.read_text())['weights'];trained=json.loads(a.trained_pool.read_text())
    anchor={'name':'retained_route','kind':'route','weights':weights};attack=list(INITIAL_WEIGHTS)
    for i in (2,4,6,10):attack[i]+=2
    preserve=list(INITIAL_WEIGHTS);preserve[5]-=2;preserve[6]-=2
    v1=next(s['weights'] for s in trained if s['kind']=='v1')
    reference={'anchor':anchor,'trained':trained,'classical':[{'kind':'starter'},{'kind':'random'},
        {'kind':'v1','weights':INITIAL_WEIGHTS},{'kind':'v1','weights':attack},
        {'kind':'v1','weights':preserve},{'kind':'v1','weights':v1}]}
    rng=np.random.default_rng(a.seed);mean=np.ones(6);sigma=np.full(6,.25)
    prior=np.zeros(6);finalists=[prior.tolist()];history=[];started=time.perf_counter()
    def archive(candidate):
        # The old static Huber checkpoint failed167M. A newly fitted stage
        # candidate must actually vary by public stage to enter fresh final.
        if max(candidate)-min(candidate)<.15:return
        if not any(np.linalg.norm(np.asarray(x)-candidate)<.1 for x in finalists):finalists.append(candidate.tolist())
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(reference,checkpoint)) as executor:
        for generation in range(a.iterations):
            population=np.clip(rng.normal(mean,sigma,(a.population,6)),0,1.5)
            population[0]=mean;population[1]=prior;population[2]=np.ones(6)
            seeds=list(range(a.train_seed_start+generation*a.train_blocks,a.train_seed_start+(generation+1)*a.train_blocks))
            rows=list(executor.map(evaluate,[(x.tolist(),seeds,mode) for mode in ('trained','route') for x in population]))
            fitness=[.7*rows[i]['points']+.3*rows[a.population+i]['points'] for i in range(a.population)]
            order=sorted(range(a.population),key=lambda i:fitness[i],reverse=True)
            elite=population[order[:max(2,a.population//4)]]
            mean=np.clip(.3*mean+.7*elite.mean(axis=0),0,1.5)
            sigma=np.minimum(.5,np.maximum(.08,.3*sigma+.7*elite.std(axis=0)))
            best=population[order[0]]
            if generation%4==3:archive(best)
            row={'generation':generation+1,'strengths':best.tolist(),'composite':fitness[order[0]],
                'incumbent_composite':fitness[1],'static_huber_composite':fitness[2],
                'trained':summarize(rows[order[0]]),'route':summarize(rows[a.population+order[0]]),
                'seconds':time.perf_counter()-started}
            history.append(row);print(json.dumps(row),flush=True)
            (a.output/'training-progress.json').write_text(json.dumps(history,indent=2)+'\n')
        archive(mean)
        seeds=list(range(a.selection_seed_start,a.selection_seed_start+a.selection_blocks));modes=('trained','route','classical')
        rows=list(executor.map(evaluate,[(x,seeds,mode) for mode in modes for x in finalists]));n=len(finalists)
        grouped={m:rows[i*n:(i+1)*n] for i,m in enumerate(modes)};eligible=[0]
        for i in range(1,n):
            gains={m:grouped[m][i]['points']-grouped[m][0]['points'] for m in modes}
            if gains['trained']>=0 and gains['route']>=-.02 and gains['classical']>=-.03:eligible.append(i)
        selected=max(eligible,key=lambda i:.7*grouped['trained'][i]['points']+.3*grouped['route'][i]['points'])
        selection={'selected':selected,'strengths':finalists,'eligible':eligible,'seed_start':a.selection_seed_start,'blocks':a.selection_blocks,
            'criterion':'70% own-trained +30% retained-route points; incumbent eligible',
            'guards':{'trained_mean_gain_min':0,'route_mean_gain_min':-.02,'classical_mean_gain_min':-.03},
            'stage_range_minimum':.15,'static_huber_not_final_eligible':True,
            'results':grouped,'paired_gains':{m:[paired_interval(row,grouped[m][0]) for row in grouped[m][1:]] for m in modes}}
        (a.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        model={'format':'triad-uno-huber-stage-v1','stages':STAGES,'strengths':finalists[selected],
            'route_weights':weights,'checkpoint':checkpoint,'training':{'method':'CEM public-stage residual strengths',
                'seed':a.seed,'train_seed_start':a.train_seed_start,'selection_seed_start':a.selection_seed_start,
                'test_seed_start':a.test_seed_start,'independent_seed_reserved':174000000,
                'iterations':a.iterations,'population':a.population,'train_blocks':a.train_blocks,
                'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                'incumbent_sha256':hashlib.sha256(a.incumbent.read_bytes()).hexdigest(),
                'trained_pool_sha256':hashlib.sha256(a.trained_pool.read_bytes()).hexdigest(),
                'candidate_exact_stdlib_throughout':True,'frozen_neural_matrices':True,'strength_bounds':[0,1.5],
                'actual_hidden_or_game_seed_policy_inputs':False,'selection_guards':selection['guards']}}
        (a.output/'selected-policy.json').write_text(json.dumps(model,indent=2)+'\n')
        if selected==0:
            report={'incumbent_selected':True,'final_not_consumed':True,'reserved_seed_start':a.test_seed_start}
        else:
            seeds=list(range(a.test_seed_start,a.test_seed_start+a.test_blocks));modes=('trained','route','classical','starter','random')
            rows=list(executor.map(evaluate,[(x,seeds,mode) for mode in modes for x in (finalists[selected],prior.tolist())]))
            report={mode:{'candidate':summarize(rows[2*i]),'incumbent':summarize(rows[2*i+1]),
                'paired_gain':paired_interval(rows[2*i],rows[2*i+1]),
                'block_points_candidate':rows[2*i]['block_points'],'block_points_incumbent':rows[2*i+1]['block_points'],
                'block_wins_candidate':rows[2*i]['block_wins'],'block_wins_incumbent':rows[2*i+1]['block_wins']} for i,mode in enumerate(modes)}
            report['provenance']={'fixed_selected':selected,'seed_start':a.test_seed_start,'blocks':a.test_blocks,
                'candidate_exact_stdlib':True,'actual_hidden_or_game_seed_policy_inputs':False,
                'independent_seed_reserved':174000000,'all_opponents_fixed_across_strength_candidates':True}
        (a.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'selection':selected,'final':{m:row.get('paired_gain') for m,row in report.items() if isinstance(row,dict)}}),flush=True)


if __name__=='__main__':main()
