"""Bounded joint30 search with fixed rivals and classical selection guards."""
import argparse,concurrent.futures,hashlib,json,multiprocessing,random,time
from pathlib import Path
import numpy as np
from train_uno_league import engine,make_policy,paired_interval,summarize
from uno_risk_policy import Policy,FEATURE_NAMES
from uno_policy import INITIAL_WEIGHTS

REFERENCE=None
STATIC_AGENTS=None


def initialize(reference):
    global REFERENCE,STATIC_AGENTS
    REFERENCE=reference
    # Our route/v1/MLP policies are stateless scorers; random styles are rebuilt
    # from the per-game rival seed. Fixed numeric matrices are reused safely.
    STATIC_AGENTS={s['name']:make_policy(s,0) for s in reference['trained']}


def rival_agent(spec,seed):
    if spec.get('name') in STATIC_AGENTS:return STATIC_AGENTS[spec['name']]
    return make_policy(spec,seed)


def evaluate(args):
    genome,seeds,mode=args;base=genome[:24];residual=genome[24:];points=[];wins=[];declines=0
    started=time.perf_counter()
    for seed in seeds:
        rng=random.Random(seed^0x4A4F494E543330)
        if mode=='trained':rivals=rng.sample(REFERENCE['trained'],3)
        elif mode=='route':rivals=[REFERENCE['anchor']]*3
        elif mode=='classical':rivals=rng.sample(REFERENCE['classical'],3)
        else:rivals=[{'kind':mode}]*3
        outcomes=[]
        for seat in range(4):
            candidate=Policy(base,residual);actors=[None]*4;actors[seat]=candidate
            for offset,spec in enumerate(rivals,1):actors[(seat+offset)%4]=rival_agent(spec,seed*10+offset)
            state=engine.new_game(engine.GameConfig(max_turns=1000),seed)
            while state.phase!=engine.FINISHED and seat not in state.finished_set:
                current=state.current_seat;o=engine.observe(state,current)
                selected=actors[current].select(o);action=engine.engine_action_from_id(state,current,selected)
                if action is None or not engine.step(state,action).ok:raise RuntimeError('illegal joint-risk decision')
            if seat in state.finished_order:
                pos=state.finished_order.index(seat);outcomes.append((state.config.position_points[pos],float(pos==0)))
            else:
                remaining=state.config.position_points[len(state.outcome.placements):]
                outcomes.append((sum(remaining)/len(state.outcome.unfinished),0.))
            declines+=candidate.declines
        points.append(sum(x[0] for x in outcomes)/4);wins.append(sum(x[1] for x in outcomes)/4)
    return {'points':float(np.mean(points)),'win_rate':float(np.mean(wins)),'games':len(seeds)*4,
            'block_points':points,'block_wins':wins,'public_prior_declines':declines,
            'seconds':time.perf_counter()-started}


def main():
    p=argparse.ArgumentParser();p.add_argument('--incumbent',type=Path,required=True)
    p.add_argument('--trained-pool',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--iterations',type=int,default=24);p.add_argument('--population',type=int,default=16)
    p.add_argument('--train-blocks',type=int,default=64);p.add_argument('--selection-blocks',type=int,default=256)
    p.add_argument('--test-blocks',type=int,default=512);p.add_argument('--seed',type=int,default=20261030)
    p.add_argument('--train-seed-start',type=int,default=151_000_000)
    p.add_argument('--selection-seed-start',type=int,default=152_000_000)
    p.add_argument('--test-seed-start',type=int,default=153_000_000);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if min(a.iterations,a.train_blocks,a.selection_blocks,a.test_blocks,a.workers)<1 or a.population<4:p.error('positive counts/population>=4 required')
    if a.train_seed_start+a.iterations*a.train_blocks>a.selection_seed_start or a.selection_seed_start+a.selection_blocks>a.test_seed_start:p.error('overlapping seed families')
    a.output.mkdir(parents=True,exist_ok=True);prior=json.loads(a.incumbent.read_text());base=prior['weights']
    if len(base)!=24:raise ValueError('retained24 prior required')
    trained=json.loads(a.trained_pool.read_text());anchor={'name':'retained_route','kind':'route','weights':base}
    if next(s['weights'] for s in trained if s['name']=='retained_route')!=base:raise ValueError('rival anchor differs from fixed retained prior')
    attack=list(INITIAL_WEIGHTS)
    for i in (2,4,6,10):attack[i]+=2.
    preserve=list(INITIAL_WEIGHTS);preserve[5]-=2.;preserve[6]-=2.
    v1=next(s['weights'] for s in trained if s['kind']=='v1')
    reference={'anchor':anchor,'trained':trained,'classical':[{'kind':'starter'},{'kind':'random'},
        {'kind':'v1','weights':INITIAL_WEIGHTS},{'kind':'v1','weights':attack},
        {'kind':'v1','weights':preserve},{'kind':'v1','weights':v1}]}
    original=np.asarray(base+[0.]*6);scale=np.asarray([max(1.,abs(x)) for x in base]+[6.]*6)
    lower=np.asarray(base)-.25*scale[:24];upper=np.asarray(base)+.25*scale[:24]
    lower=np.r_[lower,[-8.]*6];upper=np.r_[upper,[8.]*6]
    sigma=np.r_[.10*scale[:24],[1.5]*6];floor=np.r_[.02*scale[:24],[.25]*6]
    mean=original.copy();rng=np.random.default_rng(a.seed);finalists=[original.tolist()];history=[];started=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize,initargs=(reference,)) as executor:
        for generation in range(a.iterations):
            population=np.clip(rng.normal(mean,sigma,(a.population,30)),lower,upper)
            population[0]=mean;population[1]=original
            seeds=list(range(a.train_seed_start+generation*a.train_blocks,a.train_seed_start+(generation+1)*a.train_blocks))
            results=list(executor.map(evaluate,[(w.tolist(),seeds,mode) for mode in ('trained','route') for w in population]))
            fitness=[.7*results[i]['points']+.3*results[a.population+i]['points'] for i in range(a.population)]
            order=sorted(range(a.population),key=lambda i:fitness[i],reverse=True);elite=population[order[:max(2,a.population//4)]]
            mean=np.clip(.3*mean+.7*elite.mean(axis=0),lower,upper)
            sigma=np.minimum((upper-lower)/4,np.maximum(floor,.3*sigma+.7*elite.std(axis=0)))
            best=population[order[0]].tolist()
            if generation%4==3 and not any(np.linalg.norm((np.asarray(x)-best)/scale)<.15 for x in finalists):finalists.append(best)
            row={'generation':generation+1,'genome':best,'composite':fitness[order[0]],'incumbent_composite':fitness[1],
                 'trained':summarize(results[order[0]]),'route':summarize(results[a.population+order[0]]),
                 'seconds':time.perf_counter()-started}
            history.append(row);print(json.dumps(row),flush=True)
            (a.output/'training-progress.json').write_text(json.dumps(history,indent=2)+'\n')
        if not any(np.linalg.norm((np.asarray(x)-mean)/scale)<.15 for x in finalists):finalists.append(mean.tolist())
        seeds=list(range(a.selection_seed_start,a.selection_seed_start+a.selection_blocks));modes=('trained','route','classical')
        rows=list(executor.map(evaluate,[(w,seeds,m) for m in modes for w in finalists]));n=len(finalists)
        grouped={m:rows[i*n:(i+1)*n] for i,m in enumerate(modes)}
        eligible=[0];gains=[]
        for i in range(1,n):
            delta={m:grouped[m][i]['points']-grouped[m][0]['points'] for m in modes}
            gains.append(delta)
            if delta['trained']>=0 and delta['route']>=-.02 and delta['classical']>=-.03:eligible.append(i)
        chosen=max(eligible,key=lambda i:.7*grouped['trained'][i]['points']+.3*grouped['route'][i]['points'])
        selection={'selected':chosen,'eligible':eligible,'genomes':finalists,
            'scores':{m:[summarize(r) for r in grouped[m]] for m in modes},
            'paired_gains':{m:[paired_interval(r,grouped[m][0]) for r in grouped[m][1:]] for m in modes},
            'guards':{'trained_mean_minimum_gain':0.,'route_mean_minimum_gain':-.02,'classical_mean_minimum_gain':-.03},
            'criterion':'70% own-trained league +30% fixed retained-route selfplay; incumbent eligible',
            'seed_start':a.selection_seed_start,'blocks':a.selection_blocks}
        (a.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        model={'format':'triad-uno-route-unseen-risk-v1','features':FEATURE_NAMES,
               'base_weights':finalists[chosen][:24],'residual_weights':finalists[chosen][24:],
               'training':{'method':'bounded joint30 CEM','frozen_base':False,'base_trust_fraction':.25,
                'lower_bounds':lower.tolist(),'upper_bounds':upper.tolist(),'incumbent_base_weights':base,
                'train_seed_start':a.train_seed_start,'selection_seed_start':a.selection_seed_start,'test_seed_start':a.test_seed_start,
                'independent_seed_reserved':154_000_000,'iterations':a.iterations,'population':a.population,'train_blocks':a.train_blocks,
                'objective_weights':{'trained':.7,'route':.3},'selection_guards':selection['guards'],
                'incumbent_sha256':hashlib.sha256(a.incumbent.read_bytes()).hexdigest(),
                'trained_pool_sha256':hashlib.sha256(a.trained_pool.read_bytes()).hexdigest(),
                'fallback':'on incomplete public information, trained scalar base scorer is used',
                'belief_approximation':'uniform per-seat hard-constraint response features;not joint/style posterior'}}
        (a.output/'selected-policy.json').write_text(json.dumps(model,indent=2)+'\n')
        seeds=list(range(a.test_seed_start,a.test_seed_start+a.test_blocks));modes=('trained','route','classical','starter','random')
        results=list(executor.map(evaluate,[(w,seeds,m) for m in modes for w in (finalists[chosen],original.tolist())]))
        report={m:{'candidate':summarize(results[2*i]),'incumbent':summarize(results[2*i+1]),
                  'paired_gain':paired_interval(results[2*i],results[2*i+1]),
                  'block_points_candidate':results[2*i]['block_points'],'block_points_incumbent':results[2*i+1]['block_points']} for i,m in enumerate(modes)}
        report['provenance']={'fixed_selected':chosen,'seed_start':a.test_seed_start,'blocks':a.test_blocks,
            'actual_hidden_state_or_game_seed_policy_inputs':False,'all_rivals_fixed_across_candidate_coefficients':True,
            'joint_dimensions':30,'base_trust_fraction':.25,'objective_weights':{'trained':.7,'route':.3},
            'independent_seed_reserved':154_000_000}
        (a.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'selection':chosen,'final':{m:report[m]['paired_gain'] for m in modes}}),flush=True)


if __name__=='__main__':main()
