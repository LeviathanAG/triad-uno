"""Independent167M gate for the fixed161M source-diversity archive choice.

This is a new frozen checkpoint chosen for strong-source crossplay scope,
not another replication of the earlier failed ranking/original MLP finalists.
Candidate always uses the exact portable stdlib scorer. Rivals remain fixed
and use the previously disclosed training implementations. All arrays retained.
"""
import argparse,concurrent.futures,hashlib,json,multiprocessing,random,time
from pathlib import Path
import numpy as np
from train_uno_league import engine,make_policy,paired_interval,summarize
from uno_policy import INITIAL_WEIGHTS

REFERENCE=None;CANDIDATES=None;RIVALS=None


def initialize(reference,checkpoint):
    global REFERENCE,CANDIDATES,RIVALS
    REFERENCE=reference
    CANDIDATES=[make_policy(reference['anchor'],0),make_policy({'kind':'mlp_python','model':checkpoint},0)]
    RIVALS={s['name']:make_policy(s,0) for s in reference['trained']}


def evaluate(args):
    index,seeds,mode=args;points=[];wins=[];started=time.perf_counter()
    for seed in seeds:
        rng=random.Random(seed^0x4855424552313637)
        if mode=='trained':rivals=rng.sample(REFERENCE['trained'],3)
        elif mode=='route':rivals=[REFERENCE['anchor']]*3
        elif mode=='classical':rivals=rng.sample(REFERENCE['classical'],3)
        else:rivals=[{'kind':mode}]*3
        outcomes=[]
        for seat in range(4):
            actors=[None]*4;actors[seat]=CANDIDATES[index]
            for offset,spec in enumerate(rivals,1):
                actors[(seat+offset)%4]=RIVALS[spec['name']] if spec.get('name') in RIVALS else make_policy(spec,seed*10+offset)
            state=engine.new_game(engine.GameConfig(max_turns=1000),seed)
            while state.phase!=engine.FINISHED and seat not in state.finished_set:
                current=state.current_seat;o=engine.observe(state,current)
                action=engine.engine_action_from_id(state,current,actors[current].select(o))
                if action is None or not engine.step(state,action).ok:raise RuntimeError('illegal frozen Huber decision')
            if seat in state.finished_order:
                pos=state.finished_order.index(seat);outcomes.append((state.config.position_points[pos],float(pos==0)))
            else:
                remaining=state.config.position_points[len(state.outcome.placements):]
                outcomes.append((sum(remaining)/len(state.outcome.unfinished),0.))
        points.append(sum(x[0] for x in outcomes)/4);wins.append(sum(x[1] for x in outcomes)/4)
    return {'points':float(np.mean(points)),'win_rate':float(np.mean(wins)),'games':len(seeds)*4,
            'block_points':points,'block_wins':wins,'seconds':time.perf_counter()-started}


def merge(rows):
    points=[x for row in rows for x in row['block_points']];wins=[x for row in rows for x in row['block_wins']]
    return {'points':float(np.mean(points)),'win_rate':float(np.mean(wins)),
            'games':len(points)*4,'block_points':points,'block_wins':wins,
            'summed_worker_seconds':sum(row['seconds'] for row in rows)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--incumbent',type=Path,required=True);p.add_argument('--trained-pool',type=Path,required=True)
    p.add_argument('--choice-provenance',type=Path,required=True);p.add_argument('--blocks',type=int,default=256)
    p.add_argument('--seed-start',type=int,default=167000000);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    checkpoint=json.loads(a.model.read_text());choice=json.loads(a.choice_provenance.read_text())
    if choice['checkpoint_sha256']!=hashlib.sha256(a.model.read_bytes()).hexdigest():raise ValueError('frozen checkpoint identity changed')
    anchor={'name':'retained_route','kind':'route','weights':json.loads(a.incumbent.read_text())['weights']}
    trained=json.loads(a.trained_pool.read_text());attack=list(INITIAL_WEIGHTS)
    for i in (2,4,6,10):attack[i]+=2
    preserve=list(INITIAL_WEIGHTS);preserve[5]-=2;preserve[6]-=2
    v1=next(s['weights'] for s in trained if s['kind']=='v1')
    reference={'anchor':anchor,'trained':trained,'classical':[{'kind':'starter'},{'kind':'random'},
        {'kind':'v1','weights':INITIAL_WEIGHTS},{'kind':'v1','weights':attack},
        {'kind':'v1','weights':preserve},{'kind':'v1','weights':v1}]}
    modes=('trained','route','classical','starter','random')
    chunks=[list(range(s,min(s+16,a.seed_start+a.blocks))) for s in range(a.seed_start,a.seed_start+a.blocks,16)]
    keys=[];tasks=[]
    for mode in modes:
        for index in (0,1):
            for seeds in chunks:keys.append((mode,index));tasks.append((index,seeds,mode))
    collected={key:[] for key in keys}
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(reference,checkpoint)) as executor:
        for i,(key,row) in enumerate(zip(keys,executor.map(evaluate,tasks))):
            collected[key].append(row);print(json.dumps({'completed':i+1,'total':len(tasks),'mode':key[0],'candidate':bool(key[1]),'points':row['points']}),flush=True)
    report={}
    for mode in modes:
        base=merge(collected[mode,0]);candidate=merge(collected[mode,1])
        report[mode]={'incumbent':summarize(base),'candidate':summarize(candidate),
            'paired_gain':paired_interval(candidate,base),'block_points_candidate':candidate['block_points'],
            'block_points_incumbent':base['block_points'],'block_wins_candidate':candidate['block_wins'],
            'block_wins_incumbent':base['block_wins']}
    ci=report['trained']['paired_gain']['ci95'] if a.blocks>1 else None
    eligible=ci is not None and ci[0]>0 and all(report[m]['paired_gain']['mean_gain']>=limit for m,limit in [('route',-.02),('classical',-.03),('starter',-.05),('random',-.05)])
    report['eligibility']={'eligible':eligible,'criterion':choice['gate'],'empirical_guard_not_event_guarantee':True}
    report['provenance']={'fixed_choice':choice,'candidate_exact_stdlib':True,'seed_start':a.seed_start,
        'blocks':a.blocks,'workers':a.workers,'actual_hidden_or_game_seed_inputs':False,
        'incumbent_sha256':hashlib.sha256(a.incumbent.read_bytes()).hexdigest(),
        'checkpoint_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
        'scoring':'5/3/2/1; actual unfinished seats share remaining points',
        'trained_rivals_use_numpy_mlp_disclosed':True}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({m:report[m]['paired_gain'] for m in modes}),flush=True)


if __name__=='__main__':main()
