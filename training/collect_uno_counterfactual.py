"""Offline action-value targets from paired counterfactual SDK rollouts.

Hidden simulator state is used only for training rewards, as permitted by the
SDK training guide. Exported policy inputs contain own/public information only.
No deployed policy sees seeds, hidden hands, or the simulator.
"""
import argparse
import concurrent.futures
import copy
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
from collections import Counter
from uno_policy import Policy as V1, INITIAL_WEIGHTS
from uno_policy_v2 import action_features, public_beliefs, FEATURE_NAMES

ROOT=Path(__file__).resolve().parent
SDK=Path(os.environ.get('UNO_SDK',ROOT.parent/'unoSDK'))
sys.path.insert(0,str(SDK/'local_evaluator/src'))
from triad_uno import engine
from triad_uno.baselines import RandomAgent,HeuristicAgent


def actor(kind,seed,weights):
    if kind=='random':return RandomAgent(seed)
    if kind=='heuristic':return HeuristicAgent(seed)
    if kind=='initial':return V1(INITIAL_WEIGHTS)
    if len(weights)==24:
        from uno_route_policy import Policy as RoutePolicy
        return RoutePolicy(weights)
    return V1(weights)


def reward(state,seat):
    if seat in state.finished_order:
        return state.config.position_points[state.finished_order.index(seat)]
    remaining=state.config.position_points[len(state.outcome.placements):]
    return sum(remaining)/len(state.outcome.unfinished)


def advance(state,actors):
    seat=state.current_seat
    obs=engine.observe(state,seat)
    selected=actors[seat].select(obs)
    action=engine.engine_action_from_id(state,seat,selected)
    if action is None or not engine.step(state,action).ok:
        raise RuntimeError('rollout actor produced an illegal action')


def rollout(original,seat,selected,kinds,weights,seed,future_shuffle=False):
    state=copy.deepcopy(original)
    # Same trial seed across candidate actions yields common future random draws
    # and random-opponent streams. Present cards are never altered/reallocated.
    state.rng=random.Random(seed)
    if future_shuffle:
        # Future card order is unobserved. Each action shares the same trial
        # permutation, while independent trials integrate its uncertainty.
        # Existing hands are held fixed for these training-only rewards.
        state.rng.shuffle(state.draw_pile)
    actors=[actor(kind,seed*10+s,weights) for s,kind in enumerate(kinds)]
    action=engine.engine_action_from_id(state,seat,selected)
    if action is None or not engine.step(state,action).ok:
        raise RuntimeError('counterfactual action is not legal at sampled state')
    while state.phase!=engine.FINISHED and seat not in state.finished_set:
        advance(state,actors)
    return reward(state,seat)


def collect_job(job):
    seed,weights,samples,trials,max_actions,feature_version,*options=job
    future_shuffle=options[0] if options else False
    reservoir_sample=options[1] if len(options)>1 else False
    feature_function=action_features
    if feature_version==4:
        from uno_policy_features_v4 import action_features as rich_features
        feature_function=rich_features
    elif feature_version==3:
        from uno_policy_features_v3 import action_features as rich_features
        feature_function=rich_features
    rng=random.Random(seed^0x434F554E544552)
    seat=seed%4
    kinds=[rng.choice(('incumbent','initial','heuristic','random')) for _ in range(4)]
    kinds[seat]='incumbent'
    actors=[actor(kind,seed*10+s,weights) for s,kind in enumerate(kinds)]
    state=engine.new_game(engine.GameConfig(max_turns=1000),seed)
    rows=[];snapshots=[];eligible_count=0
    def label(original,index):
        obs=engine.observe(original,seat)
        legal=list(obs['legal_actions'])
        if len(legal)>max_actions:
            preferred=actor('incumbent',seed,weights).select(obs)
            kept=next(a for a in legal if a['action_id']==preferred)
            legal=[kept]+rng.sample([a for a in legal if a is not kept],max_actions-1)
        beliefs=public_beliefs(obs)
        features=[feature_function(obs,a,beliefs) for a in legal]
        trial_values=[[rollout(original,seat,a['action_id'],kinds,weights,seed*10_000_000+index*1000+trial,future_shuffle) for trial in range(trials)] for a in legal]
        values=[sum(v)/trials for v in trial_values]
        return (features,values,seed,trial_values,len(obs['hand']),obs['phase'])
    while state.phase!=engine.FINISHED and seat not in state.finished_set:
        current=state.current_seat
        if current==seat and (reservoir_sample or len(rows)<samples):
            obs=engine.observe(state,seat)
            legal=list(obs['legal_actions'])
            if len(legal)>1:
                if reservoir_sample:
                    eligible_count+=1
                    index=rng.randrange(eligible_count)
                    if index<samples:
                        saved=(copy.deepcopy(state),eligible_count)
                        if len(snapshots)<samples:snapshots.append(saved)
                        else:snapshots[index]=saved
                elif rng.random()<.5:
                    rows.append(label(state,len(rows)))
        advance(state,actors)
    if reservoir_sample:
        rows=[label(original,index) for original,index in snapshots]
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--incumbent',type=Path,default=SDK/'models/policy.json')
    parser.add_argument('--games',type=int,default=1200)
    parser.add_argument('--samples',type=int,default=4)
    parser.add_argument('--trials',type=int,default=2)
    parser.add_argument('--max-actions',type=int,default=12)
    parser.add_argument('--feature-version',type=int,choices=(2,3,4),default=2)
    parser.add_argument('--future-shuffle',action='store_true')
    parser.add_argument('--reservoir-sample',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed-start',type=int,default=10_000_000)
    parser.add_argument('--output',type=Path,default=ROOT/'uno-counterfactual-data')
    args=parser.parse_args()
    if min(args.games,args.samples,args.trials,args.workers)<1 or args.max_actions<2:parser.error('invalid counts')
    args.output.mkdir(parents=True,exist_ok=True)
    document=json.loads(args.incumbent.read_text());weights=document['weights']
    features,targets,offsets,game_ids,trial_targets=[],[],[0],[],[]
    hand_histogram,phase_histogram=Counter(),Counter()
    started=time.time()
    jobs=[(seed,weights,args.samples,args.trials,args.max_actions,args.feature_version,args.future_shuffle,args.reservoir_sample) for seed in range(args.seed_start,args.seed_start+args.games)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for index,rows in enumerate(pool.map(collect_job,jobs)):
            for vectors,values,seed,trial_values,hand_size,phase in rows:
                features.extend(vectors);targets.extend(values);offsets.append(len(features));game_ids.append(seed)
                trial_targets.extend(trial_values)
                hand_histogram[hand_size]+=1;phase_histogram[phase]+=1
            if (index+1)%100==0:print(json.dumps({'games':index+1,'states':len(game_ids),'actions':len(features),'seconds':round(time.time()-started,1)}),flush=True)
    np.savez_compressed(args.output/'counterfactual.npz',features=np.asarray(features,dtype=np.float32),targets=np.asarray(targets,dtype=np.float32),offsets=np.asarray(offsets,dtype=np.int64),game_ids=np.asarray(game_ids,dtype=np.int64),trial_targets=np.asarray(trial_targets,dtype=np.float32))
    names=FEATURE_NAMES
    if args.feature_version==4:
        from uno_policy_features_v4 import FEATURE_NAMES as rich_names
        names=rich_names
    elif args.feature_version==3:
        from uno_policy_features_v3 import FEATURE_NAMES as rich_names
        names=rich_names
    metadata={'method':'paired one-step policy-improvement rollouts','input_features':names,'feature_version':args.feature_version,'games':args.games,'states':len(game_ids),'actions':len(features),'trials':args.trials,'future_shuffle':args.future_shuffle,'reward_conditioning':'present hidden simulator hands held fixed; future draw-pile permutations averaged' if args.future_shuffle else 'original simulator card distribution','max_actions':args.max_actions,'seed_start':args.seed_start,'sdk_commit':'0cda3acc1bb5c05fa1803a8fbd7761770dc6d5fe','incumbent':document,'seconds':round(time.time()-started,1)}
    metadata.update(reservoir_sample=args.reservoir_sample,own_hand_size_histogram=dict(sorted(hand_histogram.items())),phase_histogram=dict(phase_histogram))
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('incumbent','input_features')}),flush=True)


if __name__=='__main__':main()
