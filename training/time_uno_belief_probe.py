"""Development-only referee games to measure bounded probe CPU/RAM behaviour."""
import argparse
import json
from pathlib import Path
import resource
import time
from uno_belief_rollout_probe import Policy,engine


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--games',type=int,default=8);p.add_argument('--seed',type=int,default=55_100_000)
    p.add_argument('--budget',type=float,default=.15)
    p.add_argument('--max-trials',type=int,default=8)
    p.add_argument('--max-actions',type=int,default=4)
    p.add_argument('--rival-model',choices=['route','mixture'],default='route')
    p.add_argument('--fast-simulation',action='store_true')
    p.add_argument('--cached-public-fold',action='store_true');a=p.parse_args()
    probe=Policy(json.loads(a.model.read_text())['weights'],budget=a.budget,max_trials=a.max_trials,min_trials=4,
                 fast_simulation=a.fast_simulation,cached_public_fold=a.cached_public_fold,
                 max_actions=a.max_actions,rival_model=a.rival_model)
    started=time.perf_counter();decisions=0
    for seed in range(a.seed,a.seed+a.games):
        state=engine.new_game(engine.GameConfig(),seed)
        while state.phase!=engine.FINISHED:
            obs=engine.observe(state,state.current_seat)
            selected=probe.select(obs)
            if selected not in {x['action_id'] for x in obs['legal_actions']}:raise RuntimeError('probe selected illegal action')
            if not engine.step(state,engine.engine_action_from_id(state,state.current_seat,selected)).ok:raise RuntimeError('probe step failed')
            decisions+=1
    if probe.stats['max_decision_seconds']>=5:raise RuntimeError('decision exceeded official5-second ceiling')
    print(json.dumps({'development_only':True,'games':a.games,'decisions':decisions,'seconds':time.perf_counter()-started,
                      'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'stats':probe.stats,
                      'config':{'max_trials':a.max_trials,'fast_simulation':a.fast_simulation,
                                'cached_public_fold':a.cached_public_fold,'budget_seconds':a.budget,
                                'max_actions':a.max_actions,'rival_model':a.rival_model}}))


if __name__=='__main__':main()
