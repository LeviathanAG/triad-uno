"""Development-only checks against authoritative UNO legality on small hands."""
import json
from pathlib import Path
import random
import sys
import time
import os
from uno_route_policy import route_length, Policy, action_features
from uno_policy import Policy as Base
ROOT=Path(__file__).resolve().parent
SDK=Path(os.environ.get('UNO_SDK',ROOT.parent if (ROOT.parent/'local_evaluator').is_dir() else ROOT.parent/'unoSDK'))
sys.path.insert(0,str(SDK/'local_evaluator/src'))
from triad_uno import engine
from triad_uno.cards import build_canonical_deck, COLORS


def exact(hand, colour, top):
    best=0
    for index, card in enumerate(hand):
        if not engine.is_card_playable(card,top,colour,hand):continue
        remaining=hand[:index]+hand[index+1:]
        colours=COLORS if card.kind in ('wild','wild_draw_four') else (card.color,)
        best=max(best,1+max(exact(remaining,c,card) for c in colours))
    return best


def main():
    rng=random.Random(2026101408);deck=build_canonical_deck()
    for _ in range(300):
        hand=rng.sample(deck,rng.randrange(1,6));top=rng.choice(deck);colour=rng.choice(COLORS)
        actual=route_length([c.to_public_dict() for c in hand],colour,top.to_public_dict(),max_states=10_000)
        expected=exact(hand,colour,top)
        assert actual==expected,(hand,top,colour,actual,expected)
    source=SDK/'models/incumbent-v1.json'
    if not source.is_file():source=SDK/'models/policy.json'
    base=json.loads(source.read_text())['weights']
    zero=Policy(base+[0.]*6);learned=Base(base);decisions=0;elapsed=[]
    probe=Policy(base+[2.,-2.,1.,1.,.5,-1.])
    for seed in range(24):
        state=engine.new_game(engine.GameConfig(),30_000_000+seed)
        while state.phase!=engine.FINISHED:
            obs=engine.observe(state,state.current_seat)
            assert zero.select(obs)==learned.select(obs)
            started=time.perf_counter();chosen=probe.select(obs);elapsed.append(time.perf_counter()-started)
            assert chosen in {a['action_id'] for a in obs['legal_actions']}
            for a in obs['legal_actions']:
                assert len(action_features(obs,a))==24
            selected=engine.engine_action_from_id(state,state.current_seat,chosen)
            assert engine.step(state,selected).ok
            decisions+=1
    print(json.dumps({'exhaustive_small_hand_checks':300,'zero_residual_compatible_decisions':decisions,'legal_probe_decisions':decisions,'max_decision_seconds':max(elapsed)}))


if __name__=='__main__':main()
