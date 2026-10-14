"""Restricted-container timing; prototype coefficients are not strength evidence."""
import argparse,json,os,resource,sys,time
from pathlib import Path
from uno_huber_stage_policy import Policy

SDK=Path(os.environ.get('UNO_SDK',Path(__file__).resolve().parent.parent/'unoSDK'))
sys.path.insert(0,str(SDK/'local_evaluator/src'))
from triad_uno import engine
from triad_uno.cards import build_canonical_deck


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--incumbent',type=Path,required=True);p.add_argument('--candidate',type=Path)
    p.add_argument('--games',type=int,default=8);p.add_argument('--seed',type=int,default=177000000)
    a=p.parse_args()
    if a.candidate:
        model=json.loads(a.candidate.read_text());checkpoint=model['checkpoint'];weights=model['route_weights'];strengths=model['strengths']
    else:
        checkpoint=json.loads(a.checkpoint.read_text());weights=json.loads(a.incumbent.read_text())['weights'];strengths=[1.5]*6
    policy=Policy(weights,checkpoint,strengths);maximum=0.;decisions=0;started=time.perf_counter()
    for seed in range(a.seed,a.seed+a.games):
        state=engine.new_game(engine.GameConfig(max_turns=1000),seed)
        while state.phase!=engine.FINISHED:
            o=engine.observe(state,state.current_seat);begin=time.perf_counter();selected=policy.select(o)
            maximum=max(maximum,time.perf_counter()-begin);decisions+=1
            action=engine.engine_action_from_id(state,state.current_seat,selected)
            if action is None or not engine.step(state,action).ok:raise RuntimeError('illegal stage resource-probe decision')
    state=engine.new_game(engine.GameConfig(max_turns=1000),0);deck=build_canonical_deck()
    state.hands=[deck[:104],deck[104:105],deck[105:106],deck[106:107]]
    state.draw_pile=[];state.discard_pile=[deck[107]];state.active_color='red'
    state.current_seat=0;state.direction=1;state.phase=engine.TURN
    state.pending_penalty=0;state.pending_source=None;state.drawn_card_uid=None
    state.finished_order=[];state.finished_set=set();state.turn_count=0;state.outcome=None
    state.history=[{'type':'initial_discard','turn_number':0,'seat':None,'card':deck[107].to_public_dict()}]
    o=engine.observe(state,0);begin=time.perf_counter();selected=policy.select(o);stress=time.perf_counter()-begin
    assert selected in {action['action_id'] for action in o['legal_actions']}
    maximum=max(maximum,stress);rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if maximum>=5 or rss>=1073741824:raise RuntimeError('resource ceiling exceeded')
    print(json.dumps({'development_only':True,'actual_trained_stage_candidate':bool(a.candidate),
        'games':a.games,'decisions':decisions,'seconds':time.perf_counter()-started,
        'max_decision_seconds':maximum,'max_rss_bytes':rss,'stress_own_cards':104,
        'stress_legal_actions':len(o['legal_actions']),'stress_seconds':stress,'strengths':strengths,
        'trained_neural_archive_checkpoint':True,'prototype_timing_not_strength_evidence':not bool(a.candidate)}),flush=True)


if __name__=='__main__':main()
