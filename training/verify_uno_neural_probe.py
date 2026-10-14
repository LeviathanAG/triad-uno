"""Deployment-scorer contract/timing probe; never selects a model by game reward."""
import json
from pathlib import Path
import time
from train_uno_league import engine
from uno_mlp_policy import Policy,FastPolicy

model=json.loads(Path('training/kaggle-uno-route-neural-output/output/uno-policy-mlp.json').read_text())
policy,fast=Policy(model),FastPolicy(model)
count=0;max_seconds=0.;mismatches=0;substantive=0;examples=[]
for seed in range(70_100_000,70_100_016):
    state=engine.new_game(engine.GameConfig(max_turns=1000),seed)
    while state.phase!=engine.FINISHED:
        seat=state.current_seat;obs=engine.observe(state,seat)
        start=time.perf_counter();choice=policy.select(obs);max_seconds=max(max_seconds,time.perf_counter()-start)
        
        if choice!=fast.select(obs):
            from uno_policy_v2 import public_beliefs
            beliefs=public_beliefs(obs)
            a=next(a for a in obs['legal_actions'] if a['action_id']==choice)
            b=next(a for a in obs['legal_actions'] if a['action_id']==fast.select(obs))
            mismatches+=1
            semantic_a={k:v for k,v in a.items() if k not in ('action_id','card_ref')}
            semantic_b={k:v for k,v in b.items() if k not in ('action_id','card_ref')}
            if semantic_a!=semantic_b or policy.action_features(obs,a,beliefs)!=policy.action_features(obs,b,beliefs):
                substantive+=1
                examples.append({'seed':seed,'turn':obs.get('turn_number'),'first':a,'second':b})
        assert choice==Policy(model).select(obs),'restart changed policy action'
        action=engine.engine_action_from_id(state,seat,choice)
        assert action is not None and engine.step(state,action).ok
        count+=1
state=engine.new_game(engine.GameConfig(max_turns=1000),123)
# Valid108-card accounting stress fixture, not game evidence: one card in each
# opponent hand, one current discard, all104 remaining cards in the own hand.
seat=state.current_seat
cards=list(state.draw_pile)
for other in range(4):
    if other==seat:continue
    cards+=state.hands[other][1:];state.hands[other]=state.hands[other][:1]
state.hands[seat]+=cards;state.draw_pile=[]
obs=engine.observe(state,seat)
start=time.perf_counter();choice=policy.select(obs);stress_seconds=time.perf_counter()-start
assert choice==fast.select(obs) and engine.engine_action_from_id(state,seat,choice) is not None
report={'development_resource_probe':True,'candidate_format':model['format'],'fresh_restart_equivalence_decisions':count,
        'pure_python_numpy_identical_action_refs':count-mismatches,'different_duplicate_card_refs':mismatches,'substantive_mismatches':substantive,'examples':examples,'host_max_normal_decision_seconds':max_seconds,
        'stress_hand_size':len(obs['hand']),'stress_legal_count':len(obs['legal_actions']),
        'host_stress_decision_seconds':stress_seconds,'note':'Host timing only; restricted1CPU admission remains necessary before deployment. No strength selection uses this fixture.'}
Path('training/uno-neural-probe-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
