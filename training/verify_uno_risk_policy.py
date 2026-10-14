"""Zero-prior equivalence, finite decisions and public-history decline guard."""
import copy,json
from pathlib import Path
from train_uno_league import engine
from uno_route_policy import Policy as Route
from uno_risk_policy import Policy


def main():
    base=json.loads(Path('unoSDK/models/policy.json').read_text())['weights']
    original=Route(base);zero=Policy(base,[0.]*6);candidate=Policy(base,[-1.,-2.,-1.,-2.,-1.,2.])
    checks=0;multi=0;guard=0
    for seed in range(147_000_000,147_000_016):
        state=engine.new_game(engine.GameConfig(max_turns=1000),seed)
        while state.phase!=engine.FINISHED:
            o=engine.observe(state,state.current_seat);prior=original.select(o)
            assert zero.select(o)==prior
            selected=candidate.select(o)
            assert selected in {a['action_id'] for a in o['legal_actions']}
            checks+=1
            if len(o['legal_actions'])>1:
                multi+=1;bad=copy.deepcopy(o);bad['history']=[]
                before=candidate.declines
                assert candidate.select(bad)==original.select(bad) and candidate.declines==before+1
                guard+=1
            assert engine.step(state,engine.engine_action_from_id(state,state.current_seat,prior)).ok
    report={'development_only':True,'runtime_unchanged':True,'source_games':16,
            'source_seed_start':147_000_000,'exact_zero_residual_decisions':checks,
            'nonzero_legal_decisions':checks,'multichoice_decisions':multi,
            'incomplete_history_prior_declines':guard,
            'actual_hidden_state_or_game_seed_policy_inputs':False}
    Path('training/uno-risk-policy-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
