"""Exact toy proposal support proof plus independent public-observation audit."""
import itertools,json,math,random,time
from collections import Counter
from pathlib import Path
from uno_public_belief import PreparedSampler,compatible
from uno_public_belief_conditioned import PreparedConditionedSampler
from uno_belief_rollout_probe import sampled_world,bind
from train_uno_league import engine,make_policy


def toy_proof():
    checks=[];cards=[('red','number',0),('red','number',1),('yellow','number',2),
        ('green','number',3),('blue','number',4),(None,'wild',None),(None,'wild_draw_four',None)]
    empty=lambda:{'colours':set(),'symbols':set(),'wilds_absent':False}
    for exclusions in ((),('red',),('red','yellow'),('red','yellow','green')):
        constraints={s:empty() for s in range(4)}
        constraints[1]['colours']=set(exclusions)
        constraints[2]['colours']={'blue'}
        constraints[2]['wilds_absent']=True
        constraints[3]['symbols']={('number',3)}
        obj=PreparedConditionedSampler.__new__(PreparedConditionedSampler)
        obj.cards=cards;obj.seat=0;obj.sizes=[0,2,1,1];obj.constraints=constraints;obj._prepare_anchor()
        h=obj.sizes[obj.anchor];allowed=obj.allowed;n=len(cards)
        def valid(order):
            offset=0
            for seat in (1,2,3):
                values=order[offset:offset+obj.sizes[seat]];offset+=obj.sizes[seat]
                if not all(compatible(cards[i],constraints[seat]) for i in values):return False
            return True
        original={order for order in itertools.permutations(range(n)) if valid(order)}
        proposal=set();accepted=set()
        for chosen in itertools.permutations(allowed,h):
            remaining=[i for i in range(n) if i not in chosen]
            for rest in itertools.permutations(remaining):
                pieces={obj.anchor:chosen};offset=0
                for seat in (1,2,3):
                    if seat==obj.anchor:continue
                    pieces[seat]=rest[offset:offset+obj.sizes[seat]];offset+=obj.sizes[seat]
                order=tuple(i for seat in (1,2,3) for i in pieces[seat])+rest[offset:]
                proposal.add(order)
                if valid(order):accepted.add(order)
        cardinality=math.perm(len(allowed),h)*math.factorial(n-h)
        assert len(proposal)==cardinality and original==accepted
        # Every ordered anchor/rest combination has exactly the same proposal
        # mass1/cardinality and maps bijectively to one ordered whole deck.
        checks.append({'anchor':obj.anchor,'proposal_cardinality':cardinality,
            'valid_cardinality':len(accepted),'accepted_support_equals_original':True,
            'uniform_conditional_probability_denominator':len(accepted)})
    return checks


def natural_audit():
    weights=json.loads(Path('unoSDK/models/policy.json').read_text())['weights']
    rows=[];observations=0;legality=0
    for seed in range(160000000,160000032):
        state=engine.new_game(engine.GameConfig(max_turns=1000),seed)
        actors=[make_policy({'kind':'route','weights':weights},0) for _ in range(4)]
        while state.phase!=engine.FINISHED:
            seat=state.current_seat;o=engine.observe(state,seat);observations+=1
            if 2<=len(o['hand'])<=3 and len(o['legal_actions'])>=2 and o['phase'] in (engine.TURN,engine.AFTER_DRAW):
                old=PreparedSampler(o);new=PreparedConditionedSampler(o)
                row={'turn':o['turn_number'],'own_size':len(o['hand']),
                    'opponent_sizes':[size for s,size in enumerate(o['hand_sizes']) if s!=seat],
                    'anchor':new.anchor,'sample_calls_per_sampler':16}
                for name,sampler in (('original',old),('conditioned',new)):
                    accepted=0;elapsed=0
                    for trial in range(16):
                        rng=random.Random(seed*100000+observations*32+trial)
                        start=time.perf_counter();possible=sampled_world(o,rng,sampler,True)
                        elapsed+=time.perf_counter()-start
                        if possible is None:continue
                        world,refs=possible;accepted+=1
                        assert sum(map(len,world.hands))+len(world.draw_pile)+len(world.discard_pile)==108
                        for other,hand in enumerate(world.hands):
                            if other==seat:continue
                            assert all(compatible((c.color,c.kind,c.value if c.kind=='number' else None),sampler.constraints[other]) for c in hand)
                        for action in o['legal_actions']:
                            assert engine._validate(world,bind(action,refs)) is None;legality+=1
                    row[name]={'accepted':accepted,'seconds':elapsed}
                rows.append(row)
            action=engine.engine_action_from_id(state,seat,actors[seat].select(o))
            assert action is not None and engine.step(state,action).ok
    totals={name:{'accepted':sum(r[name]['accepted'] for r in rows),
                  'calls':len(rows)*16,'seconds':sum(r[name]['seconds'] for r in rows)}
            for name in ('original','conditioned')}
    return {'source_games':32,'seed_start':160000000,'observations':observations,
            'eligible_observations':len(rows),'known_legal_action_checks':legality,
            'totals':totals,'rows':rows,'strength_claim':False}


if __name__=='__main__':
    report={'exact_toy_proofs':toy_proof(),'public_observation_audit':natural_audit(),
            'proposal_law':'uniform whole-card allocation conditioned on exact public exclusions',
            'prior_limitation':'Uniform allocation does not model rival behaviour or private strategy',
            'live_jobs_unchanged':True}
    Path('training/uno-conditioned-sampler-verification160M.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report['public_observation_audit'].items() if k!='rows'}),flush=True)
