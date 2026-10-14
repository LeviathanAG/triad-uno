"""Observation-only unseen-card multiset and conservative public constraints.

This is groundwork for a development-only belief-rollout probe, not deployed
inference. It never imports or receives trusted simulator state or real seeds.
"""
from collections import Counter
from uno_policy_v2 import public_beliefs, symbol

COLOURS=('red','yellow','green','blue')


def key(card):
    return (card.get('color'),card['kind'],card.get('value') if card['kind']=='number' else None)


def canonical_counts():
    counts=Counter()
    for colour in COLOURS:
        for value in range(10):counts[(colour,'number',value)]=1 if value==0 else 2
        for kind in ('skip','reverse','draw_two'):counts[(colour,kind,None)]=2
    counts[(None,'wild',None)]=4;counts[(None,'wild_draw_four',None)]=4
    return counts


def public_unknown_pool(observation):
    """Subtract own hand and the current public discard suffix from 108 cards."""
    seen=[event['card'] for event in observation['history'] if event.get('type') in ('initial_discard','play')]
    size=observation['discard_pile_size']
    if not 1<=size<=len(seen):raise ValueError('incomplete discard history')
    discards=seen[-size:]
    if key(discards[-1])!=key(observation['top_discard']):raise ValueError('discard suffix does not match public top')
    pool=canonical_counts()
    pool.subtract(key(c) for c in observation['hand'])
    pool.subtract(key(c) for c in discards)
    if any(n<0 for n in pool.values()):raise ValueError('public card count exceeds deck')
    required=sum(observation['hand_sizes'])-len(observation['hand'])+observation['draw_pile_size']
    if sum(pool.values())!=required:raise ValueError('public pile/hand counts inconsistent')
    return +pool,discards


def proven_constraints(observation):
    beliefs=public_beliefs(observation)
    wild_absent={};pending={}
    for event in observation['history']:
        kind=event.get('type');seat=event.get('seat');turn=event.get('turn_number')
        if kind=='draw':
            # Normal draws are forced, hence the pre-draw hand has no wilds.
            got=event.get('drawn',0)
            wild_absent[seat]=(got==0)
            pending[seat]=turn
        elif kind=='penalty_draw':
            if event.get('count',0):wild_absent[seat]=False
            pending.pop(seat,None)
        elif kind=='play':
            if event.get('card',{}).get('kind') in ('wild','wild_draw_four'):
                wild_absent[seat]=False
            if event.get('from_draw') and pending.get(seat)==turn:
                wild_absent[seat]=True
                pending.pop(seat,None)
        elif kind=='keep_auto' and pending.get(seat)==turn:
            # The drawn card was unplayable, so it could not be either wild.
            wild_absent[seat]=True;pending.pop(seat,None)
        elif kind=='keep':
            pending.pop(seat,None)
    return {seat:{'colours':{c for (s,c),v in beliefs['colours'].items() if s==seat and v==1.},
                  'symbols':{c for (s,c),v in beliefs['symbols'].items() if s==seat and v==1.},
                  'wilds_absent':wild_absent.get(seat,False)}
            for seat in range(len(observation['hand_sizes']))}


def compatible(card_key,constraint):
    colour,kind,value=card_key
    card_symbol=(kind,value) if kind=='number' else (kind,None) if kind in ('skip','reverse','draw_two') else None
    return (colour not in constraint['colours'] and card_symbol not in constraint['symbols']
            and not(constraint['wilds_absent'] and kind in ('wild','wild_draw_four')))


def sample_unknown_hands(observation,rng,max_attempts=32):
    """Rejection sampler; returns None on its fixed budget, never hidden fallback.

    Draw order and hypothetical hands are sampled possibilities, not factual
    opponent cards. Only exact public absence constraints are enforced.
    """
    pool,_=public_unknown_pool(observation);constraints=proven_constraints(observation)
    cards=list(pool.elements());seat=observation['seat']
    for _ in range(max_attempts):
        rng.shuffle(cards);offset=0;hands={};ok=True
        for other,count in enumerate(observation['hand_sizes']):
            if other==seat:continue
            hand=cards[offset:offset+count];offset+=count
            if not all(compatible(c,constraints[other]) for c in hand):ok=False;break
            hands[other]=hand
        if ok:return hands,cards[offset:]
    return None
