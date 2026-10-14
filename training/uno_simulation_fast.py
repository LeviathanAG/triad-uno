"""Training-only snapshots and filtered observations of the official referee.

Never deploy this module or pass real private simulator state into a policy.
For a belief probe, snapshots must come only from its observation-built sampled
world. All mutable containers/history/config/outcome/RNG are isolated; frozen
Card values and primitive JSON scalars may be shared safely.
"""
import copy
import os
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parent
SDK=Path(os.environ.get('UNO_SDK',ROOT.parent/'unoSDK'))
sys.path.insert(0,str(SDK/'local_evaluator/src'))
from triad_uno import engine
from triad_uno.cards import Card


def json_copy(value):
    if isinstance(value,dict):return {key:json_copy(item) for key,item in value.items()}
    if isinstance(value,list):return [json_copy(item) for item in value]
    if isinstance(value,tuple):return tuple(json_copy(item) for item in value)
    if isinstance(value,set):return {json_copy(item) for item in value}
    if value is None or type(value) in (str,int,float,bool):return value
    # Future unforeseen mutable payloads retain ordinary deep-copy semantics.
    return copy.deepcopy(value)


def snapshot(state):
    result=copy.copy(state)
    result.config=copy.copy(state.config)
    result.rng=random.Random(0);result.rng.setstate(state.rng.getstate())
    # The reference Card dataclass is frozen and contains immutable scalars.
    if any(type(card) is not Card for hand in state.hands for card in hand) or any(type(card) is not Card for card in state.draw_pile+state.discard_pile):
        raise TypeError('snapshot only supports official frozen Card values')
    result.hands=[list(hand) for hand in state.hands]
    result.draw_pile=list(state.draw_pile);result.discard_pile=list(state.discard_pile)
    result.finished_order=list(state.finished_order);result.finished_set=set(state.finished_set)
    result.history=json_copy(state.history);result.initial_seed_setup=json_copy(state.initial_seed_setup)
    result.outcome=copy.deepcopy(state.outcome)
    return result


def observe_with_actions(state,seat):
    """Same filtered JSON view as official observe, retaining its legal objects.

    Policies receive only the JSON view. The development simulator resolves a
    selected request id against this already-generated legal-action list, saving
    a second enumeration; no mutable trusted state is included in the view.
    """
    if not 0<=seat<state.config.players:raise ValueError('seat must be0..3')
    hand=state.hands[seat];ordered=engine.order_play_cards(list(hand))
    refs={card.uid:f'h{i}' for i,card in enumerate(ordered)}
    by_uid={card.uid:card for card in hand}
    actions=engine.legal_actions(state) if seat==state.current_seat else []
    legal=[]
    for i,action in enumerate(actions):
        entry={'action_id':f'a{i}','type':action.type}
        if action.card_uid is not None:
            entry['card_ref']=refs.get(action.card_uid)
            if action.card_uid in by_uid:entry['card']=by_uid[action.card_uid].to_public_dict()
        if action.color is not None:entry['color']=action.color
        legal.append(entry)
    observation={'seat':seat,'phase':state.phase,'turn_number':state.turn_count+1,
        'turns_completed':state.turn_count,'direction':state.direction,'active_color':state.active_color,
        'hand':[engine._card_view(card,refs[card.uid]) for card in ordered],
        'hand_sizes':engine._hand_sizes(state),'top_discard':state.discard_pile[-1].to_public_dict(),
        'draw_pile_size':len(state.draw_pile),'discard_pile_size':len(state.discard_pile),
        'history':list(state.history),'private_events':[],
        'drawn_card_ref':refs.get(state.drawn_card_uid) if state.phase==engine.AFTER_DRAW and seat==state.current_seat else None,
        'legal_actions':legal,'finished_order':list(state.finished_order),'is_finished':state.phase==engine.FINISHED}
    return observation,actions


def resolve_action(observation,actions,selected):
    # Deliberately match IDs from the filtered view; never accept an unchecked
    # array index supplied by a policy or use its card-ref as a real global uid.
    return next((action for entry,action in zip(observation['legal_actions'],actions)
                 if entry['action_id']==selected),None)
