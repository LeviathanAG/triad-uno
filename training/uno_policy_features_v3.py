"""Richer numeric own-hand encoding, canonicalized under colour permutations."""
from collections import Counter
import math
from uno_policy_v2 import FEATURE_NAMES as V2_FEATURES, action_features as v2_features, public_beliefs, symbol

KINDS=[('number',n) for n in range(10)]+[('skip',None),('reverse',None),('draw_two',None)]
FEATURE_NAMES=V2_FEATURES+[
 f'remaining_colour_slot_{slot}_{kind}_{value}' for slot in range(4) for kind,value in KINDS]+[
 'remaining_wild_count','remaining_wild_draw_four_count']+[
 f'action_number_{number}' for number in range(10)]+[
 f'opponent_slot_{slot}_{name}' for slot in range(3) for name in ('present','hand_size','one_card','colour_absence','symbol_absence','colour_preference')]+[
 'selected_is_active_colour','pending_penalty_size','turn_progress','active_opponents','closest_opponent_risk']


def action_features(observation,action,beliefs=None):
 beliefs=beliefs or public_beliefs(observation)
 hand=observation['hand'];play=action['type']=='play'
 card=action.get('card') or next((c for c in hand if c['card_ref']==action.get('card_ref')), {})
 selected=action.get('color') or card.get('color') or observation.get('active_color')
 remaining=[c for c in hand if not(play and c['card_ref']==action.get('card_ref'))]
 counts=Counter((c.get('color'),c.get('kind'),c.get('value') if c.get('kind')=='number' else None) for c in remaining)
 colours=['red','yellow','green','blue']
 order=[]
 for colour in (selected,observation.get('active_color')):
  if colour in colours and colour not in order:order.append(colour)
 def key(colour):
  vector=tuple(counts[(colour,kind,value)] for kind,value in KINDS)
  return sum(vector),vector
 order+=sorted((c for c in colours if c not in order),key=key,reverse=True)
 vector=[counts[(colour,kind,value)]/2. for colour in order for kind,value in KINDS]
 vector.extend(sum(c.get('kind')==kind for c in remaining)/4. for kind in ('wild','wild_draw_four'))
 vector.extend(float(card.get('kind')=='number' and card.get('value')==n) for n in range(10))
 seat=observation['seat'];sizes=observation['hand_sizes'];direction=observation.get('direction',1)
 if play and card.get('kind')=='reverse':direction=-direction
 finished=set(observation.get('finished_order',[]))
 opponents=[(seat+i*direction)%len(sizes) for i in range(1,len(sizes)) if (seat+i*direction)%len(sizes) not in finished]
 for slot in range(3):
  if slot>=len(opponents):vector.extend([0.]*6);continue
  other=opponents[slot]
  vector.extend([1.,sizes[other]/20.,float(sizes[other]==1),
   beliefs['colours'].get((other,selected),0.),beliefs['symbols'].get((other,symbol(card)),0.),
   float(beliefs['preference'].get(other)==selected)])
 vector.extend([float(selected==observation.get('active_color')),min(48,beliefs['penalty'])/12.,
  min(1000,observation.get('turn_number',0))/1000.,len(opponents)/3.,
  1./max(1,sizes[opponents[0]]) if opponents else 0.])
 return v2_features(observation,action,beliefs)+vector
