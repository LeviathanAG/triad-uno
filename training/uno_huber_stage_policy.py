"""Portable own/public stage-dependent residual strength; no engine imports."""
import math
from uno_mlp_policy import Policy as Neural
from uno_route_policy import Policy as Route

STAGES=['multi_opponents_hand_le3','multi_opponents_hand4to6','multi_opponents_hand_ge7',
        'one_opponent_hand_le3','one_opponent_hand4to6','one_opponent_hand_ge7']


class Policy:
    def __init__(self,route_weights,checkpoint,strengths):
        if len(strengths)!=6 or not all(math.isfinite(x) and 0<=x<=1.5 for x in strengths):
            raise ValueError('invalid stage residual strengths')
        self.strengths=list(strengths);self.route=Route(route_weights);self.neural=Neural(checkpoint)

    def stage(self,o):
        size=len(o['hand']);bin=0 if size<=3 else 1 if size<=6 else 2
        rivals=len(o['hand_sizes'])-len(o.get('finished_order',[]))-1
        return bin+(3 if rivals<=1 else 0)

    def select(self,o):
        legal=o['legal_actions']
        if len(legal)==1:return legal[0]['action_id']
        strength=self.strengths[self.stage(o)]
        if strength==0:return self.route.select(o)
        if strength==1:return self.neural.select(o)
        from uno_policy_v2 import public_beliefs
        beliefs=public_beliefs(o)
        def score(action):
            vector=self.neural.action_features(o,action,beliefs)
            hidden=[max(0.,sum(w*x for w,x in zip(row,vector))+bias)
                    for row,bias in zip(self.neural.w1,self.neural.b1)]
            base=sum(w*x for w,x in zip(self.neural.base,vector))
            residual=sum(w*x for w,x in zip(self.neural.w2,hidden))+self.neural.b2
            return base+strength*residual
        return max(legal,key=score)['action_id']
