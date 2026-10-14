"""Six own/public unseen-response residuals over an unchanged route prior."""
import math
from uno_route_policy import Policy as RoutePolicy,action_features,FEATURE_NAMES as BASE_NAMES
from uno_policy_v2 import public_beliefs
from uno_unseen_risk import PreparedRisks,FEATURE_NAMES as RISK_NAMES

FEATURE_NAMES=BASE_NAMES+RISK_NAMES


class Policy:
    def __init__(self,base_weights,residual_weights):
        self.base=RoutePolicy(base_weights);self.base_weights=list(base_weights)
        if len(residual_weights)!=6 or not all(math.isfinite(w) for w in residual_weights):
            raise ValueError('six finite risk residual coefficients required')
        self.residual_weights=list(residual_weights);self.declines=0

    def select(self,observation):
        legal=observation['legal_actions']
        if len(legal)==1 or not any(self.residual_weights):return self.base.select(observation)
        try:
            prepared=PreparedRisks(observation)
            risks=[prepared.features(action) for action in legal]
            if any(len(vector)!=6 or not all(math.isfinite(v) for v in vector) for vector in risks):
                raise ValueError('invalid risk vector')
        except (ValueError,KeyError,TypeError,StopIteration):
            self.declines+=1
            return self.base.select(observation)
        beliefs=public_beliefs(observation);cache={}
        scores=[sum(w*x for w,x in zip(self.base_weights,action_features(observation,a,beliefs,cache)))
                +sum(w*x for w,x in zip(self.residual_weights,risk)) for a,risk in zip(legal,risks)]
        if not all(math.isfinite(score) for score in scores):
            self.declines+=1
            return self.base.select(observation)
        return legal[max(range(len(scores)),key=lambda i:scores[i])]['action_id']
