"""Full own/public encoding plus exact retained hand-route feature values."""
from uno_policy_features_v3 import FEATURE_NAMES as RICH_NAMES, action_features as rich_features
from uno_route_policy import EXTRA_NAMES, action_features as route_features

FEATURE_NAMES=RICH_NAMES+['route_'+name for name in EXTRA_NAMES]


def action_features(observation,action,beliefs=None):
    return rich_features(observation,action,beliefs)+route_features(observation,action,beliefs)[-len(EXTRA_NAMES):]
