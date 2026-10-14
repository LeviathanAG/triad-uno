Private development search, no changes to main or the deployed model.

All24 route coefficients are eligible for optimization. The current retained
route is the warm start; each coefficient remains within plus/minus25% of
max(1,abs(original coefficient)). Initial perturbation standard deviation is10%
of that scale. Cross-entropy search uses shared four-seat seed blocks per
population, and keeps the exact incumbent eligible during selection. This is a
bounded practical search, with no claim of global optimality.

The fixed11-member training/selection league contains the retained route,
learnedv1, initial, attacking and wild-preserving linear variants, starter,
heuristic and random baselines, plus three frozen neural alternatives (original
route network, calibrated ranking winner, calibrated MSE selection winner).
None of these alternatives is promoted by being included as a rival.

Training starts80,000,000 (24generations×24population×32seat blocks); a common
fresh81,000,000 selection series256blocks picks one candidate including the
incumbent. Only this candidate receives the independent82,000,000 final series
512blocks/regime. Smoke tests use distinct84M–89M series. The rival pool differs
from prior experiments: use paired differences against the retained route in
this same pool, not raw point comparisons across reports.

The checked-in notebook is private free CPU v1. Main715e740 remains held.
Any promotion still requires independent strength and exact runtime checks.
