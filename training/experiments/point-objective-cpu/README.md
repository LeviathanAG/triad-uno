Private development experiment, not a deployed policy. The current runtime on
main remains the tested route linear model.

The fixed 125-feature counterfactual dataset uses own-hand/public observations
and synthetic future-deck permutations, paired across actions. Real private
training simulator state is used only for rollouts and reward labels. Complete
source games are split before supervised fitting. All action pairs including
zero observed gaps remain eligible; positive paired standard-error weights are
used. Such estimated uncertainty weights can alter the conditional objective;
this experiment makes no unbiased-estimation claim.

Ranking, raw point-difference MSE, and Huber fit the same architecture and source
game split, initialization seeds, and minibatch schedule. Each uses a frozen
route model with positive clipped least-squares score calibration fitted on
training pairs only. A three-regularization by three-residual-scale grid exports
nine alternatives per objective. Calibration alone preserves the base ordering.

All 27 alternatives and the retained route are compared on the same fresh
58,000,000 seed series, with 256 four-seat blocks. Only that selected policy is
compared with the route incumbent on the fresh 59,000,000 series, 512 blocks per
regime. Final outcomes never select a different candidate. Promotion still
requires independent game evidence and runtime/contract/resource checks.

The two free T4 session slots were occupied; this small network therefore uses
a private free CPU notebook, preserving other running jobs. The checked-in
notebook is the exact submitted v1 source; no paid hardware is requested.
