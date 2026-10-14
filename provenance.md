# UNO test-round candidate provenance

Team: TungMasterSoloClear (TECHTEAM11600). Private source repository:
https://github.com/LeviathanAG/triad-uno. The October 14 test-round source is prepared for the authorized 20:30 IST push.
The retained runtime/model source is verified at
`715e74042973c99838b537e7fac027fed63ba64a`; the complete prepared source commit
and any published image digest are recorded in the publication receipt. The
registry/upload destination is pending organizer confirmation; no image URI is
fabricated in the manifest.

The organizer UNO simulator used for training is the exact snapshot
`0cda3acc1bb5c05fa1803a8fbd7761770dc6d5fe`. The later SDK build/checker update
`a1e9592407c816edc51fce024d7344516c8d3074` is merged and does not change game
logic. Organizer protocol/vendor files retain their attribution and manifests.
No local evaluator or copied organizer engine is imported by deployed inference.

The policy is an original non-language linear action scorer using Python's
standard library. It has 24 weights: 18 frozen weights from the previous learned
policy, plus six newly trained residual weights. Own-hand sequence features use
an original deterministic search capped at 128 visited states and depth eight.
The search imagines consecutive own plays with no intervening opponent changes;
its result is a hand-structure feature, not a guaranteed future sequence.
Other features use public history to infer colour/symbol absences and preferences.
Inference reads only the filtered observation and recomputes history evidence on
every request. It consumes no hidden hands, simulator state, seed, or game ID.

The initial 18-weight model was trained by population cross-entropy optimization
in https://www.kaggle.com/code/xdantalianx/triad-tactics-training (version 1).
It is retained as `models/incumbent-v1.json`, with its original evaluation report.
The additional six weights were trained in the private CPU notebook
https://www.kaggle.com/code/xdantalianx/triad-uno-route-residual (version 1):
24 generations, population 24, 24 four-seat seed blocks per candidate. Finalists
were compared once on a fixed 256-block selection set. The first final test used
512 fresh blocks per opponent regime. Training source, selection and progress
are under `training/`; NumPy is a training dependency, never a runtime dependency.
Set `UNO_SDK` to this repository and pass `models/incumbent-v1.json` to reproduce
residual training. No external pretrained model or participant agent is used.

`models/evaluation.json` reports 2,048 games per regime. The varied league gain
was +0.07275 points per game (paired 95% interval +0.02588 to +0.11914).
`models/independent-evaluation.json` reports another 4,096 games per regime from
fresh seed 26,000,000 with the chosen checkpoint fixed. That replication gave a
smaller, uncertain league gain +0.00977 (interval -0.02319 to +0.04224), a
starter/heuristic gain +0.04663 (interval +0.01563 to +0.07861), and a random-rival
gain +0.03223 (interval -0.00245 to +0.06812). All tested point means improved;
these local results establish neither event winning probability nor universal
superiority. League first-place rate was slightly lower on the replication.

Validation: 300 exhaustive small-hand checks against authoritative SDK legality;
2,423 zero-residual compatibility/legality decisions; packaged inference matches
the training policy across 1,649 decisions in 16 complete games. SDK tests,
protocol smoke checks, three local restart games, and three network-disabled,
read-only Docker restart games pass with zero agent faults or fallbacks.
The CPU image continues to use the ordinary Dockerfile and claims no shared
PyTorch/CUDA size exemption. Later private neural experiments remain separate
until they demonstrate a gain over this retained checkpoint.

The test-round manifest identifies team TungMasterSoloClear / TECHTEAM11600 and
declares the actual loaded `models/policy.json` asset (SHA-256
`d458ef2894c0ce51d62ed30a4ad1b685c85db2c1a97417a8a93efab16d5abe14`).
The CPU parent restores the official literal `FROM python:3.11-slim` declaration
for organizer compatibility. The exact tested index and Linux/amd64 platform
digests are recorded in the admission verification receipt; the tag is mutable.
Current admission preparation repeats the eight protocol checks, the SDK suite
(27 passed, six optional checks skipped), and three local plus three restricted
Docker restart games, with zero unintended faults or fallbacks. The test image
uses UID10000:10000, no network, read-only root, one CPU, one GiB RAM, and bounded
scratch. Required SDK resource labels remain unchanged. Later joint24 and
mirrored searches failed to establish an upgrade; the shipped route policy is
unchanged from the verified source above.

Private belief-rollout efficiency experiment (not deployed): audited snapshots
matched 7,808 filtered observations and 300 transitions. Incremental public
fold matched 18,892 folds and 4,691 route choices, including reset cases.
The fold cache assumes the pinned SDK appends events without changing old
events. In 232 fixed synthetic-world rollouts, identical returns took 1.85x
less time with both speed paths. Prepared sampling preserves the original
public-only samples and RNG state. No actual hidden cards or real game seed
is supplied to the probe. Official referee imports are development-only.

Fresh 112M development comparison (256 full games per regime) selected fast16
by league placement points with the incumbent eligible. League gain +0.015625
(95% paired seed-block bootstrap CI -0.003906 to +0.039063); starter +0.027344
(-0.007813 to +0.066406); random -0.019531 (-0.066406 to +0.019531). These
small, mixed results do not justify changing the retained runtime. The uniform
eight-style hypothetical rival mixture did not win selection. Approximate
sampled-model Gaussian override bounds do not guarantee real-game strength.
The fixed fast16 fresh 113M gate completed privately on free Kaggle CPU as
xdantalianx/triad-uno-belief-fast16 version 1, 512 four-seat blocks per unique
regime (league/starter/random). Maximum 16 trials, minimum 4, budget 0.15s,
all hypothetical rivals use retained route. Actual opponent identities are
unknown. Tonight's runtime remains unchanged; corrected test-round main ee37f8c restores the official Python base declaration.
Restricted development-only Docker check (1 CPU, 1 GiB, read-only, network
none, pids 128): 16 games / 1,438 decisions; maximum 0.153045s decision and
26,402,816 bytes peak RSS. Zero hypothetical-world legality rejections.
The 115.1M resource fixture is not a policy strength selection family.

Two additional isolated speed attempts were rejected. A precomputed own-hand
route search matched 12,000 cases across hands of size 0..24 and varying
visit/depth budgets, but was slower (0.859x baseline speed). Native JSON copy
of the pinned public-history/initial-setup schema matched 742 views, 1,484
isolated snapshots and 210 full fixed-work rollouts; whole-rollout speed was
0.997x, so it was not integrated. Generic tuple/set copy semantics were never
replaced. The live fast16 gate and all runtime code remain unchanged.

The fixed fast16 fresh 113M gate failed to establish an upgrade: 2,048 games
per regime, league gain -0.004883 (95% paired seed-block CI -0.013672 to
+0.003418), starter +0.001465 (-0.008301 to +0.011230), random -0.002441
(-0.015137 to +0.010254). All downloaded JSON and ten embedded source hashes
validated. Cloud completed approximately 4.5--4.9 trials per probe under the
0.15-second budget, versus 12.7 in the local restricted Docker resource
fixture; these fixtures differ and are not paired strength evidence. The
failed gate is retained as reported and the route policy remains deployed.

New private 118M development comparison follows the failed fixed gate:
0.5-second budget, maximum 32 paired trials, either 2 or 4 semantic candidate
actions, approximate route or uniform eight-style rival priors. The common
league placement-point selection includes the incumbent. 119M is reserved
for one fresh fixed-candidate gate if development selects an improvement.
These changes affect experiments only; runtime policy remains retained.

The four larger-budget configurations passed restricted 1-CPU/1-GiB Docker
resource checks (eight complete games each): maximum decision 0.501939s and
peak RSS 26,001,408 bytes, no sampled-world legality rejections. These 121.1M
fixtures are resource checks, not strength evidence. Locally the probes
completed about 27--29 trials each. The two-action mixture fixture exhausted
all 32 attempts below its 0.5s budget; 125/2,240 attempts still produced an
incomplete paired rollout at the 128-step cap, so truncated/model uncertainty
remains. A further per-decision route-feature memoization experiment matched
440 fixed-work rollouts but yielded only 1.015x overall speed in one timing
pass, providing no useful new headroom; it was not integrated into the job or
runtime. Cache keys used queried card multisets and public top/colour only;
the hypothetical rival queries were never factual hidden cards.

Private isolated risk-residual experiment (not deployed): six own/public
response, one-card finish and penalty-pressure features use a uniform per-seat
hard-constraint prior, not factual rival cards or a joint/style posterior.
The retained 24 route weights are frozen; only six bounded coefficients
[-8,8] are tuned by small CEM against eight of our own diverse trained/archive
policies, three distinct rivals per seed block and four seat rotations.
Zero residual is eligible in training and common league selection. Seed
families are 140M training,141M selection,142M fresh final;143M is reserved
for independent replication if supported. No competitor private source is used.
The finite/legality/public-constraint audit is saved alongside the helper.
Additional scorer checks matched 1,810 zero-residual decisions and verified
1,810 nonzero legal decisions plus845 incomplete-history declines to the
unchanged scalar prior. No runtime, model, manifest or main branch change.

Prototype scorer resource probe (arbitrary six coefficients, not the eventual
trained candidate): restricted 1CPU/1GiB/read-only/network-none Docker, eight
full development games and a synthetic108-card partition with104 own cards.
Maximum decision 0.003835s, peak RSS 24551424bytes,
all selected actions legal. The selected learned checkpoint must repeat its
actual inference/export resource gates before deployment.
