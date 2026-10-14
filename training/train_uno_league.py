"""PSRO-inspired archive league with paired seat blocks and separate test seeds.

This is a practical population training method, not an exact equilibrium solver.
All policies act on filtered observations; simulator state only computes rewards.
"""
import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
SDK = Path(os.environ.get("UNO_SDK", ROOT.parent / "unoSDK"))
sys.path.insert(0, str(SDK / "local_evaluator" / "src"))
from triad_uno import engine
from triad_uno.baselines import HeuristicAgent, RandomAgent
from uno_policy import Policy as V1, INITIAL_WEIGHTS
from uno_policy_v2 import Policy as V2, FEATURE_NAMES


class Starter:
    def select(self, observation):
        legal = observation["legal_actions"]
        return next((a["action_id"] for a in legal if a["type"] == "play"), legal[0]["action_id"])


def make_policy(spec, seed):
    if spec["kind"] == "route":
        from uno_route_policy import Policy as RoutePolicy
        return RoutePolicy(spec["weights"])
    if spec["kind"] == "mlp":
        from uno_mlp_policy import FastPolicy
        return FastPolicy(spec["model"])
    if spec["kind"] == "v1":
        return V1(spec["weights"])
    if spec["kind"] == "v2":
        return V2(spec["weights"])
    if spec["kind"] == "random":
        return RandomAgent(seed)
    if spec["kind"] == "starter":
        return Starter()
    return HeuristicAgent(seed)


def score_game(seed, seat, candidate, rivals):
    actors = [None] * 4
    actors[seat] = make_policy(candidate, seed * 10 + 9)
    for offset, spec in enumerate(rivals, 1):
        actors[(seat + offset) % 4] = make_policy(spec, seed * 10 + offset)
    state = engine.new_game(engine.GameConfig(max_turns=1000), seed)
    while state.phase != engine.FINISHED and seat not in state.finished_set:
        current = state.current_seat
        observation = engine.observe(state, current)
        selected = actors[current].select(observation)
        action = engine.engine_action_from_id(state, current, selected)
        if action is None:
            raise RuntimeError("policy selected an illegal action")
        if not engine.step(state, action).ok:
            raise RuntimeError("policy transition failed")
    if seat in state.finished_order:
        placement = state.finished_order.index(seat)
        return state.config.position_points[placement], float(placement == 0)
    outcome = state.outcome
    remaining = state.config.position_points[len(outcome.placements):]
    return sum(remaining) / len(outcome.unfinished), 0.


def evaluate_job(job):
    candidate, seeds, pool, mode = job
    blocks, win_blocks = [], []
    for seed in seeds:
        rng = random.Random(seed ^ 0x4C4541475545)
        if mode == "league":
            rivals = [rng.choice(pool) for _ in range(3)]
        else:
            rivals = [{"kind": mode}] * 3
        outcomes = [score_game(seed, seat, candidate, rivals) for seat in range(4)]
        blocks.append(sum(row[0] for row in outcomes) / 4.)
        win_blocks.append(sum(row[1] for row in outcomes) / 4.)
    return {"points": float(np.mean(blocks)), "win_rate": float(np.mean(win_blocks)),
            "games": 4 * len(seeds), "block_points": blocks, "block_wins": win_blocks}


def paired_interval(candidate, incumbent, seed=20261014):
    differences = np.asarray(candidate["block_points"]) - np.asarray(incumbent["block_points"])
    rng = np.random.default_rng(seed)
    if len(differences) < 2:
        return {"mean_gain": float(np.mean(differences)), "ci95": None}
    means = differences[rng.integers(0, len(differences), (5000, len(differences)))].mean(axis=1)
    return {"mean_gain": float(differences.mean()),
            "ci95": [float(v) for v in np.quantile(means, [.025, .975])],
            "unit": "points per game", "sampling_unit": "four-seat seed block"}


def summarize(result):
    return {k: v for k, v in result.items() if not k.startswith("block_")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", type=Path, default=SDK / "models" / "policy.json")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--train-blocks", type=int, default=16)
    parser.add_argument("--validation-blocks", type=int, default=128)
    parser.add_argument("--test-blocks", type=int, default=256)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--output", type=Path, default=ROOT / "uno-league-output")
    args = parser.parse_args()
    if min(args.iterations, args.population, args.train_blocks, args.validation_blocks, args.test_blocks, args.workers) < 1:
        parser.error("counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    document = json.loads(args.incumbent.read_text())
    incumbent = {"kind": "v1", "weights": document["weights"]}
    mean = np.asarray(document["weights"] + [0.] * (len(FEATURE_NAMES) - len(document["weights"])))
    best = mean.copy()
    sigma = np.full(len(mean), 1.5)
    rng = np.random.default_rng(args.seed)
    attacking = list(INITIAL_WEIGHTS)
    for i in (2, 4, 6, 10):
        attacking[i] += 2.
    preserve = list(INITIAL_WEIGHTS)
    preserve[5] -= 2.
    preserve[6] -= 2.
    reference = [{"kind": "starter"}, {"kind": "heuristic"}, {"kind": "random"},
                 incumbent, {"kind": "v1", "weights": INITIAL_WEIGHTS},
                 {"kind": "v1", "weights": attacking}, {"kind": "v1", "weights": preserve}]
    archive = list(reference)
    selection_seeds = list(range(6_000_000, 6_000_000 + args.validation_blocks))
    history, started = [], time.time()
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        baseline = list(pool.map(evaluate_job, [(incumbent, selection_seeds, reference, "league")]))[0]
        selected_score = baseline["points"]
        for generation in range(args.iterations):
            if generation and generation % 10 == 0:
                mean, sigma = best.copy(), np.full(len(mean), 1.5)
            population = rng.normal(mean, sigma, (args.population, len(mean)))
            # Broad probes keep the population from collapsing into one local basin.
            probes = max(1, args.population // 5)
            population[-probes:] = rng.normal(best, 3., (probes, len(mean)))
            population = np.clip(population, -20., 20.)
            population[0] = mean
            if args.population > 1:
                population[1] = best
            seeds = list(range(30_000 + generation * args.train_blocks, 30_000 + (generation + 1) * args.train_blocks))
            scores = list(pool.map(evaluate_job,
                [({"kind": "v2", "weights": w.tolist()}, seeds, archive, "league") for w in population]))
            ranked = sorted(range(len(scores)), key=lambda i: (scores[i]["points"], scores[i]["win_rate"]), reverse=True)
            elite = population[ranked[:max(2, args.population // 5)]]
            mean = .35 * mean + .65 * elite.mean(axis=0)
            sigma = np.maximum(.3, .35 * sigma + .65 * elite.std(axis=0))
            candidate = population[ranked[0]]
            spec = {"kind": "v2", "weights": candidate.tolist()}
            validation = list(pool.map(evaluate_job, [(spec, selection_seeds, reference, "league")]))[0]
            if validation["points"] > selected_score:
                best, selected_score = candidate.copy(), validation["points"]
            if generation % 4 == 3 and not any(s.get("kind") == "v2" and np.linalg.norm(np.asarray(s["weights"]) - candidate) < .2 for s in archive):
                archive.append(spec)
                archive = reference + archive[len(reference):][-12:]
            record = {"generation": generation + 1, "train": summarize(scores[ranked[0]]),
                      "validation": summarize(validation), "paired_validation_gain": paired_interval(validation, baseline),
                      "selected_points": selected_score, "archive_size": len(archive),
                      "elapsed_seconds": round(time.time() - started, 1)}
            history.append(record)
            print(json.dumps(record), flush=True)
            V2(best).save(args.output / "uno-policy-v2.json", {"method": "archive-league cross-entropy policy optimization",
                "seed": args.seed, "generations": generation + 1, "population": args.population,
                "train_blocks": args.train_blocks, "validation_blocks": args.validation_blocks,
                "selected_validation_points": selected_score, "incumbent_validation_points": baseline["points"],
                "sdk_commit": "0cda3acc1bb5c05fa1803a8fbd7761770dc6d5fe"})
            (args.output / "training-progress.json").write_text(json.dumps(history, indent=2) + "\n")
            (args.output / "archive.json").write_text(json.dumps(archive, indent=2) + "\n")
        test_seeds = list(range(7_000_000, 7_000_000 + args.test_blocks))
        chosen = {"kind": "v2", "weights": best.tolist()}
        modes = ("league", "starter", "heuristic", "random")
        jobs = [(spec, test_seeds, reference, mode) for mode in modes for spec in (chosen, incumbent)]
        results = list(pool.map(evaluate_job, jobs))
        report = {mode: {"candidate": summarize(results[2*i]), "incumbent": summarize(results[2*i+1]),
                        "paired_gain": paired_interval(results[2*i], results[2*i+1])}
                  for i, mode in enumerate(modes)}
        (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"held_out_evaluation": report}), flush=True)


if __name__ == "__main__":
    main()
