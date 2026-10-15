"""Fresh fixed conditioned163M comparator; all paired points/wins retained.

Development selection precedes163M. Selection eligibility is predeclared:
trained-league paired bootstrap lower bound >0 and both ancillary mean gains
at least -.02. This is an empirical guard, not a guarantee of event strength.
Only an eligible fixed candidate may advance to the reserved164M final.
"""
import argparse, concurrent.futures, datetime, hashlib, json, multiprocessing, os, signal, time
from pathlib import Path
from compare_uno_conditioned_budgets import evaluate, merge
from train_uno_league import paired_interval, summarize


def enforce_compute_deadline():
    """Each spawned worker independently exits by05:28UTC, even if orphaned."""
    cutoff=datetime.datetime(2026,10,15,5,28,tzinfo=datetime.timezone.utc).timestamp()
    remaining=cutoff-time.time()
    if remaining<=0:os._exit(124)
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124))
    signal.setitimer(signal.ITIMER_REAL,remaining)


def main():
    enforce_compute_deadline()
    p=argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--trained-pool', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--blocks', type=int, default=256)
    p.add_argument('--seed-start', type=int, default=163000000)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args(); selected=json.loads(a.selection.read_text())
    config=selected['configuration']
    if config != {'name':'conditioned_1500ms64_actions2','kind':'conditioned','budget':1.5,'max_trials':64,'max_actions':2}:
        raise ValueError('This gate accepts only the frozen162M conditioned selection')
    weights=json.loads(a.model.read_text())['weights']
    reference=json.loads(a.trained_pool.read_text())
    chunks=[list(range(s,min(s+16,a.seed_start+a.blocks)))
            for s in range(a.seed_start,a.seed_start+a.blocks,16)]
    modes=('trained','route','random'); keys=[]; tasks=[]
    for mode in modes:
        for i,c in enumerate(({'name':'retained_route','kind':'route'},config)):
            for chunk in chunks:
                keys.append((mode,i));tasks.append((c,weights,chunk,reference,mode))
    collected={key:[] for key in keys}
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,
            mp_context=multiprocessing.get_context('spawn'),initializer=enforce_compute_deadline) as pool:
        for index,(key,row) in enumerate(zip(keys,pool.map(evaluate,tasks))):
            collected[key].append(row)
            print(json.dumps({'completed':index+1,'total':len(tasks),
                             'mode':key[0],'candidate':bool(key[1]),'stats':row['probe_stats']}),flush=True)
    report={}
    for mode in modes:
        base=merge(collected[mode,0]); candidate=merge(collected[mode,1])
        report[mode]={'incumbent':summarize(base),'candidate':summarize(candidate),
                      'paired_gain':paired_interval(candidate,base),
                      'block_points_candidate':candidate['block_points'],
                      'block_points_incumbent':base['block_points'],
                      'block_wins_candidate':candidate['block_wins'],
                      'block_wins_incumbent':base['block_wins']}
    eligible=(report['trained']['paired_gain']['ci95'] is not None
              and report['trained']['paired_gain']['ci95'][0]>0
              and all(report[m]['paired_gain']['mean_gain']>=-.02 for m in ('route','random')))
    report['eligibility']={'eligible_for_fresh_final':eligible,
        'criterion':'trained lower95 bootstrap bound>0; route/random mean gain>=-.02',
        'empirical_selection_guard_not_event_guarantee':True}
    report['provenance']={'fixed_selection':selected,'seed_start':a.seed_start,
        'blocks':a.blocks,'workers':a.workers,'development_only':True,
        'final_seed_reserved':164000000,'actual_hidden_or_game_seed_policy_inputs':False,
        'scoring':'5/3/2/1; actual unfinished seats share remaining placement points',
        'uncertainty':'uniform public belief and route rival model approximate; Gaussian sampled-world test heuristic',
        'policy_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
        'trained_pool_sha256':hashlib.sha256(a.trained_pool.read_bytes()).hexdigest(),
        'selection_sha256':hashlib.sha256(a.selection.read_bytes()).hexdigest()}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({m:report[m]['paired_gain'] for m in modes}),flush=True)


if __name__=='__main__':main()
