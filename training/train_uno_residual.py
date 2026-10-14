"""GPU pairwise policy distillation, followed by real held-out SDK game tests.

Split complete source games, not actions, for supervised validation. Preserve
incumbent until a separately tested deterministic candidate shows useful gains.
"""
import argparse
import concurrent.futures
import copy
import json
import multiprocessing
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from uno_policy import INITIAL_WEIGHTS
from uno_policy_v2 import FEATURE_NAMES
from uno_mlp_policy import Policy
from train_uno_league import evaluate_job, paired_interval, summarize


class Residual(nn.Module):
    def __init__(self, base, width):
        super().__init__()
        self.register_buffer('base',torch.tensor(base,dtype=torch.float32))
        self.hidden=nn.Linear(len(base),width)
        self.output=nn.Linear(width,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,x):
        residual=self.output(F.relu(self.hidden(x))).squeeze(-1)
        return x@self.base+residual,residual


def export(model,base,metadata):
    names=metadata['source_dataset']['input_features']
    version={3:'v2',4:'v3'}.get(metadata['source_dataset'].get('feature_version'),'v1')
    return {'format':'triad-uno-residual-mlp-'+version,'features':names,'base_weights':base,
            'hidden_weight':model.hidden.weight.detach().cpu().tolist(),
            'hidden_bias':model.hidden.bias.detach().cpu().tolist(),
            'output_weight':model.output.weight.detach().cpu().flatten().tolist(),
            'output_bias':float(model.output.bias.detach().cpu()[0]),'training':metadata}


def make_pairs(data,seed):
    # NPZ access decompresses a member. Load each array once, rather than once
    # per state/pair, especially the new Monte Carlo trial matrix.
    game_ids=data['game_ids'];targets=data['targets'];offsets=data['offsets']
    trial_targets=data['trial_targets'] if 'trial_targets' in data else None
    ids=np.unique(game_ids)
    rng=np.random.default_rng(seed);rng.shuffle(ids)
    val_ids=set(ids[:max(1,len(ids)//5)].tolist())
    train,val=[],[]
    for state,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
        target=val if int(game_ids[state]) in val_ids else train
        q=targets[a:b]
        for i in range(len(q)):
            for j in range(i+1,len(q)):
                gap=float(q[i]-q[j])
                if abs(gap)>=.5:
                    weight=min(1.,abs(gap)/3.)
                    if trial_targets is not None and trial_targets.shape[1] > 1:
                        differences=trial_targets[a+i]-trial_targets[a+j]
                        stderr=float(np.std(differences,ddof=1)/np.sqrt(len(differences)))
                        # Reward differences sharing future draws, not independent
                        # action variances, provide the relevant noise estimate.
                        weight*=max(.05,1.-stderr/abs(gap))
                    target.append((a+i,a+j,1./(1.+np.exp(-gap/.75)),weight))
    if not train or not val:raise ValueError('not enough informative game-held-out pairs')
    return np.asarray(train,dtype=np.float64),np.asarray(val,dtype=np.float64),len(ids)-len(val_ids),len(val_ids)


def loss(model,x,pairs,regularization):
    first=pairs[:,0].long();second=pairs[:,1].long()
    scores_a,residual_a=model(x[first]);scores_b,residual_b=model(x[second])
    point=F.binary_cross_entropy_with_logits(scores_a-scores_b,pairs[:,2],reduction='none')
    rank=(point*pairs[:,3]).sum()/pairs[:,3].sum()
    regularizer=regularization*(residual_a.square().mean()+residual_b.square().mean())/2.
    return rank+regularizer,rank


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--evaluation-incumbent',type=Path,help='optional current retained policy for actual-game comparisons')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--width',type=int,default=32)
    parser.add_argument('--batch',type=int,default=2048)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--validation-blocks',type=int,default=128)
    parser.add_argument('--test-blocks',type=int,default=256)
    parser.add_argument('--validation-seed-start',type=int,default=11_000_000)
    parser.add_argument('--test-seed-start',type=int,default=12_000_000)
    parser.add_argument('--seed',type=int,default=20261017)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--residual-scales',default='1',help='comma-separated conservative learned-residual scales')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'uno-residual-output')
    args=parser.parse_args()
    scales=[float(s) for s in args.residual_scales.split(',')]
    if not scales or any(not np.isfinite(s) or not 0<s<=1 for s in scales):parser.error('residual scales must be finite in(0,1]')
    if min(args.epochs,args.width,args.batch,args.workers,args.validation_blocks,args.test_blocks)<1:parser.error('positive counts required')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('requested authorized GPU unavailable; do not silently select other hardware')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((args.data.parent/'metadata.json').read_text())
    names=FEATURE_NAMES
    if metadata.get('feature_version')==4:
        from uno_policy_features_v4 import FEATURE_NAMES as rich_names
        names=rich_names
    elif metadata.get('feature_version')==3:
        from uno_policy_features_v3 import FEATURE_NAMES as rich_names
        names=rich_names
    if metadata['input_features']!=names:raise ValueError('feature schema mismatch')
    data=np.load(args.data)
    if metadata.get('feature_version')==4:
        weights=metadata['incumbent']['weights']
        if metadata['incumbent']['format']!='triad-uno-route-linear-v1' or len(weights)!=24:raise ValueError('route feature schema requires retained route teacher')
        base=weights[:18]+[0.]*(len(names)-24)+weights[18:]
    else:
        base=metadata['incumbent']['weights']+[0.]*(len(names)-len(metadata['incumbent']['weights']))
    train,val,train_games,val_games=make_pairs(data,args.seed)
    x=torch.as_tensor(data['features'],dtype=torch.float32,device=args.device)
    train=torch.as_tensor(train,dtype=torch.float32,device=args.device)
    val=torch.as_tensor(val,dtype=torch.float32,device=args.device)
    print(json.dumps({'device':args.device,'devices':torch.cuda.device_count(),'states':len(data['game_ids']),
                      'actions':len(x),'train_games':train_games,'validation_games':val_games,
                      'train_pairs':len(train),'validation_pairs':len(val)}),flush=True)
    candidates=[];history=[];started=time.time()
    for index,regularization in enumerate((.01,.03,.1)):
        torch.manual_seed(args.seed+index)
        model=Residual(base,args.width).to(args.device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
        with torch.no_grad():
            _,baseline_rank=loss(model,x,val,regularization)
        best_loss=float(baseline_rank);best_epoch=0;best=copy.deepcopy(model.state_dict())
        for epoch in range(args.epochs):
            model.train();order=torch.randperm(len(train),device=args.device)
            for start in range(0,len(order),args.batch):
                batch=train[order[start:start+args.batch]]
                objective,_=loss(model,x,batch,regularization)
                optimizer.zero_grad(set_to_none=True);objective.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                objective,rank=loss(model,x,val,regularization)
            current=float(rank)
            if current<best_loss:
                best_loss,best_epoch,best=current,epoch+1,copy.deepcopy(model.state_dict())
            if (epoch+1)%10==0:
                row={'regularization':regularization,'epoch':epoch+1,'validation_rank_loss':current,'best_rank_loss':best_loss,'seconds':round(time.time()-started,1)}
                history.append(row);print(json.dumps(row),flush=True)
        model.load_state_dict(best)
        document=export(model,base,{'method':'paired counterfactual policy distillation with residual scorer',
            'seed':args.seed+index,'regularization':regularization,'best_epoch':best_epoch,
            'validation_rank_loss':best_loss,'baseline_validation_rank_loss':float(baseline_rank),
            'training_games':train_games,'validation_games':val_games,'source_dataset':metadata,
            'sdk_commit':metadata['sdk_commit']})
        check=data['features'][:min(200,len(data['features']))]
        with torch.no_grad():
            tensor=torch.as_tensor(check,dtype=torch.float32,device=args.device)
            _,residual=model(tensor)
            base_scores=tensor@model.base
        for scale in scales:
            scaled=copy.deepcopy(document)
            scaled['output_weight']=[scale*w for w in document['output_weight']]
            scaled['output_bias']=scale*document['output_bias']
            scaled['training']['residual_scale']=scale
            deployed=Policy(scaled) # Deployment dimensions/finite parameters.
            expected=(base_scores+scale*residual).cpu().numpy()
            actual=np.asarray([deployed.score_features(row.tolist()) for row in check])
            max_error=float(np.max(np.abs(actual-expected)))
            if max_error>1e-4:raise RuntimeError('exported standard-library scorer differs from training network')
            candidate_index=len(candidates)
            print(json.dumps({'candidate':candidate_index,'residual_scale':scale,'export_max_score_error':max_error}),flush=True)
            path=args.output/f'candidate-{candidate_index}.json';path.write_text(json.dumps(scaled,indent=2)+'\n')
            candidates.append(scaled)
        del model,optimizer
    (args.output/'training-progress.json').write_text(json.dumps(history,indent=2)+'\n')
    incumbent_document=metadata['incumbent']
    if args.evaluation_incumbent:
        incumbent_document=json.loads(args.evaluation_incumbent.read_text())
    kind={'triad-uno-linear-v1':'v1','triad-uno-route-linear-v1':'route'}.get(incumbent_document.get('format'))
    if kind is None:raise ValueError('unsupported actual-game evaluation incumbent')
    incumbent={'kind':kind,'weights':incumbent_document['weights']}
    source_kind={'triad-uno-linear-v1':'v1','triad-uno-route-linear-v1':'route'}.get(metadata['incumbent']['format'])
    if source_kind is None:raise ValueError('unsupported rollout teacher format')
    source_incumbent={'kind':source_kind,'weights':metadata['incumbent']['weights']}
    attacking=list(INITIAL_WEIGHTS)
    for i in (2,4,6,10):attacking[i]+=2.
    preserve=list(INITIAL_WEIGHTS);preserve[5]-=2.;preserve[6]-=2.
    reference=[{'kind':'starter'},{'kind':'heuristic'},{'kind':'random'},incumbent,
               {'kind':'v1','weights':INITIAL_WEIGHTS},{'kind':'v1','weights':attacking},
               {'kind':'v1','weights':preserve}]
    if source_incumbent != incumbent:reference.insert(4,source_incumbent)
    specs=[incumbent]+[{'kind':'mlp','model':m} for m in candidates]
    seeds=list(range(args.validation_seed_start,args.validation_seed_start+args.validation_blocks))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=list(pool.map(evaluate_job,[(s,seeds,reference,'league') for s in specs]))
        chosen=max(range(len(results)),key=lambda i:results[i]['points'])
        selection={'selected':chosen,'scores':[summarize(r) for r in results],
                   'paired_gains':[paired_interval(r,results[0]) for r in results[1:]],'evaluation_incumbent':incumbent_document}
        (args.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n');print(json.dumps({'game_selection':selection}),flush=True)
        selected=specs[chosen]
        if chosen:
            (args.output/'uno-policy-mlp.json').write_text(json.dumps(candidates[chosen-1],indent=2)+'\n')
        else:
            (args.output/'retained-incumbent.json').write_text(json.dumps(incumbent_document,indent=2)+'\n')
        seeds=list(range(args.test_seed_start,args.test_seed_start+args.test_blocks));modes=('league','starter','heuristic','random')
        results=list(pool.map(evaluate_job,[(s,seeds,reference,m) for m in modes for s in (selected,incumbent)]))
        report={mode:{'candidate':summarize(results[2*i]),'incumbent':summarize(results[2*i+1]),
                      'paired_gain':paired_interval(results[2*i],results[2*i+1])} for i,mode in enumerate(modes)}
        (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'held_out_evaluation':report}),flush=True)


if __name__=='__main__':main()
