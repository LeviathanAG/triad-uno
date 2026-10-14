import json
from pathlib import Path
ROOT=Path(__file__).parent
files=['uno_policy.py','uno_policy_v2.py','uno_policy_features_v3.py','uno_policy_features_v4.py','uno_route_policy.py','uno_mlp_policy.py','train_uno_league.py','evaluate_uno_checkpoint.py','train_uno_residual.py','compare_uno_objectives.py']
sources={f:(ROOT/f).read_text() for f in files}
SDK=ROOT.parent if (ROOT.parent/'models/policy.json').exists() else ROOT.parent/'unoSDK'
route=json.loads((SDK/'models/policy.json').read_text())
candidate=json.loads((ROOT/'kaggle-uno-route-neural-output/output/uno-policy-mlp.json').read_text())
for slug, title, gpu in [('route-neural-verification','Triad UNO Route Neural Verification',False),('point-objective-comparison','Triad UNO Point Objective Comparison',True)]:
    directory=ROOT/('kaggle-uno-'+slug);directory.mkdir(exist_ok=True)
    header='Private free compute experiment. Observation-only own/public features. Main held. Fresh fixed-checkpoint57m replication.' if not gpu else 'Private free T4 comparison: ranking/MSE/Huber, all zero-gap pairs, paired uncertainty weights, training-only base calibration. Common58m game selection, once-only59m final. No event submission.'
    code='import pathlib,subprocess,json,os\nwork=pathlib.Path("/kaggle/temp/triad-uno-'+slug+'")\nwork.mkdir(parents=True,exist_ok=True)\nsources='+repr(sources)+'\nfor name,source in sources.items():(work/name).write_text(source)\n'
    code+='(work/"incumbent.json").write_text('+repr(json.dumps(route))+')\n'
    code+='subprocess.run(["git","clone","--quiet","https://github.com/Sir-Nike/unoSDK.git",str(work/"unoSDK")],check=True)\nsubprocess.run(["git","checkout","--quiet","0cda3acc1bb5c05fa1803a8fbd7761770dc6d5fe"],cwd=work/"unoSDK",check=True)\nos.environ["UNO_SDK"]=str(work/"unoSDK")\nos.environ["OPENBLAS_NUM_THREADS"]="1"\nos.environ["OMP_NUM_THREADS"]="1"\n'
    if gpu:
        code+='data=next(pathlib.Path("/kaggle/input").rglob("counterfactual.npz"))\n'
        # Fit-only3epochs smoke does not touch any actual game selection seeds.
        code+='subprocess.run(["python",str(work/"train_uno_residual.py"),"--data",str(data),"--device","cuda","--epochs","3","--width","64","--objective","mse","--pair-gap","0","--uncertainty-only-weights","--calibrate-base","--fit-only","--output",str(work/"smoke-fit")],check=True,timeout=600)\n'
        code+='subprocess.run(["python",str(work/"compare_uno_objectives.py"),"--data",str(data),"--incumbent",str(work/"incumbent.json"),"--output","/kaggle/working/output","--workers","4"],check=True,timeout=10800)\n'
    else:
        code+='(work/"candidate.json").write_text('+repr(json.dumps(candidate))+')\n'
        code+='subprocess.run(["python",str(work/"evaluate_uno_checkpoint.py"),"--candidate",str(work/"candidate.json"),"--incumbent",str(work/"incumbent.json"),"--source-incumbent",str(work/"incumbent.json"),"--blocks","2048","--seed-start","57000000","--workers","4","--output","/kaggle/working/output/evaluation.json"],check=True,timeout=10800)\n'
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[{'cell_type':'markdown','metadata':{},'source':header},{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':code}]}
    filename='uno-'+slug+'.ipynb';(directory/filename).write_text(json.dumps(nb,indent=1)+'\n')
    meta={'id':'xdantalianx/triad-uno-'+slug,'title':title,'code_file':filename,'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':gpu,'enable_tpu':False,'enable_internet':True,'dataset_sources':[],'competition_sources':[],'kernel_sources':['xdantalianx/triad-uno-route-counterfactual'] if gpu else []}
    if gpu:meta['machine_shape']='NvidiaTeslaT4'
    (directory/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    compile(code,str(directory/filename),'exec')
    print(directory)
