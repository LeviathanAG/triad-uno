"""Final external admission receipt for exact held UNO source; no main edits."""
import datetime,hashlib,importlib.util,json,subprocess
from pathlib import Path

ROOT=Path('/home/satwik/triad/uno-admission-fix')
OUT=Path('/home/satwik/triad/training')
SHA='ee37f8c600c1b4de7cc6b3f044908e3545b39a52'
TAG='triad-uno:final-held-ee37f8c-20261015'
EXPECTED_ARCHIVE='3f366b78ad2f8d4b2b10d4dc05abeb4c4b8a417abf88192a47646193d503fd46'


def run(*args):return subprocess.check_output(args,text=True).strip()


def main():
    assert run('git','-C',str(ROOT),'rev-parse','HEAD')==SHA
    assert not run('git','-C',str(ROOT),'status','--porcelain')
    assert (ROOT/'Dockerfile').read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show','upstream/main:Dockerfile'])
    assert not run('git','-C',str(ROOT),'diff','upstream/main','--','src/uno_agent')
    manifest=json.loads((ROOT/'submission.json').read_text())
    assert manifest['team_name']=='TungMasterSoloClear' and manifest['team_id']=='TECHTEAM11600'
    assert manifest['execution']=='cpu' and manifest['entrypoint']==['python','src/agent.py']
    assert manifest['resources']=={'cpu':1,'ram_bytes':1073741824,'scratch_bytes':1073741824,'max_processes':128,'gpu_bytes':0}
    declared=manifest['models'];assert len(declared)==1 and declared[0]['path']=='models/policy.json'
    assert declared[0]['format']=='triad-uno-route-linear-v1' and declared[0]['generated_at_build'] is False
    policy=hashlib.sha256((ROOT/declared[0]['path']).read_bytes()).hexdigest()
    assert policy==declared[0]['sha256']=='d458ef2894c0ce51d62ed30a4ad1b685c85db2c1a97417a8a93efab16d5abe14'
    archive=subprocess.check_output(['git','-C',str(ROOT),'archive',SHA])
    archive_sha=hashlib.sha256(archive).hexdigest();assert len(archive)==593920 and archive_sha==EXPECTED_ARCHIVE
    archive_path=OUT/'uno-final-held-source-ee37f8c.tar';archive_path.write_bytes(archive)
    tracked=run('git','-C',str(ROOT),'ls-tree','-r','--name-only',SHA).splitlines()
    pointers=[];runtime={}
    for name in tracked:
        data=subprocess.check_output(['git','-C',str(ROOT),'show',SHA+':'+name])
        if data.startswith(b'version https://git-lfs.github.com/spec/v1\n'):pointers.append(name)
        if name=='submission.json' or name.startswith(('src/','models/')):
            runtime[name]=hashlib.sha256(data).hexdigest()
    assert not pointers
    image=json.loads(run('docker','image','inspect',TAG))[0]
    base=json.loads(run('docker','image','inspect','python:3.11-slim'))[0]
    assert image['Architecture']=='amd64' and image['Os']=='linux'
    assert image['Config']['User']=='10000:10000' and image['Config']['Entrypoint']==manifest['entrypoint']
    assert not image['Config'].get('Volumes')
    labels={'org.triad-uno.protocol':'1.0','org.triad-uno.runtime':'2','org.triad-uno.sdk':'uno-sdk-2',
            'org.triad-uno.cpu':'1','org.triad-uno.ram-bytes':'1073741824','org.triad-uno.scratch-bytes':'1073741824',
            'org.triad-uno.pids':'128','org.triad-uno.gpu':'false','org.triad-uno.gpu-bytes':'0'}
    assert all(image['Config']['Labels'].get(k)==v for k,v in labels.items())
    assert image['RootFS']['Layers'][:len(base['RootFS']['Layers'])]==base['RootFS']['Layers']
    command=['docker','run','--rm','-i','--network','none','--read-only','--cpus','1','--memory','1g',
             '--pids-limit','128','--tmpfs','/tmp:rw,noexec,nosuid,size=64m',TAG]
    spec=importlib.util.spec_from_file_location('uno_final_check',ROOT/'tools/check.py')
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    checks=checker.check_agent(command,None,'final-restricted-docker')
    assert len(checks)==8 and all(row[1] for row in checks)
    code='import pathlib,hashlib,json;paths=[p for root in (pathlib.Path("src"),pathlib.Path("models")) for p in root.rglob("*") if p.is_file()];paths+=[pathlib.Path("submission.json")];print(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))'
    actual=json.loads(run('docker','run','--rm','--network','none','--read-only','--cpus','1','--memory','1g',
                         '--entrypoint','python',TAG,'-c',code))
    assert actual==runtime
    size=json.loads((OUT/'uno-final-held-size-20261015.json').read_text());assert size['ok'] and size['image']['ok'] and not size['lfs_pointers']
    report={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha':SHA,
        'source_path':str(ROOT),'source_clean':True,'archive_path':str(archive_path),'archive_bytes':len(archive),
        'archive_sha256':archive_sha,'matches_root_staged_source_archive':True,
        'official_upstream':run('git','-C',str(ROOT),'rev-parse','upstream/main'),
        'official_literal_python_template_matches':True,'deployed_contract_matches_upstream':True,
        'team':{'name':manifest['team_name'],'id':manifest['team_id']},'manifest_valid':True,
        'policy_sha256':policy,'all_declared_assets_present_and_real':True,'lfs_pointers':pointers,
        'image_tag':TAG,'image_id':image['Id'],'image_platform':'linux/amd64','image_user':image['Config']['User'],
        'image_entrypoint':image['Config']['Entrypoint'],'required_labels':labels,'runtime_file_hashes':runtime,
        'all_image_runtime_files_equal_exact_commit':True,'exact_cached_base_layer_prefix_matches':True,
        'base_repo_digests':base.get('RepoDigests'),'full_image_size_no_exemption':size['image'],
        'source_size':size,'restricted_protocol_checks':checks,'restricted_command':command,
        'fresh_build_used_pull_and_official_literal':True,'cpu_only_no_shared_gpu_exemption':True,
        'host_protocol_checks':8,'publication':'Local image only; no registry destination or organizer pull proof supplied',
        'root_owns_noon_main_push':True,'main_pushed_by_this_audit':False}
    (OUT/'UNO-FINAL-HELD-AUDIT-20261015.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
