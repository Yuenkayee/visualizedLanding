"""Fault injection: process replacement, seed preservation and frozen-source audit."""
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
from types import SimpleNamespace

import pytest

from main.batch.common import ROOT, read_json, write_json
from main.batch.__main__ import parser, round_tasks, supervise
from main.batch.recovery import BatchWorkersFailed, RetryLedger, renderer_failure, resume_sources
from main.sensors import Sensors


GPU_ERROR = "RuntimeError: Error: Illegal address in CUDA queue copy_from_device (integrator_shade_volume integrator_queued_paths_array)"


def rpc_error(message):
    sensor = Sensors.__new__(Sensors)
    sensor.c = {"sensor_timeout_s": 0}
    sensor.worker = SimpleNamespace(stdin=io.StringIO())
    sensor.messages = queue.Queue()
    if message is not None:
        sensor.messages.put(dict(error=message))
    try:
        sensor._request({})
    except Exception as exc:
        return exc
    raise AssertionError("Expected injected fault")


@pytest.mark.parametrize("message,category", [
    (GPU_ERROR, "renderer_gpu_illegal_address"),
    ("CUDA_ERROR_ILLEGAL_ADDRESS", "renderer_gpu_illegal_address"),
    ("CUDA error: out of memory", "renderer_gpu_out_of_memory"),
    (None, "renderer_timeout"),
    ("NameError: unknown scene variable", None),
    ("Renderer exited; inspect renderer.log", None),
])
def test_only_recognized_renderer_rpc_errors_are_retryable(message, category):
    assert renderer_failure(rpc_error(message)) == category


def test_cuda_error_elsewhere_and_cleanup_failure_and_stop_are_not_retried():
    assert renderer_failure(RuntimeError(GPU_ERROR)) is None
    assert renderer_failure(KeyboardInterrupt("batch worker received signal 15")) is None
    try:
        try:
            raise rpc_error(GPU_ERROR)
        finally:
            raise RuntimeError("MATLAB cleanup failed")
    except RuntimeError as exc:
        assert renderer_failure(exc) is None


# Execute the real worker main/run_case and supervisor. Only the external plant,
# renderer response and scientific summarizer are replaced; use real processes,
# task claims, archives, exception handling, reservations and session cleanup.
WORKER_FIXTURE = r'''
import io, os, queue, sys, time, types
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1))
from main.batch.common import read_json, write_json
import main.batch.worker as worker
import main.batch.metrics as metrics
import main.config as config
from main.sensors import Sensors
out=Path(sys.argv[1]); index=int(sys.argv[2]); mode=os.environ['TEST_RECOVERY_MODE']
class Engine:
    def feature(self,*args,**kwargs): return os.getpid()
    def finalize_external_model(self,**kwargs): pass
class Session:
    engine=Engine()
    def __enter__(self): return self
    def __exit__(self,*exc):
        (out/f'closed_{os.getpid()}').touch()
        if mode=='cleanup': raise RuntimeError('MATLAB cleanup failed')
    def call(self,*args): return None
sys.modules['simulation.bridge.matlab_session']=types.SimpleNamespace(MatlabSession=Session)
config.load=lambda p: (read_json(p),None,None,None)
def run(c,landing,cal,geometry,case,matlab_session):
    with matlab_session:
        log=out/f"calls_{case.name}.json"
        calls=read_json(log) if log.exists() else []
        calls.append(dict(pid=os.getpid(),config=c))
        write_json(log,calls)
        write_json(case/'run.json',dict(config=c))
        if c['batch']['sea']==5:
            if mode=='code': raise ValueError('invalid geometry')
            if mode=='stop': raise KeyboardInterrupt('batch worker received signal 15')
            if len(calls)<=int(os.environ['TEST_FAILURES']):
                sensor=Sensors.__new__(Sensors)
                sensor.c={}; sensor.worker=types.SimpleNamespace(stdin=io.StringIO())
                sensor.messages=queue.Queue()
                sensor.messages.put(dict(error='Illegal address in CUDA queue copy_from_device'))
                sensor._request({})
        else:
            time.sleep(3)  # Must survive another worker's failure/restart.
sys.modules['main.run']=types.SimpleNamespace(run=run)
metrics.summarize=lambda case,p: dict(termination='timeout', config=read_json(case/'requested.json'),pid=os.getpid())
worker.main()
'''


@pytest.mark.parametrize("failures,limit,mode,success,retries", [
    (1, 2, "gpu", True, 1),
    (9, 2, "gpu", False, 2),
    (1, 0, "gpu", False, 0),
    (1, 2, "code", False, 0),
    (1, 2, "cleanup", False, 0),
    (1, 2, "stop", False, 0),
])
def test_real_worker_recovery_preserves_other_process_and_seeds(
        tmp_path, monkeypatch, failures, limit, mode, success, retries):
    script=tmp_path/'fixture.py';script.write_text(WORKER_FIXTURE)
    policy=dict(master_seed=5232026,max_replacements=5,methods=['C3'],wall_timeout_s=60,
        seas={'5':dict(wind_mps=10.,wave_height_m=2.),'6':dict(wind_mps=12.,wave_height_m=2.5)},
        runaway={},matlab_threads=1)
    manifest=dict(policy=policy,base_config={},groups=[dict(id='clear_s00',weather='clear',slot=0)])
    write_json(tmp_path/'manifest.json',manifest)
    tasks=round_tasks(tmp_path,manifest)
    args=parser().parse_args(['run','--output','outputs/batch/test','--workers','2',
        '--max-infra-retries',str(limit),'--retry-delay-s','0'])
    real_popen=subprocess.Popen
    launched=[]
    def popen(argv,**kwargs):
        if 'main.batch.worker' in argv:
            argv=[sys.executable,str(script),str(ROOT),*argv[-2:]]
            kwargs['env'].update(TEST_RECOVERY_MODE=mode,TEST_FAILURES=str(failures))
            proc=real_popen(argv,**kwargs);launched.append(proc)
            return proc
        return real_popen(argv,**kwargs)
    monkeypatch.setattr(subprocess,'Popen',popen)
    if success:
        supervise(tmp_path,manifest,args)
    else:
        with pytest.raises(BatchWorkersFailed):supervise(tmp_path,manifest,args)
    assert len(launched)==2+retries
    healthy=read_json(tmp_path/tasks[1]['case']/'metrics.json')
    healthy_calls=read_json(tmp_path/'calls_sea6_C3.json')
    assert len(healthy_calls)==1 and healthy['pid']==healthy_calls[0]['pid']
    calls=read_json(tmp_path/'calls_sea5_C3.json')
    assert len(calls)==1+retries
    assert len({c['pid'] for c in calls})==len(calls)
    assert all(c['config']==calls[0]['config'] for c in calls)
    assert calls[0]['config']['batch']['replacement']==0
    assert all((tmp_path/f"closed_{c['pid']}").exists() for c in calls)
    assert not list((tmp_path/'groups').rglob('claim'))
    case=tmp_path/tasks[0]['case']
    assert (case/'metrics.json').exists()==success
    archives=list(case.parent.glob('sea5_C3_interrupted_*'))
    assert len(archives)==retries
    assert all(read_json(a/'requested.json')==calls[0]['config'] for a in archives)
    if mode!='stop':
        entry=read_json(tmp_path/'infrastructure_retries.json')['cases'][tasks[0]['case']]
        assert entry['retries_started']==retries
        assert entry['status']==('completed' if success else 'failed')
    if success:
        assert round_tasks(tmp_path,manifest)==[]
        assert read_json(tmp_path/'groups/clear_s00/result.json')['attempt']==0
    else:
        assert read_json(tmp_path/'batch_failure.json')['failures']
        pending=round_tasks(tmp_path,manifest)
        assert len(pending)==1 and pending[0]['case']==tasks[0]['case']


def test_retry_budget_survives_resume(tmp_path):
    a=RetryLedger(tmp_path)
    assert a.failure('groups/case',0,123,'GPU error','renderer_gpu_illegal_address',2)==(True,1)
    b=RetryLedger(tmp_path)
    assert b.failure('groups/case',1,124,'GPU error','renderer_gpu_illegal_address',2)==(True,2)
    c=RetryLedger(tmp_path)
    assert c.failure('groups/case',2,125,'GPU error','renderer_gpu_illegal_address',2)==(False,2)


def test_retry_upgrade_is_explicit_preserves_manifest_and_freezes_new_sources(tmp_path):
    old={'main/batch/worker.py':'old','main/run.py':'science'}
    manifest=dict(source_sha256=old)
    write_json(tmp_path/'manifest.json',manifest)
    original=(tmp_path/'manifest.json').read_bytes()
    current={**old,'main/batch/worker.py':'new','main/batch/recovery.py':'added'}
    with pytest.raises(RuntimeError,match='Source/model'):
        resume_sources(tmp_path,manifest,current)
    assert resume_sources(tmp_path,manifest,current,True)==current
    assert (tmp_path/'manifest.json').read_bytes()==original
    assert resume_sources(tmp_path,manifest,current)==current
    record=read_json(tmp_path/'retry_source_upgrade.json')
    assert record['original_source_sha256']==old
    with pytest.raises(RuntimeError,match='since retry upgrade'):
        resume_sources(tmp_path,manifest,{**current,'main/batch/worker.py':'another'},True)
    write_json(tmp_path/'manifest.json',dict(manifest,altered=True))
    with pytest.raises(RuntimeError,match='Original manifest changed'):
        resume_sources(tmp_path,manifest,current)


@pytest.mark.parametrize('path', ['main/run.py','navigation/pipeline.py','main/batch_closed_loop.yaml',
    'model.slx','weights.pt','docs/论文.docx','main/sensors.py'])
def test_retry_upgrade_cannot_bypass_scientific_hashes(tmp_path,path):
    old={'main/batch/worker.py':'old',path:'frozen'}
    manifest=dict(source_sha256=old);write_json(tmp_path/'manifest.json',manifest)
    with pytest.raises(RuntimeError,match='Retry-only upgrade rejected'):
        resume_sources(tmp_path,manifest,{**old,path:'changed'},True)
    assert not (tmp_path/'retry_source_upgrade.json').exists()


def test_retry_upgrade_rejects_source_removal(tmp_path):
    manifest=dict(source_sha256={'main/batch/worker.py':'old'})
    write_json(tmp_path/'manifest.json',manifest)
    with pytest.raises(RuntimeError,match='removed'):
        resume_sources(tmp_path,manifest,{},True)


@pytest.mark.parametrize('flags', [['--max-infra-retries','-1'],['--retry-delay-s','nan'],['--retry-delay-s','-1']])
def test_invalid_retry_settings_rejected_before_startup(tmp_path,flags):
    args=parser().parse_args(['run','--output','outputs/batch/test',*flags])
    with pytest.raises(ValueError,match='nonnegative'):supervise(tmp_path,{},args)
