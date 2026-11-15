"""Dependency-time zero-duration probe of temporary copies of both conditions."""
import json
import tempfile
from pathlib import Path
import yaml
from .common import ROOT, sha, seeds
from .energy import validate_policy, evaluate
from .thesis_figures import configure_font


def check_model(engine, policy_path, report_directory):
    from main.config import load, path
    from navigation.common.cg_signal import parameters
    policy=yaml.safe_load(path(policy_path).read_text());validate_policy(policy)
    c,landing,_,geometry=load(policy['base_config'])
    before=sha(path(c['model']));reports=[]
    engine.addpath(str(ROOT/'simulation/matlab'),nargout=0)
    engine.setup_paths(str(ROOT),nargout=0)
    # Keep generated caches on the data disk and remove after closing the copy.
    parent=Path(report_directory);parent.mkdir(parents=True,exist_ok=True)
    for sea in (5,6):
        with tempfile.TemporaryDirectory(prefix='model-probe-',dir=parent) as work:
            cfg=dict(initialize_callback='landing_live_initialize',model=str(path(c['model'])),
                geometry=geometry,gimbal=landing.get('gimbal',{}),inputs=c['inputs'],outputs=c['outputs'],
                workdir=work,plant_step=c['plant_step'],seed=seeds(policy['master_seed'],'clear',0,0)['sensor_seed'],
                setup_script=str(path(c['setup_script'])) if c.get('setup_script') else '',
                evaluation_mode=c['evaluation_mode'],input_contract=c['input_contract'],
                autonomous_subsystem=c['autonomous_subsystem'],ship_velocity_frame=c['ship_velocity_frame'],
                cg_signal_parameters=parameters(c.get('cg_signal')),control_interface=c['control_interface'],
                control_invalid_policy=c['control_invalid_policy'],
                batch=dict(method='C3',sea=sea,analysis_profile=policy['analysis_profile'],
                    **policy['seas'][sea],**seeds(policy['master_seed'],'clear',0,0)))
            cache=str(Path(work)/'cache').replace("'","''")
            engine.eval("Simulink.fileGenControl('set','CacheFolder','"+cache+"','CodeGenFolder','"+cache+"/codegen','createDir',true);",nargout=0)
            try:
                engine.initialize_external_model(json.dumps(cfg),nargout=1)
                status=json.loads(engine.read_landing_live_status(nargout=1))
                telemetry=status['batch_telemetry'];ei=evaluate(telemetry,geometry['H_ship_body'])
                if ei['value'] is None:raise RuntimeError('Actual EI telemetry missing: '+str(ei))
                if 'touchdown_stop' not in telemetry:raise RuntimeError('Original STOP input tap missing')
                info=json.loads((Path(work)/'batch_model.json').read_text())
                reports.append(dict(sea=sea,model=info,initial_EI=ei,initial_stop=telemetry['touchdown_stop']))
            finally:
                engine.finalize_external_model(nargout=1)
    unchanged=sha(path(c['model']))==before
    if not unchanged:raise RuntimeError('Source model changed during read-only batch probe')
    return dict(profile=policy['analysis_profile'],model=str(path(c['model'])),model_sha256=before,
                source_unchanged=unchanged,font=configure_font(),conditions=reports,
                note='Zero-duration compile and actual telemetry check; six-worker pressure run still required')
