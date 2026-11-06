function info=landing_batch_configure(model,c)
% Configure only the disposable execution copy. Never save the source model.
b=c.batch; base=[model '/' c.autonomous_subsystem];
waveParents={[base '/LAND_CMD/target_pos/Ship_motion'], ...
    [base '/LAND_CMD/target_pos/PredictModel']};
wavePaths=cell(1,2);
for k=1:2
    found=find_system(waveParents{k},'LookUnderMasks','all','FollowLinks','on', ...
        'BlockType','MATLABSystem','System','tauAndFDI');
    assert(numel(found)==1,'landing:Batch','Expected one tauAndFDI in %s',waveParents{k});
    wavePaths{k}=found{1};
end
fixed=struct('PeakPeriod',11.2,'RelativeWaveDirectionDeg',150, ...
    'NumberOfWaveComponents',64,'MinimumWaveFrequency',.2,'MaximumWaveFrequency',2);
wave=struct('SignificantWaveHeight',b.wave_height_m,'RandomSeed',b.wave_seed);
fields=fieldnames(fixed);
for k=1:2
    p=wavePaths{k}; assert(getSimulinkBlockHandle(p)>0,'landing:Batch','Missing %s',p);
    for j=1:numel(fields)
        name=fields{j}; actual=slResolve(get_param(p,name),p);
        assert(abs(actual-fixed.(name))<1e-10,'landing:Batch', ...
            'Frozen wave parameter %s differs from thesis: %s',name,p);
    end
    set_param(p,'SignificantWaveHeight',num2str(wave.SignificantWaveHeight,17), ...
        'RandomSeed',num2str(wave.RandomSeed,17));
    assert(slResolve(get_param(p,'SignificantWaveHeight'),p)==b.wave_height_m && ...
        slResolve(get_param(p,'RandomSeed'),p)==b.wave_seed,'landing:Batch','Wave parameter write failed');
end
% Names differ between model revisions; identify the System class, not a guessed label.
wake=[base '/HELI_CONT/UWV_CONT/Inner system/CETI'];
ceti=find_system(wake,'LookUnderMasks','all','FollowLinks','on','BlockType','MATLABSystem','System','CETI');
assert(numel(ceti)==1,'landing:Batch','Expected one CETI System block');
set_param(ceti{1},'Uinf',num2str(b.wind_mps,17));
noise=find_system(wake,'SearchDepth',1,'LookUnderMasks','all','FollowLinks','off', ...
    'Name',sprintf('Band-Limited\nWhite Noise'));
if isempty(noise)
    candidate=[wake '/noise'];
    if getSimulinkBlockHandle(candidate)>0, noise={candidate}; end
end
assert(numel(noise)==1,'landing:Batch','Expected one wake noise block');
names=fieldnames(get_param(noise{1},'DialogParameters'));
key=names(strcmpi(names,'seed')); assert(numel(key)==1,'landing:Batch','Noise seed parameter missing');
set_param(noise{1},key{1},num2str(b.wind_seed,17));
assert(slResolve(get_param(ceti{1},'Uinf'),ceti{1})==b.wind_mps && ...
    slResolve(get_param(noise{1},key{1}),noise{1})==b.wind_seed,'landing:Batch','Wake parameter write failed');

obj=tauAndFDI;
props=properties(obj); values=struct();
for j=1:numel(props)
    try
        value=slResolve(get_param(wavePaths{1},props{j}),wavePaths{1});
        other=slResolve(get_param(wavePaths{2},props{j}),wavePaths{2});
        assert(isequal(value,other),'landing:Batch','Wave module property mismatch: %s',props{j});
        obj.(props{j})=value; values.(props{j})=value;
    catch err
        error('landing:Batch','Cannot freeze wave parameter %s: %s',props{j},err.message);
    end
end
components=obj.getWaveComponent();
cmd=[base '/LAND_CMD']; controller=[cmd '/MATLAB System'];
scales=zeros(3,1); keys={'maxDeltaVel_n','maxDeltaVel_e','maxDeltaVel_d'};
for j=1:3, scales(j)=slResolve(get_param(controller,keys{j}),controller); end
assert(all(scales>0),'landing:Batch','Invalid guidance limits');
% The existing atan law has an asymptotic bound, not a hard clipping block.
info=struct('wave_parameters',values,'wave_component',components, ...
    'wave_blocks',{wavePaths},'ceti_block',ceti{1},'noise_block',noise{1}, ...
    'wind_mps',b.wind_mps,'wind_seed',b.wind_seed, ...
    'command_limit_mps',scales*pi/2*(1852/3600), ...
    'saturation_definition','abs(relative command) >= 0.95 * atan asymptotic bound; not hard actuator saturation', ...
    'render_limitation','Visual ocean is procedural; same sea-state label, not the same stochastic wave field');
chart=sfroot; chart=chart.find('-isa','Stateflow.Chart','Path',[cmd '/target_pos/Chart']);
assert(numel(chart)==1,'landing:Batch','Missing guidance stage chart');
state=chart.find('-isa','Stateflow.State','Name','touchDown');
assert(numel(state)==1,'landing:Batch','Missing touchdown state');
target=regexp(state.LabelString,'target_pos\s*=\s*(\[[^\]]+\])','tokens','once');
assert(~isempty(target),'landing:Batch','Cannot read nominal target');
info.nominal_cg_target_ned_m=str2num(target{1}); %#ok<ST2NM>
assert(numel(info.nominal_cg_target_ned_m)==3,'landing:Batch','Invalid target');

% Disable bulky source logging only in the disposable copy.
sinks=find_system(model,'LookUnderMasks','all','FollowLinks','on','BlockType','ToWorkspace');
for j=1:numel(sinks), set_param(sinks{j},'MaxDataPoints','2'); end
fields={'stage','cg_used','control_valid','target','error','relative_command','command','feedforward','wave_command'};
sources={[cmd '/target_pos/Chart'],[model '/LandingControlCG'], ...
    [model '/LandingControlValid'],[cmd '/Plus'],[cmd '/Plus'], ...
    [cmd '/NavigationValidityGate'],[cmd '/Plus1'],[cmd '/Plus1'],wavePaths{1}};
ports=[2,1,1,2,1,1,1,2,4]; inputs=[false,false,false,true,false,false,false,true,true];
% Latches are created later by landing_prepare_control; use the source plant
% inputs for the structural probe and read the latch itself at runtime.
sources{2}=[base '/cg_relative_estimate']; sources{3}=[base '/cg_navigation_valid'];
% Resolve actual subsystem Inport names by number, independent of labels.
ip=find_system(base,'SearchDepth',1,'BlockType','Inport');
for j=1:numel(ip)
    number=str2double(get_param(ip{j},'Port'));
    if number==1, sources{2}=ip{j}; elseif number==2, sources{3}=ip{j}; end
end
info.taps=struct(); info.variables=struct();
for j=1:numel(fields)
    name=fields{j}; source=sources{j}; parent=get_param(source,'Parent');
    ph=get_param(source,'PortHandles');
    if inputs(j)
        line=get_param(ph.Inport(ports(j)),'Line'); phsrc=get_param(line,'SrcPortHandle');
    else
        phsrc=ph.Outport(ports(j));
    end
    tap=[parent '/LandingBatch_' name]; sink=[parent '/LandingBatchLog_' name];
    add_block('simulink/Math Operations/Gain',tap,'Gain','1');
    variable=['landing_batch_' name];
    add_block('simulink/Sinks/To Workspace',sink,'VariableName',variable, ...
        'SaveFormat','Timeseries','MaxDataPoints','2','SampleTime',num2str(c.plant_step,17));
    tp=get_param(tap,'PortHandles'); sp=get_param(sink,'PortHandles');
    add_line(parent,phsrc,tp.Inport,'autorouting','on'); add_line(parent,tp.Outport,sp.Inport,'autorouting','on');
    info.taps.(name)=tap; info.variables.(name)=variable;
end
file=fullfile(c.workdir,'batch_model.json'); fid=fopen(file,'w');
assert(fid~=-1,'landing:Batch','Cannot write model audit');
cleanup=onCleanup(@()fclose(fid)); fprintf(fid,'%s\n',jsonencode(info));
end
