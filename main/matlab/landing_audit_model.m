function result=landing_audit_model(requestJSON)
c=jsondecode(requestJSON); [folder,model]=fileparts(c.model); addpath(folder);
if isfield(c,'setup_script') && ~isempty(c.setup_script)
    evalin('base',['run(''' strrep(char(c.setup_script),'''','''''') ''')']);
end
load_system(c.model);
inputs=find_system(model,'SearchDepth',1,'BlockType','Inport');
outputs=find_system(model,'SearchDepth',1,'BlockType','Outport');
systems=find_system(model,'LookUnderMasks','all','FollowLinks','on','BlockType','MATLABSystem');
classes=unique(cellfun(@(b)get_param(b,'System'),systems,'UniformOutput',false));
missing={}; sources=struct();
for k=1:numel(classes)
    sources.(classes{k})=which(classes{k});
    if exist(classes{k},'class')~=8 && exist(classes{k},'file')==0
        missing{end+1}=classes{k}; %#ok<AGROW>
    end
end
inames=cellfun(@(b)get_param(b,'Name'),inputs,'UniformOutput',false);
onames=cellfun(@(b)get_param(b,'Name'),outputs,'UniformOutput',false);
try
    timing=landing_check_timing(model,.02); timing.valid=true;
catch err
    timing=struct('valid',false,'error',err.message);
end
nativeReady=true; nativeError='';
try
    landing_native_port(model,c.outputs.helicopter_state);
    landing_native_port(model,c.outputs.ship_state);
catch err
    nativeReady=false; nativeError=err.message;
end
try
    observer=landing_check_observer(model,c); observer.valid=true;
catch err
    observer=struct('valid',false,'error',err.message);
end
stops=find_system(model,'LookUnderMasks','all','FollowLinks','on','BlockType','Stop');
try
    feedback=landing_check_feedback(model,c); feedback.valid=true;
catch err
    feedback=struct('valid',false,'error',err.message);
end
result=jsonencode(struct('model',c.model,'inputs',{inames},'outputs',{onames}, ...
    'timing',timing,'solver_type',get_param(model,'SolverType'), ...
    'fixed_step_expression',get_param(model,'FixedStep'), ...
    'system_classes',{classes},'class_sources',sources,'missing_classes',{unique(missing)}, ...
    'native_state_ready',nativeReady,'native_state_error',nativeError, ...
    'observer',observer,'feedback',feedback,'stop_blocks',{stops}, ...
    'root_port_contract_ready',isequal(sort(inames(:)),sort(fieldnames(c.inputs))) && ...
        nativeReady, ...
    'matlab',version,'platform',computer,'simulink_license',license('test','Simulink')));
close_system(model,0);
end
