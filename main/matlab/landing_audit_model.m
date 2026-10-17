function result=landing_audit_model(requestJSON)
c=jsondecode(requestJSON); [folder,model]=fileparts(c.model); addpath(folder);
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
result=jsonencode(struct('model',c.model,'inputs',{inames},'outputs',{onames}, ...
    'system_classes',{classes},'class_sources',sources,'missing_classes',{unique(missing)}, ...
    'root_port_contract_ready',isequal(sort(inames(:)),sort(fieldnames(c.inputs))) && ...
        all(ismember({c.outputs.helicopter_state,c.outputs.ship_state},onames)), ...
    'matlab',version,'platform',computer,'simulink_license',license('test','Simulink')));
close_system(model,0);
end
