function result=landing_configure_geometry(requestJSON)
% Align training geometry and the fixed inner-loop period, keeping all wiring.
c=jsondecode(requestJSON); rows=cell(numel(c.models),1);
assert(isfield(c,'plant_step') && abs(c.plant_step-.02)<1e-12, ...
    'landing:Timing','configure requires plant_step=0.02');
for k=1:numel(c.models)
    source=char(c.models{k}); [~,model]=fileparts(source);
    load_system(source);
    assert(strcmp(get_param(model,'Dirty'),'off'),'landing:Unsaved','Save %s before configuring',model);
    backup=fullfile(c.backup_dir,[model '.slx']);
    assert(~isfile(backup),'landing:Backup','Backup already exists');
    copyfile(source,backup);
    block=[model '/Frame_Unit_ReferencePoint_Conversion'];
    assert(getSimulinkBlockHandle(block)>0,'landing:Block','Conversion block missing');
    ws=get_param(model,'ModelWorkspace');
    for field={'H_ship_body','gear_body','T_body_camera'}
        name=field{1}; value=c.geometry.(name);
        if numel(value)==3, value=value(:); end
        assignin(ws,name,value); set_param(block,name,name);
    end
    if isfield(c,'cg_signal_parameters')
        set_param(block,'CgSignalParameters',mat2str(c.cg_signal_parameters(:),17));
    end
    set_param(model,'SolverType','Fixed-step','FixedStep',num2str(c.plant_step,17));
    timing=landing_check_timing(model,c.plant_step);
    save_system(model); close_system(model,0);
    load_system(source); ws=get_param(model,'ModelWorkspace');
    landing_check_timing(model,c.plant_step);
    for field={'H_ship_body','gear_body','T_body_camera'}
        name=field{1}; actual=getVariable(ws,name); expected=c.geometry.(name);
        assert(norm(actual(:)-expected(:))<1e-12,'landing:Geometry','Saved geometry failed verification');
    end
    close_system(model,0);
    rows{k}=struct('model',source,'backup',backup,'geometry',c.geometry,'timing',timing);
end
result=jsonencode(rows);
end
