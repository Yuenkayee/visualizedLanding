function report=landing_check_observer(model,c)
% This observer contract requires an autonomous root subsystem. Navigation
% may only drive convert, whose outputs may only drive root outputs/sinks.
report=struct('enabled',false);
if ~isfield(c,'evaluation_mode') || ~strcmp(c.evaluation_mode,'observer'), return; end
assert(isfield(c,'input_contract') && strcmp(c.input_contract,'conversion'), ...
    'landing:Observer','Observer mode requires the conversion input contract');
plant=[model '/' char(c.autonomous_subsystem)];
if getSimulinkBlockHandle(plant)<=0
    candidates=find_system(model,'SearchDepth',1,'BlockType','SubSystem');
    available=strjoin(candidates,', ');
    if isempty(available), available='(none)'; end
    error('landing:Observer', ...
        ['Autonomous subsystem is missing: %s. Available root subsystems: %s. ' ...
         'Update autonomous_subsystem and outputs.helicopter_state/ship_state in the observer config.'], ...
        plant,available);
end
ports=get_param(plant,'PortHandles');
assert(isempty(ports.Inport),'landing:Observer','Observer plant must have no external inputs');
% Cross-boundary Goto/Data Store/function calls would bypass physical ports.
for type={'From','Goto','DataStoreRead','DataStoreWrite','FunctionCaller','ModelReference','S-Function'}
    blocks=find_system(model,'LookUnderMasks','all','FollowLinks','on','BlockType',type{1});
    if ismember(type{1},{'From','Goto','DataStoreRead','DataStoreWrite'})
        % Internal tags are fine if no matching mechanism exists outside.
        blocks=blocks(~startsWith(blocks,[plant '/']));
    end
    if strcmp(type{1},'S-Function')
        blocks=blocks(~cellfun(@(b)strcmp(get_param(b,'FunctionName'),'sf_sfun'),blocks));
    end
    assert(isempty(blocks),'landing:Observer', ...
        'Observer isolation requires review of %s blocks in %s',type{1},plant);
end
conversion=[model '/Frame_Unit_ReferencePoint_Conversion'];
assert(strcmp(get_param(conversion,'System'),'convert'),'landing:Observer','Expected convert block');
inputs=find_system(model,'SearchDepth',1,'BlockType','Inport');
for k=1:numel(inputs)
    p=get_param(inputs{k},'PortHandles'); destinations=get_param(get_param(p.Outport,'Line'),'DstBlockHandle');
    assert(all(destinations==getSimulinkBlockHandle(conversion)), ...
        'landing:Observer','Root inputs may only feed the conversion block');
end
p=get_param(conversion,'PortHandles');
expected=innerloop_output_fields(true);
assert(numel(p.Outport)==numel(expected),'landing:Observer', ...
    'Conversion CG ports are missing. Sync both SLX files or run upgrade_innerLoop_gimbal after saving a backup.');
for k=16:numel(expected)
    wire=get_param(p.Outport(k),'Line');
    assert(wire~=-1,'landing:Observer','CG output %s is disconnected',expected{k});
    destinations=get_param(wire,'DstBlockHandle');
    names=arrayfun(@(h)get_param(h,'Name'),destinations,'UniformOutput',false);
    assert(any(strcmp(names,expected{k})),'landing:Observer','CG output %s is miswired',expected{k});
end
for port=p.Outport(:)'
    line=get_param(port,'Line'); if line==-1, continue; end
    destinations=get_param(line,'DstBlockHandle');
    for dest=destinations(:)'
        assert(ismember(get_param(dest,'BlockType'),{'Outport','Scope','Terminator','ToWorkspace'}), ...
            'landing:Observer','Conversion output feeds a non-observer block');
    end
end
report=struct('enabled',true,'verified_no_navigation_control_path',true, ...
    'autonomous_subsystem',plant,'cg_signals_available',true, ...
    'cg_signal_parameters',get_param(conversion,'CgSignalParameters'));
end
