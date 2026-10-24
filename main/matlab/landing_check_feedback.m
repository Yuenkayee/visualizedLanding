function report=landing_check_feedback(model,c)
% Strict contract for the user-supplied CG navigation feedback plant.
report=struct('enabled',false);
if ~isfield(c,'control_interface') || ~strcmp(c.control_interface,'cg_position'), return; end
assert(strcmp(c.evaluation_mode,'feedback') && strcmp(c.input_contract,'conversion'), ...
    'landing:Feedback','CG control requires feedback mode and conversion inputs');
assert(strcmp(c.control_invalid_policy,'zero_relative_velocity'), ...
    'landing:Feedback','Unsupported invalid-navigation policy');
plant=[model '/' c.autonomous_subsystem]; cmd=[plant '/LAND_CMD'];
conversion=[model '/Frame_Unit_ReferencePoint_Conversion'];
assert(strcmp(get_param(conversion,'System'),'convert'),'landing:Feedback','Expected convert');
p=get_param(conversion,'PortHandles'); q=get_param(plant,'PortHandles');
assert(numel(p.Outport)==23 && numel(q.Inport)==2,'landing:Feedback', ...
    ['CG feedback interface mismatch in %s: convert has %d outputs (expected 23); ' ...
     '%s has %d inputs (expected 2: CG position and cg_signal_valid). ' ...
     'Sync the updated SLX, or run bash shell/run_landing_closed_loop.sh connect-navigation ' ...
     '--config main/navigation_feedback.yaml (backs up the model before repair). convert class: %s'], ...
    get_param(model,'FileName'),numel(p.Outport),plant,numel(q.Inport),which('convert'));
rootInputs=find_system(model,'SearchDepth',1,'BlockType','Inport');
for k=1:numel(rootInputs)
    h=get_param(rootInputs{k},'PortHandles'); line=get_param(h.Outport,'Line');
    assert(line~=-1 && all(get_param(line,'DstBlockHandle')==getSimulinkBlockHandle(conversion)), ...
        'landing:Feedback','Navigation root inputs must feed conversion only');
end
assert_edge(conversion,20,plant,1); assert_edge(conversion,22,plant,2);
% Every other conversion output is evaluation only; true positions may not feed control.
for k=1:23
    line=get_param(p.Outport(k),'Line'); if line==-1, continue; end
    for dest=get_param(line,'DstBlockHandle')'
        if ismember(k,[20 22]) && dest==getSimulinkBlockHandle(plant), continue; end
        assert(ismember(get_param(dest,'BlockType'),{'Outport','Scope','Terminator','ToWorkspace'}), ...
            'landing:Feedback','Unexpected control path from conversion output %d',k);
    end
end
assert_edge([plant '/cg_relative_estimate'],1,cmd,2);
assert_edge([plant '/cg_navigation_valid'],1,cmd,3);
assert_edge([cmd '/target_pos'],1,[cmd '/Plus'],2);
signForm=landing_guidance_error_contract(cmd);
assert_edge([cmd '/cg_relative_estimate'],1,[cmd '/target_pos'],2);
assert_edge([cmd '/target_pos/current_pos'],1,[cmd '/target_pos/Selector'],1);
assert_edge([cmd '/target_pos/current_pos'],1,[cmd '/target_pos/Selector1'],1);
assert_edge([cmd '/target_pos/Selector'],1,[cmd '/target_pos/Chart'],2);
assert_edge([cmd '/target_pos/Selector1'],1,[cmd '/target_pos/Chart'],3);
assert_edge([cmd '/Plus'],1,[cmd '/MATLAB System'],1);
assert(strcmp(get_param([cmd '/MATLAB System'],'System'),'uvw_Gen'),'landing:Feedback','Unexpected guidance law');
gate=[cmd '/NavigationValidityGate'];
assert(strcmp(get_param(gate,'BlockType'),'Switch') && strcmp(get_param(gate,'Criteria'),'u2 >= Threshold') && ...
    abs(slResolve(get_param(gate,'Threshold'),gate)-.5)<1e-12,'landing:Feedback','Invalid validity gate');
assert_edge([cmd '/MATLAB System'],1,gate,1);
assert_edge([cmd '/cg_navigation_valid'],1,gate,2);
assert_edge([cmd '/cg_navigation_valid'],1,[cmd '/target_pos'],3);
assert_edge([cmd '/target_pos/nav_valid'],1,[cmd '/target_pos/Chart'],4);
assert_edge([cmd '/ZeroRelativeVelocity'],1,gate,3);
assert(all(slResolve(get_param([cmd '/ZeroRelativeVelocity'],'Value'),cmd)==0),'landing:Feedback','Invalid velocity fallback');
assert_edge(gate,1,[cmd '/Plus1'],1);
chart=sfroot; chart=chart.find('-isa','Stateflow.Chart','Path',[cmd '/target_pos/Chart']);
assert(numel(chart)==1,'landing:Feedback','Target phase chart missing');
transitions=chart.find('-isa','Stateflow.Transition');
for k=1:numel(transitions)
    assert(~contains(transitions(k).LabelString,'ship_pos_x'),'landing:Feedback','Absolute ship coordinate in relative phase guard');
    label=transitions(k).LabelString;
    if startsWith(label,'[')
        assert(contains(label,'nav_valid >= 0.5'),'landing:Feedback','Navigation validity missing from phase transition');
    end
end
report=struct('enabled',true,'position_source','convert/20 cg_relative_estimate_50hz_ned', ...
    'valid_source','convert/22 cg_signal_valid','sign','target CG relative - helicopter CG relative', ...
    'phase_position_source','navigation CG relative','invalid_policy',c.control_invalid_policy, ...
    'guidance_error_implementation',signForm, ...
    'host_control_latch',true,'truth_feedback',false, ...
    'conversion_output_count',numel(p.Outport),'plant_input_count',numel(q.Inport));
end
function assert_edge(source,out,destination,in)
p=get_param(source,'PortHandles'); q=get_param(destination,'PortHandles');
line=get_param(q.Inport(in),'Line');
assert(line~=-1 && get_param(line,'SrcPortHandle')==p.Outport(out), ...
    'landing:Feedback','Wrong source for %s/%d; expected %s/%d',destination,in,source,out);
end
