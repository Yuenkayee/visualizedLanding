function result=landing_connect_navigation(requestJSON)
% Repair the user's CG connection in place, without replacing controllers.
% Input/target are both heli CG minus ship CG, NED metres. uvw_Gen expects
% target MINUS current. Navigation-invalid cancels only relative velocity.
c=jsondecode(requestJSON); [~,model]=fileparts(c.model);
load_system(c.model);
assert(strcmp(get_param(model,'Dirty'),'off'),'landing:Unsaved','Save source model first');
assert(~isfile(c.backup),'landing:Backup','Backup already exists'); copyfile(c.model,c.backup);
plant=[model '/innerLoop']; cmd=[plant '/LAND_CMD']; conversion=[model '/Frame_Unit_ReferencePoint_Conversion'];
% Replace local root Goto/From with explicit edges, retaining all output logs.
for name={'From','Goto'}
    block=[model '/' name{1}];
    if getSimulinkBlockHandle(block)>0 && strcmp(get_param(block,'GotoTag'),'cg_relative_ned')
        disconnect(block); delete_block(block);
    end
end
connect(model,[conversion '/20'],[plant '/1']);
add_input(plant,'cg_navigation_valid','2',[215 155 245 169]);
add_input(cmd,'cg_navigation_valid','3',[-30 240 0 254]);
connect(model,[conversion '/22'],[plant '/2']);
connect(plant,[plant '/cg_navigation_valid/1'],[cmd '/3']);
% The ship absolute position must not be added to a CG-relative navigation value.
set_param([cmd '/Plus'],'Inputs','-+');
connect(cmd,[cmd '/target_pos/1'],[cmd '/Plus/2']);
connect(cmd,[cmd '/cg_relative_estimate/1'],[cmd '/target_pos/2']);
% Phase transition now compares relative CG coordinates with relative targets.
machine=sfroot; chart=machine.find('-isa','Stateflow.Chart','Path',[cmd '/target_pos/Chart']);
assert(numel(chart)==1,'landing:Guidance','Expected target phase chart');
transitions=chart.find('-isa','Stateflow.Transition'); changed=false;
for k=1:numel(transitions)
    label=transitions(k).LabelString;
    if contains(label,'x - target_pos(1) - ship_pos_x')
        transitions(k).LabelString=strrep(label,'x - target_pos(1) - ship_pos_x','x - target_pos(1)');
        changed=true;
    elseif contains(label,'x - target_pos(1)')
        changed=true;
    end
end
assert(changed,'landing:Guidance','Expected target-relative phase transition');
data=chart.find('-isa','Stateflow.Data','Name','ship_pos_x');
if ~isempty(data), data.Name='nav_valid'; end
for k=1:numel(transitions)
    label=transitions(k).LabelString;
    if startsWith(label,'[') && ~contains(label,'nav_valid')
        transitions(k).LabelString=['[nav_valid >= 0.5 && (' label(2:end-1) ')]'];
    end
end
add_input([cmd '/target_pos'],'nav_valid','3',[35 450 65 464]);
connect(cmd,[cmd '/cg_navigation_valid/1'],[cmd '/target_pos/3']);
connect([cmd '/target_pos'],[cmd '/target_pos/nav_valid/1'],[cmd '/target_pos/Chart/4']);
% Remove the now-unused absolute target adder; retain Ship_trace/Ship_motion.
if getSimulinkBlockHandle([cmd '/Sum'])>0, disconnect([cmd '/Sum']); delete_block([cmd '/Sum']); end
gate=[cmd '/NavigationValidityGate']; zero=[cmd '/ZeroRelativeVelocity'];
if getSimulinkBlockHandle(gate)<0
    add_block('simulink/Signal Routing/Switch',gate,'Criteria','u2 >= Threshold', ...
        'Threshold','0.5','Position',[310 165 355 225]);
    add_block('simulink/Sources/Constant',zero,'Value','zeros(3,1)','Position',[230 270 285 295]);
end
connect(cmd,[cmd '/MATLAB System/1'],[gate '/1']);
connect(cmd,[cmd '/cg_navigation_valid/1'],[gate '/2']);
connect(cmd,[zero '/1'],[gate '/3']);
connect(cmd,[gate '/1'],[cmd '/Plus1/1']);
save_system(model); close_system(model,0);
result=jsonencode(struct('model',c.model,'backup',c.backup,'signal_port',20,'valid_port',22, ...
    'error','target_relative_cg_ned - estimated_relative_cg_ned', ...
    'invalid_policy','zero relative velocity; retain original ship velocity feedforward'));
end
function add_input(parent,name,port,position)
path=[parent '/' name];
if getSimulinkBlockHandle(path)<0
    add_block('simulink/Sources/In1',path,'Port',port,'PortDimensions','1', ...
        'OutDataTypeStr','double','Position',position);
end
end
function connect(parent,source,destination)
split=find(destination=='/',1,'last'); block=destination(1:split-1); index=str2double(destination(split+1:end));
ph=get_param(block,'PortHandles'); line=get_param(ph.Inport(index),'Line');
if line~=-1
    src=get_param(line,'SrcPortHandle');
    delete_line(parent,endpoint(src,parent),extractAfter(destination,[parent '/']));
end
add_line(parent,extractAfter(source,[parent '/']),extractAfter(destination,[parent '/']),'autorouting','on');
end
function disconnect(block)
p=get_param(block,'PortHandles');
parent=get_param(block,'Parent');
for h=p.Inport(:)'
    line=get_param(h,'Line');
    if line~=-1, delete_line(parent,endpoint(get_param(line,'SrcPortHandle'),parent),endpoint(h,parent)); end
end
for h=p.Outport(:)'
    line=get_param(h,'Line'); if line~=-1, delete_line(line); end
end
end
function value=endpoint(port,parent)
value=[extractAfter(get_param(port,'Parent'),[parent '/']) '/' num2str(get_param(port,'PortNumber'))];
end
