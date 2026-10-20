function source=landing_native_port(model,name)
% Resolve a root Outport or an Outport in an immediate child subsystem.
p=[model '/' char(name)];
assert(getSimulinkBlockHandle(p)>0 && strcmp(get_param(p,'BlockType'),'Outport'), ...
    'landing:Ports','Missing native-state Outport %s',p);
parent=get_param(p,'Parent');
if strcmp(parent,model)
    ports=get_param(p,'PortHandles'); line=get_param(ports.Inport,'Line');
    assert(line~=-1,'landing:Ports','Unconnected native output %s',p);
    source=get_param(line,'SrcPortHandle');
else
    assert(strcmp(get_param(parent,'Parent'),model),'landing:Ports', ...
        'Native outputs must be at root or one subsystem deep: %s',p);
    ports=get_param(parent,'PortHandles');
    source=ports.Outport(str2double(get_param(p,'Port')));
end
end
