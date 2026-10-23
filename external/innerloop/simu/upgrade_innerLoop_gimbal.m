function upgrade_innerLoop_gimbal(names)
% Append gimbal/CG ports in-place; preserve guidance/controller subsystems.
if nargin<1, names={'innerLoop','innerLoop_landing'}; end
folder=fileparts(mfilename('fullpath')); addpath(folder);
inputs={'gimbal_angles_rad',3;'nav_R_body_camera','[3 3]'; ...
    'nav_gimbal_omega_camera',3;'nav_gimbal_covariance','[3 3]'; ...
    'nav_gimbal_valid',1;'nav_gimbal_timestamp',1;'nav_angular_velocity_camera',3};
allOutputs=innerloop_output_fields(true); outputs=allOutputs(13:end);
for j=1:numel(names)
    name=names{j}; load_system(fullfile(folder,[name '.slx']));
    block=[name '/Frame_Unit_ReferencePoint_Conversion'];
    % Serialized MATLAB System ports retain the old class signature on load.
    % Recreate only this boundary block, restoring every surrounding wire.
    old=get_param(block,'PortHandles'); sources=cell(size(old.Inport)); destinations=cell(size(old.Outport));
    for k=1:numel(old.Inport)
        wire=get_param(old.Inport(k),'Line');
        if wire~=-1, sources{k}=endpoint(get_param(wire,'SrcPortHandle'),name); end
    end
    for k=1:numel(old.Outport)
        wire=get_param(old.Outport(k),'Line');
        if wire~=-1
            handles=get_param(wire,'DstPortHandle');
            destinations{k}=arrayfun(@(h)endpoint(h,name),handles,'UniformOutput',false);
        end
    end
    parameters={get_param(block,'T_body_camera'),get_param(block,'H_ship_body'),get_param(block,'gear_body')};
    position=get_param(block,'Position');
    signalParameters='[0.04;5;0.30;1]';
    try, signalParameters=get_param(block,'CgSignalParameters'); catch, end
    % Delete branches explicitly before removing the block; do not retain port
    % handles, because Simulink can recycle them when constructing the new block.
    for k=1:numel(sources)
        if ~isempty(sources{k})
            delete_line(name,sources{k},['Frame_Unit_ReferencePoint_Conversion/' num2str(k)]);
        end
    end
    for k=1:numel(old.Outport)
        wire=get_param(old.Outport(k),'Line');
        if wire~=-1, delete_line(wire); end
    end
    delete_block(block);
    add_block('simulink/User-Defined Functions/MATLAB System',block, ...
        'System','convert','SimulateUsing','Interpreted execution', ...
        'T_body_camera',parameters{1},'H_ship_body',parameters{2},'gear_body',parameters{3}, ...
        'CgSignalParameters',signalParameters,'Position',position);
    for k=1:numel(sources)
        if ~isempty(sources{k})
            add_line(name,sources{k},['Frame_Unit_ReferencePoint_Conversion/' num2str(k)],'autorouting','on');
        end
    end
    for k=1:numel(destinations)
        for d=1:numel(destinations{k})
            add_line(name,['Frame_Unit_ReferencePoint_Conversion/' num2str(k)],destinations{k}{d},'autorouting','on');
        end
    end
    for k=1:size(inputs,1)
        path=[name '/' inputs{k,1}];
        if getSimulinkBlockHandle(path)<0
            dims=inputs{k,2}; if isnumeric(dims), dims=num2str(dims); end
            add_block('simulink/Sources/In1',path,'Port',num2str(k+8), ...
                'PortDimensions',dims,'OutDataTypeStr','double', ...
                'Position',[40 520+k*55 70 534+k*55]);
            add_line(name,[inputs{k,1} '/1'], ...
                ['Frame_Unit_ReferencePoint_Conversion/' num2str(k+8)],'autorouting','on');
        end
    end
    for k=1:numel(outputs)
        path=[name '/' outputs{k}];
        if getSimulinkBlockHandle(path)<0
            ports=find_system(name,'SearchDepth',1,'BlockType','Outport');
            add_block('simulink/Sinks/Out1',path,'Port',num2str(numel(ports)+1), ...
                'Position',[970 700+k*55 1000 714+k*55]);
            add_line(name,['Frame_Unit_ReferencePoint_Conversion/' num2str(k+12)], ...
                [outputs{k} '/1'],'autorouting','on');
        end
    end
    set_param(name,'PreLoadFcn',"addpath(fileparts(get_param(bdroot,'FileName')));");
    save_system(name); close_system(name,0);
end
end

function value=endpoint(handle,model)
assert(handle~=-1,'landing:Migration','Disconnected interface wire');
parent=get_param(handle,'Parent');
value=[extractAfter(parent,[model '/']) '/' num2str(get_param(handle,'PortNumber'))];
end
